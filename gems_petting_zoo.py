import argparse, time, csv, os, random, math
import numpy as np
import torch, torch.nn as nn
import imageio.v2 as imageio

def _load_env(env_name):
    assert env_name in ["simple_spread_v3", "simple_tag_v3"]
    if env_name == "simple_spread_v3":
        try:
            from pettingzoo.mpe2 import simple_spread_v3 as simple_spread_v3
        except Exception:
            from pettingzoo.mpe import simple_spread_v3 as simple_spread_v3
        return simple_spread_v3
    else:
        try:
            from pettingzoo.mpe2 import simple_tag_v3 as simple_tag_v3
        except Exception:
            from pettingzoo.mpe import simple_tag_v3 as simple_tag_v3
        return simple_tag_v3

def _seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.zeros_(m.bias)

def viz_seed(it: int):
    return (SEED * 9973 + it * 7919) & 0x7fffffff

try:
    import psutil
    def _ram_mb():
        return psutil.Process().memory_info().rss / (1024**2)
except Exception:
    try:
        import resource, sys
        if sys.platform == "darwin":
            def _ram_mb():
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        else:
            def _ram_mb():
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        def _ram_mb():
            return float("nan")

def _mem_mb():
    try:
        return float(_ram_mb()), "rss"
    except Exception:
        return float("nan"), "n/a"

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def eta_t(eta0: float, t: int, sched: str) -> float:
    if sched == "const":
        return eta0
    if sched == "sqrt":
        return eta0 / max(1.0, math.sqrt(t))
    if sched == "harmonic":
        return eta0 / (1.0 + 0.5 * t)
    return eta0


def scale_count(base: int, grow: float, t: int) -> int:
    return max(1, int(round(base * (1.0 + grow * math.sqrt(max(1, t))))))

def empirical_sigma_rng(rng: np.random.Generator, pvec: np.ndarray, N: int):
    if N <= 1:
        return [int(rng.choice(len(pvec), p=pvec))]
    target = N * pvec
    base = np.floor(target).astype(int)
    rem = int(N - base.sum())
    if rem > 0:
        frac = target - base
        order = np.argsort(-frac + 1e-12 * np.arange(len(pvec))[::-1])
        for k in range(rem):
            base[order[k]] += 1
    seq = []
    for idx, cnt in enumerate(base.tolist()):
        seq.extend([idx] * cnt)
    if len(seq) == 0:
        seq = [int(rng.choice(len(pvec), p=pvec))]
    return seq

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="simple_tag_v3",
                   choices=["simple_spread_v3", "simple_tag_v3"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--agents", type=int, default=3,
                   help="simple_spread: #agents (homogeneous). Ignored for simple_tag.")
    p.add_argument("--zdim", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cycles", type=int, default=100)

    p.add_argument("--rollout_min_steps", type=int, default=1600)
    p.add_argument("--ppo_epochs", type=int, default=10)
    p.add_argument("--ppo_batch", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent_beta", type=float, default=1e-3)

    p.add_argument("--eta", type=float, default=0.35)
    p.add_argument("--eta_sched", type=str, default="sqrt", choices=["const","sqrt","harmonic"])
    p.add_argument("--ema", type=float, default=0.0, help="EMA(0..1) for vhat/rbar")
    p.add_argument("--stratified", type=int, default=1, help="1 = stratified marginal sampling")
    p.add_argument("--grow", type=float, default=0.0, help="sqrt(t) growth factor for MC budgets")
    p.add_argument("--mc_ni", type=int, default=8, help="episodes for vhat per iter")
    p.add_argument("--mc_B", type=int, default=16, help="episodes for rbar per iter")

    p.add_argument("--pool_mut", type=int, default=2, help="oracle: mutated samples to try")
    p.add_argument("--pool_rand", type=int, default=1, help="oracle: random samples to try")
    p.add_argument("--oracle_nz", type=int, default=1, help="oracle: how many z to add per iteration")
    p.add_argument("--oracle_m", type=int, default=1, help="oracle: eval episodes per candidate")

    p.add_argument("--log_ucb", type=int, default=0, help="1=compute EB–UCB best z per agent (costly)")
    p.add_argument("--ucb_nz", type=int, default=8)
    p.add_argument("--delta0", type=float, default=1e-3)

    p.add_argument("--csv", type=str, default="gems_results.csv")
    p.add_argument("--video", type=str, default="gems_last.gif")  # always GIF
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    p.add_argument("--continuous_actions", action="store_true",
                   help="Use continuous actions if env supports it")

    p.add_argument("--tag_adversaries", type=int, default=3, help="#taggers (adversaries)")
    p.add_argument("--tag_runners", type=int, default=1, help="#runners (good agents)")
    p.add_argument("--tag_obstacles", type=int, default=2, help="#obstacles (explicit; defaults to 2)")
    return p.parse_args()

args = parse_args()

if "DISPLAY" not in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

if args.device == "auto":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
else:
    dev = args.device
device = torch.device(dev)

SEED = args.seed
_seed_everything(SEED)
_rng = np.random.default_rng(SEED)

EnvCls = _load_env(args.env)

def make_env(render=False, mode=None):
    render_mode = mode if render else None
    if args.env == "simple_spread_v3":
        env = EnvCls.parallel_env(
            N=args.agents,
            max_cycles=args.max_cycles,
            continuous_actions=args.continuous_actions,
            render_mode=render_mode
        )
    else:
        env = EnvCls.parallel_env(
            num_good=args.tag_runners,
            num_adversaries=args.tag_adversaries,
            num_obstacles=args.tag_obstacles,
            max_cycles=args.max_cycles,
            continuous_actions=args.continuous_actions,
            render_mode=render_mode
        )
    return env

from gymnasium.spaces import Discrete, Box
_probe_env = make_env(False, None)
_init_obs, _ = _probe_env.reset(seed=SEED)
AGENT_IDS = list(_probe_env.agents)
_obs_dims = {aid: _probe_env.observation_space(aid).shape[0] for aid in AGENT_IDS}
_act_spaces = {aid: _probe_env.action_space(aid) for aid in AGENT_IDS}
_is_all_discrete = all(isinstance(_act_spaces[aid], Discrete) for aid in AGENT_IDS)
if not _is_all_discrete:
    assert all(isinstance(_act_spaces[aid], Box) for aid in AGENT_IDS), "Mixed action spaces not supported"

GOOD_IDX = [i for i,a in enumerate(AGENT_IDS) if a.startswith("agent_")]
BAD_IDX  = [i for i,a in enumerate(AGENT_IDS) if a.startswith("adversary_")]

N_AGENTS = len(AGENT_IDS)
ZDIM = args.zdim

def write_video(frames, path, fps):
    if not frames:
        return None
    gif_path = os.path.splitext(path)[0] + ".gif"
    imageio.mimsave(gif_path, frames, duration=1.0/max(fps,1))
    return gif_path

class CategoricalHead(nn.Module):
    def __init__(self, in_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, act_dim)
        )
        self.apply(_init_weights)
    def forward(self, x):
        return self.net(x)

class GaussianHead(nn.Module):
    def __init__(self, in_dim, act_dim):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.apply(_init_weights)
    def forward(self, x):
        mu = self.mu(x)
        return mu, self.logstd.expand_as(mu)

class VNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.apply(_init_weights)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class AC(nn.Module):
    def __init__(self, obs_dim, act_space, zdim):
        super().__init__()
        in_dim = obs_dim + zdim
        self.discrete = isinstance(act_space, Discrete)
        if self.discrete:
            self.pi = CategoricalHead(in_dim, act_space.n)
        else:
            self.pi = GaussianHead(in_dim, act_space.shape[0])
        self.v  = VNet(in_dim)
        self.apply(_init_weights)
    def forward(self, obs, z):
        x = torch.cat([obs, z], -1)
        if self.discrete:
            logits = self.pi(x)
            return logits, self.v(x)
        else:
            mu, logstd = self.pi(x)
            return (mu, logstd), self.v(x)

class Agent:
    def __init__(self, obs_dim, act_space, zdim):
        self.discrete = isinstance(act_space, Discrete)
        self.act_space = act_space
        self.net = AC(obs_dim, act_space, zdim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)

    @torch.no_grad()
    def act(self, obs_np, z_np):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        z   = torch.tensor(z_np,  dtype=torch.float32, device=device).unsqueeze(0)
        pi_out, v = self.net(obs, z)
        if self.discrete:
            d = torch.distributions.Categorical(logits=pi_out)
            a = d.sample()
            return a.item(), d.log_prob(a).squeeze(0), d.entropy().squeeze(0), v.squeeze(0)
        else:
            mu, logstd = pi_out
            std = logstd.exp()
            d = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
            a = d.sample()
            a_np = a.squeeze(0).cpu().numpy()
            if isinstance(self.act_space, Box):
                a_np = np.clip(a_np, self.act_space.low, self.act_space.high)
            return a_np, d.log_prob(a).squeeze(0), d.entropy().squeeze(0), v.squeeze(0)

    def evaluate(self, obs_t, z_t, act_t):
        pi_out, v = self.net(obs_t, z_t)
        if self.discrete:
            d = torch.distributions.Categorical(logits=pi_out)
        else:
            mu, logstd = pi_out
            std = logstd.exp()
            d = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        logp = d.log_prob(act_t)
        ent = d.entropy()
        return logp, ent, v

ENT_BETA = args.ent_beta; GAMMA = args.gamma; LAMBDA = args.gae_lambda
PPO_EPOCHS, PPO_BATCH = args.ppo_epochs, args.ppo_batch
MIN_STEPS_PER_ABR = args.rollout_min_steps
GEMS_ITERS = args.iters

Z = [[] for _ in range(N_AGENTS)]
LOGS = []
G_PREV = []


def init_population():
    for p in range(N_AGENTS):
        z0 = np.random.normal(0, 1, size=(ZDIM,)).astype(np.float32)
        Z[p].append(z0)
    for p in range(N_AGENTS):
        LOGS.append(np.array([0.0], dtype=np.float64))
        G_PREV.append(np.zeros(1, dtype=np.float64))

init_population()
agents = [Agent(_obs_dims[aid], _act_spaces[aid], ZDIM) for aid in AGENT_IDS]


def sigma_list():
    return [softmax(LOGS[p]) for p in range(N_AGENTS)]


def sample_profile_from(sigma_seq, rng):
    prof = []
    for p in range(N_AGENTS):
        probs = sigma_seq[p]
        idx = int(rng.choice(len(probs), p=probs))
        prof.append(idx)
    return prof


def stratified_profile_batches(N, rng):
    sig = sigma_list()
    per_agent_lists = []
    for p in range(N_AGENTS):
        per_agent_lists.append(empirical_sigma_rng(rng, sig[p], N))
    profiles = []
    for i in range(N):
        profiles.append([per_agent_lists[p][i % len(per_agent_lists[p])] for p in range(N_AGENTS)])
    return profiles

def run_episode(prof, z_override=None, render=False, seed=None, record_rgb=False):
    mode = "rgb_array" if (render or record_rgb) else None
    env = make_env(render=(render or record_rgb), mode=mode)

    obs, _ = env.reset(seed=seed if seed is not None else random.randint(0, 1<<30))
    frames = []
    rets = np.zeros(N_AGENTS, dtype=np.float32)
    done = False

    if record_rgb:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while env.agents and not done:
        acts = {}
        for aid in env.agents:
            p = AGENT_IDS.index(aid)
            z = z_override[p] if z_override is not None else Z[p][prof[p]]
            a, _, _, _ = agents[p].act(obs[aid], z)
            acts[aid] = a
        obs, r, term, trunc, _ = env.step(acts)

        if record_rgb:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        for aid, v in r.items():
            p = AGENT_IDS.index(aid)
            rets[p] += float(v)

        done = all(term.values()) or all(trunc.values())

    env.close()
    return rets, frames


def record_episode(prof, z_override, seed, path, fps):
    rets, frames = run_episode(prof, z_override, render=True, seed=seed, record_rgb=True)
    out = write_video(frames, path, fps)
    return rets, out

def meta_estimate(it):
    ni_now = scale_count(args.mc_ni, args.grow, it)
    B_now  = scale_count(args.mc_B,  args.grow, it)

    vhat = [np.zeros(len(Z[p]), dtype=np.float64) for p in range(N_AGENTS)]
    vcnt = [np.zeros(len(Z[p]), dtype=np.int64)   for p in range(N_AGENTS)]
    rbar = np.zeros(N_AGENTS, dtype=np.float64)

    if args.stratified:
        profiles = stratified_profile_batches(ni_now, _rng)
    else:
        sig = sigma_list()
        profiles = [sample_profile_from(sig, _rng) for _ in range(ni_now)]

    for prof in profiles:
        zr = [Z[p][prof[p]] for p in range(N_AGENTS)]
        rets, _ = run_episode(prof, zr, render=False)
        for p in range(N_AGENTS):
            k = prof[p]
            vhat[p][k] += rets[p]
            vcnt[p][k] += 1

    for p in range(N_AGENTS):
        vcnt_p = np.maximum(1, vcnt[p])
        vhat[p] = vhat[p] / vcnt_p

    if args.stratified:
        profiles_B = stratified_profile_batches(B_now, _rng)
    else:
        sig = sigma_list()
        profiles_B = [sample_profile_from(sig, _rng) for _ in range(B_now)]

    for prof in profiles_B:
        zr = [Z[p][prof[p]] for p in range(N_AGENTS)]
        rets, _ = run_episode(prof, zr, render=False)
        rbar += rets
    rbar /= max(1, B_now)

    return vhat, rbar

_VHAT_EMA = None
_RBAR_EMA = None


def ema_blend(vhat, rbar):
    global _VHAT_EMA, _RBAR_EMA
    if args.ema <= 0.0:
        return vhat, rbar
    if _VHAT_EMA is None:
        _VHAT_EMA = [v.copy() for v in vhat]
        _RBAR_EMA = rbar.copy()
        return vhat, rbar
    beta = args.ema
    out_v = []
    for p in range(N_AGENTS):
        if _VHAT_EMA[p].shape[0] != len(Z[p]):
            old = _VHAT_EMA[p]
            add = len(Z[p]) - old.shape[0]
            if add > 0:
                pad = np.full(add, old[-1] if old.size > 0 else 0.0, dtype=np.float64)
                _VHAT_EMA[p] = np.concatenate([old, pad], axis=0)
        v = (1 - beta) * _VHAT_EMA[p] + beta * vhat[p]
        out_v.append(v)
        _VHAT_EMA[p] = v.copy()
    _RBAR_EMA = (1 - beta) * _RBAR_EMA + beta * rbar
    return out_v, _RBAR_EMA.copy()


def mwu_update_omwu(vhat, rbar, it):
    eta_now = eta_t(args.eta, it, args.eta_sched)
    for p in range(N_AGENTS):
        if LOGS[p].shape[0] != len(Z[p]):
            add = len(Z[p]) - LOGS[p].shape[0]
            if add > 0:
                new_logits = np.full(add, LOGS[p].min() - 5.0, dtype=np.float64)
                LOGS[p] = np.concatenate([LOGS[p], new_logits], axis=0)
                G_PREV[p] = np.concatenate([G_PREV[p], np.zeros(add, dtype=np.float64)], axis=0)

        gains = np.array(vhat[p], dtype=np.float64) - float(rbar[p])
        grad_eff = 2.0 * gains - G_PREV[p]
        LOGS[p] = LOGS[p] + eta_now * grad_eff
        G_PREV[p] = gains

def oracle_select(p, it):
    base = Z[p][-1]
    cand = []
    for _ in range(args.pool_mut):
        noise = np.random.normal(0, 0.25, size=(ZDIM,)).astype(np.float32)
        cand.append(base + noise)
    for _ in range(args.pool_rand):
        cand.append(np.random.normal(0, 1, size=(ZDIM,)).astype(np.float32))

    scores = []
    sig = sigma_list()
    for zc in cand:
        s_acc = 0.0
        for _ in range(args.oracle_m):
            prof = sample_profile_from(sig, _rng)
            zr = [zc if q == p else Z[q][prof[q]] for q in range(N_AGENTS)]
            rets, _ = run_episode(prof, zr, render=False)
            s_acc += rets[p]
        scores.append(s_acc / max(1, args.oracle_m))

    order = np.argsort(scores)[::-1]
    add_n = min(args.oracle_nz, len(order))
    if add_n > 0:
        for j in range(add_n):
            Z[p].append(cand[order[j]].copy())
        add = add_n
        new_logits = np.full(add, LOGS[p].min() - 5.0, dtype=np.float64)
        LOGS[p] = np.concatenate([LOGS[p], new_logits], axis=0)
        G_PREV[p] = np.concatenate([G_PREV[p], np.zeros(add, dtype=np.float64)], axis=0)

def collect_rollouts(p, z_anchor):
    O, Zs, A, LP, R, ADV = [], [], [], [], [], []
    steps = 0
    while steps < MIN_STEPS_PER_ABR:
        sig = sigma_list()
        prof = sample_profile_from(sig, _rng)
        env = make_env(False, None)
        obs, _ = env.reset(seed=random.randint(0, 1<<30))
        traj = []
        done = False
        while env.agents and not done:
            acts = {}
            for aid in env.agents:
                i = AGENT_IDS.index(aid)
                z = z_anchor if i == p else Z[i][prof[i]]
                a, lp, _, v = agents[i].act(obs[aid], z)
                acts[aid] = a
                if i == p:
                    traj.append([obs[aid], z, a, lp.item(), v.item(), 0.0])
            obs, r, term, trunc, _ = env.step(acts)
            if traj:
                traj[-1][5] = float(r[AGENT_IDS[p]])
            done = all(term.values()) or all(trunc.values())
        env.close()

        vals = [x[4] for x in traj] + [0.0]
        rews = [x[5] for x in traj]
        advs, G = [], 0.0
        for t in reversed(range(len(rews))):
            delta = rews[t] + GAMMA * vals[t+1] - vals[t]
            G = delta + GAMMA * LAMBDA * G
            advs.append(G)
        advs = list(reversed(advs))
        rets = [advs[t] + vals[t] for t in range(len(rews))]

        for (o,z,a,lp,_v,_r), R_t, Adv in zip(traj, rets, advs):
            O.append(o); Zs.append(z); A.append(a); LP.append(lp); R.append(R_t); ADV.append(Adv)
        steps += len(traj)

    O = torch.tensor(np.array(O), dtype=torch.float32, device=device)
    Zs= torch.tensor(np.array(Zs), dtype=torch.float32, device=device)
    if agents[p].discrete:
        A = torch.tensor(np.array(A), dtype=torch.long, device=device)
    else:
        A = torch.tensor(np.array(A), dtype=torch.float32, device=device)
    LP= torch.tensor(np.array(LP), dtype=torch.float32, device=device)
    R = torch.tensor(np.array(R), dtype=torch.float32, device=device)
    ADV = torch.tensor(np.array(ADV), dtype=torch.float32, device=device)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    return (O, Zs, A, LP, R, ADV)


def ppo_update(p, batch):
    O, Zs, A, LP_old, R_t, ADV = batch
    N = O.shape[0]
    idx = np.arange(N)
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idx)
        for j in range(0, N, PPO_BATCH):
            jj = idx[j:j+PPO_BATCH]
            obs_t = O[jj]; z_t = Zs[jj]; act_t = A[jj]
            logp, ent, val = agents[p].evaluate(obs_t, z_t, act_t)
            ratio = torch.exp(logp - LP_old[jj])
            s1 = ratio * ADV[jj]
            s2 = torch.clamp(ratio, 1.0-args.clip, 1.0+args.clip) * ADV[jj]
            loss = -torch.min(s1,s2).mean() - ENT_BETA * ent.mean() + 0.5 * (R_t[jj]-val).pow(2).mean()
            agents[p].opt.zero_grad(set_to_none=True); loss.backward(); agents[p].opt.step()

def eb_ucb_best_index_for_agent(p, t):
    if args.log_ucb == 0:
        return -1, float("nan")
    n = max(1, args.ucb_nz)
    delta_t = max(1e-12, args.delta0 * (t ** -2))
    sig = sigma_list()
    best_idx, best_score = -1, -1e18
    for k in range(len(Z[p])):
        vals = []
        for _ in range(n):
            prof = sample_profile_from(sig, _rng)
            zr = [Z[q][prof[q]] if q != p else Z[p][k] for q in range(N_AGENTS)]
            rets, _ = run_episode(prof, zr, render=False)
            vals.append(float(rets[p]))
        vals = np.array(vals, dtype=np.float64)
        mu = float(vals.mean())
        var = float(vals.var(ddof=1)) if vals.shape[0] > 1 else 0.0
        ln = math.log(max(1.0000001, 3.0 / delta_t))
        rad = math.sqrt(2.0 * var * ln / n) + (3.0 * ln / max(1, n - 1))
        ucb = mu + rad
        if ucb > best_score:
            best_score, best_idx = ucb, k
    return best_idx, best_score

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)

print(f"[GEMS] env={args.env} agents={len(AGENT_IDS)} device={device.type}" +
      (f" gpu={torch.cuda.get_device_name(0)}" if device.type=='cuda' else ""))

with open(args.csv, "w", newline="") as f:
    w = csv.writer(f)

    base_header = [
        "iter","timestamp","time_sec","mem_mb","mem_type"
    ] + [f"ret_{i}" for i in range(len(AGENT_IDS))] + [
        "ret_mean","ret_sum","pop_sizes","video_path"
    ]

    if args.env == "simple_tag_v3":
        header = base_header[:-2] + ["good_avg","bad_avg","good_sum","bad_sum"] + base_header[-2:]
    else:
        header = base_header

    if args.log_ucb:
        header = header[:-2] + [f"ucb_best_idx_p{p}" for p in range(N_AGENTS)] + header[-2:]

    w.writerow(header)

    for it in range(1, GEMS_ITERS + 1):
        t0 = time.time()

        vhat, rbar = meta_estimate(it)
        vhat, rbar = ema_blend(vhat, rbar)

        mwu_update_omwu(vhat, rbar, it)

        for p in range(N_AGENTS):
            oracle_select(p, it)

        for p in range(N_AGENTS):
            batch = collect_rollouts(p, Z[p][-1])
            ppo_update(p, batch)

        prof = [len(Z[p]) - 1 for p in range(N_AGENTS)]
        s_eval = viz_seed(it)
        rets, _ = run_episode(prof, [Z[q][prof[q]] for q in range(N_AGENTS)], render=False, seed=s_eval)

        dt = time.time() - t0
        mem, mtype = _mem_mb()
        overall_mean = float(np.mean(rets))
        overall_sum  = float(np.sum(rets))

        if args.env == "simple_tag_v3":
            good_avg = float(np.mean(rets[GOOD_IDX])) if GOOD_IDX else float('nan')
            bad_avg  = float(np.mean(rets[BAD_IDX]))  if BAD_IDX  else float('nan')
            good_sum = float(np.sum(rets[GOOD_IDX]))  if GOOD_IDX else float('nan')
            bad_sum  = float(np.sum(rets[BAD_IDX]))   if BAD_IDX  else float('nan')

            print(f"[GEMS] iter {it}/{GEMS_ITERS} time={dt:.2f}s "
                  f"good_avg={good_avg:.2f} bad_avg={bad_avg:.2f} "
                  f"(overall mean={overall_mean:.2f} sum={overall_sum:.2f}) "
                  f"pop={ [len(Z[p]) for p in range(N_AGENTS)] }")
        else:
            print(f"[GEMS] iter {it}/{GEMS_ITERS} time={dt:.2f}s "
                  f"mean={overall_mean:.2f} sum={overall_sum:.2f} "
                  f"pop={ [len(Z[p]) for p in range(N_AGENTS)] }")

        row = [it, time.strftime("%Y-%m-%d %H:%M:%S"), f"{dt:.3f}", f"{mem:.2f}", mtype] + \
              [f"{r:.3f}" for r in rets.tolist()] + \
              [f"{overall_mean:.3f}", f"{overall_sum:.3f}", str([len(Z[p]) for p in range(N_AGENTS)]), ""]

        if args.env == "simple_tag_v3":
            row = row[:-2] + [f"{good_avg:.3f}", f"{bad_avg:.3f}", f"{good_sum:.3f}", f"{bad_sum:.3f}"] + row[-2:]

        if args.log_ucb:
            ucb_cols = []
            for p in range(N_AGENTS):
                idx, score = eb_ucb_best_index_for_agent(p, it)
                ucb_cols.append(str(idx))
            row = row[:-2] + ucb_cols + row[-2:]

        w.writerow(row); f.flush()

    print("[GEMS] recording last iteration...")
    seed_for_record = viz_seed(GEMS_ITERS)
    prof = [len(Z[p]) - 1 for p in range(N_AGENTS)]
    zov = [Z[q][prof[q]] for q in range(N_AGENTS)]
    rets, vpath = record_episode(prof, zov, seed_for_record, args.video, args.fps)
    with open(args.csv, "a", newline="") as f2:
        w2 = csv.writer(f2)
        mem, mtype = _mem_mb()
        overall_mean = float(np.mean(rets))
        overall_sum  = float(np.sum(rets))
        if args.env == "simple_tag_v3":
            good_avg = float(np.mean(rets[GOOD_IDX])) if GOOD_IDX else float('nan')
            bad_avg  = float(np.mean(rets[BAD_IDX]))  if BAD_IDX  else float('nan')
            good_sum = float(np.sum(rets[GOOD_IDX]))  if GOOD_IDX else float('nan')
            bad_sum  = float(np.sum(rets[BAD_IDX]))   if BAD_IDX  else float('nan')
            row = [GEMS_ITERS, time.strftime("%Y-%m-%d %H:%M:%S"),
                   f"0.000", f"{mem:.2f}", mtype] + \
                  [f"{r:.3f}" for r in rets.tolist()] + \
                  [f"{overall_mean:.3f}", f"{overall_sum:.3f}",
                   f"{good_avg:.3f}", f"{bad_avg:.3f}", f"{good_sum:.3f}", f"{bad_sum:.3f}",
                   str([len(Z[p]) for p in range(N_AGENTS)]), vpath or ""]
        else:
            row = [GEMS_ITERS, time.strftime("%Y-%m-%d %H:%M:%S"),
                   f"0.000", f"{mem:.2f}", mtype] + \
                  [f"{r:.3f}" for r in rets.tolist()] + \
                  [f"{overall_mean:.3f}", f"{overall_sum:.3f}",
                   str([len(Z[p]) for p in range(N_AGENTS)]), vpath or ""]
        w2.writerow(row)
    print(f"[GEMS] saved video at: {vpath}")

print("done")
