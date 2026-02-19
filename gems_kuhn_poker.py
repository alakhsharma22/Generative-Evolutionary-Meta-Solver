from __future__ import annotations
import argparse, csv, glob, math, os, random, sys, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import ttest_ind
except Exception:
    ttest_ind = None



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def process_mem_mb() -> Tuple[float, str]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024**2), "rss"
    except Exception:
        try:
            import resource
            if sys.platform == "darwin":
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), "ru_maxrss"
            else:
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "ru_maxrss"
        except Exception:
            return float("nan"), "n/a"

def read_last_col(csv_path: str, col: str) -> float:
    with open(csv_path, "r") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    header = rows[0]
    if col not in header:
        raise ValueError(f"Column '{col}' not found in {csv_path}. Header: {header}")
    idx = header.index(col)
    return float(rows[-1][idx])

def welch_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if ttest_ind is None:
        return float("nan"), float("nan")
    t, p = ttest_ind(a, b, equal_var=False)
    return float(t), float(p)



CARDS = [0, 1, 2]

def ev_p1_vs(p1: np.ndarray, p2: np.ndarray) -> float:
    b1, c1 = p1[:3], p1[3:]
    c2, b2 = p2[:3], p2[3:]
    ev = 0.0
    for c1i in CARDS:
        for c2i in CARDS:
            if c1i == c2i:
                continue
            s4 = 2.0 if c1i > c2i else -2.0
            s2 = 1.0 if c1i > c2i else -1.0
            term_B = b1[c1i] * (c2[c2i]*s4 + (1.0-c2[c2i])*1.0)
            term_C = (1.0-b1[c1i]) * (
                b2[c2i]*(c1[c1i]*s4 + (1.0-c1[c1i])*(-1.0)) + (1.0-b2[c2i])*s2
            )
            ev += term_B + term_C
    return ev / 6.0

def enumerate_pure_policies_p1():
    out = []
    for mask in range(64):
        b = [(mask >> 0) & 1, (mask >> 1) & 1, (mask >> 2) & 1]
        c = [(mask >> 3) & 1, (mask >> 4) & 1, (mask >> 5) & 1]
        out.append(np.array(b + c, dtype=np.float64))
    return out

def enumerate_pure_policies_p2():
    out = []
    for mask in range(64):
        c = [(mask >> 0) & 1, (mask >> 1) & 1, (mask >> 2) & 1]
        b = [(mask >> 3) & 1, (mask >> 4) & 1, (mask >> 5) & 1]
        out.append(np.array(c + b, dtype=np.float64))
    return out

PURE1 = enumerate_pure_policies_p1()
PURE2 = enumerate_pure_policies_p2()



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, zdim: int, hidden: int = 64, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def sigmoid_probs(logits: torch.Tensor, tau: float, eps: float) -> torch.Tensor:
    p = torch.sigmoid(logits / max(1e-8, tau))
    p = torch.nan_to_num(p, nan=0.5)
    return torch.clamp(p, eps, 1.0 - eps)

def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))



@dataclass
class Args:
    iters: int = 40
    k0: int = 1
    kmax: int = 32

    zdim: int = 8
    tau: float = 1.0

    mc_n: int = 8
    mc_m: int = 2
    mc_B: int = 128
    ema: float = 0.0

    oracle_n: int = 8
    oracle_m: int = 2
    cand_mut: int = 32
    cand_rand: int = 32
    mut_sigma: float = 0.2
    delta0: float = 0.5

    eta: float = 0.03
    eta_sched: str = "const"
    logit_cap: float = 50.0

    abr_steps: int = 30
    abr_batch_z: int = 16
    abr_lr: float = 2e-4
    beta_kl: float = 0.05
    q_new_frac: float = 0.25
    clip_grad: float = 0.5

    eval_every: int = 1

    outdir: str = "results"
    seeds: str = "0,1,2,3,4"
    device: str = "auto"
    no_plots: bool = False
    ttest_against_glob: Optional[str] = None


def parse_args() -> Args:
    p = argparse.ArgumentParser("GEMS on Kuhn Poker — multi-seed")

    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--k0", type=int, default=1)
    p.add_argument("--kmax", type=int, default=32)

    p.add_argument("--zdim", type=int, default=8)
    p.add_argument("--tau", type=float, default=1.0)

    p.add_argument("--mc_n", type=int, default=8)
    p.add_argument("--mc_m", type=int, default=2)
    p.add_argument("--mc_B", type=int, default=128)
    p.add_argument("--ema", type=float, default=0.0)

    p.add_argument("--oracle_n", type=int, default=8)
    p.add_argument("--oracle_m", type=int, default=2)
    p.add_argument("--cand_mut", type=int, default=32)
    p.add_argument("--cand_rand", type=int, default=32)
    p.add_argument("--mut_sigma", type=float, default=0.2)
    p.add_argument("--delta0", type=float, default=0.5)

    p.add_argument("--eta", type=float, default=0.03)
    p.add_argument("--eta_sched", choices=["const", "sqrt", "harmonic"], default="const")
    p.add_argument("--logit_cap", type=float, default=50.0)

    p.add_argument("--abr_steps", type=int, default=30)
    p.add_argument("--abr_batch_z", type=int, default=16)
    p.add_argument("--abr_lr", type=float, default=2e-4)
    p.add_argument("--beta_kl", type=float, default=0.05)
    p.add_argument("--q_new_frac", type=float, default=0.25)
    p.add_argument("--clip_grad", type=float, default=0.5)

    p.add_argument("--eval_every", type=int, default=1)

    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--ttest_against_glob", type=str, default=None)

    return Args(**vars(p.parse_args()))



def safe_np_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(p, dtype=np.float64) / max(1, p.size)
    p = p / s

    p[-1] = 1.0 - float(p[:-1].sum())
    if p[-1] < 0.0:
        p = np.clip(p, 0.0, None)
        p = p / float(p.sum())
    return p


class GEMSRunner:
    def __init__(self, args: Args, seed: int, device: torch.device):
        self.args = args
        self.seed = seed
        self.device = device

        seed_everything(seed)

        self.gen = Generator(args.zdim).to(device)
        self.opt = torch.optim.Adam(self.gen.parameters(), lr=args.abr_lr)

        self.Z1 = torch.randn(args.k0, args.zdim, device=device)
        self.Z2 = torch.randn(args.k0, args.zdim, device=device)

        self.L1 = torch.zeros(args.k0, dtype=torch.float32, device=device)
        self.L2 = torch.zeros(args.k0, dtype=torch.float32, device=device)
        self.prev_g1 = torch.zeros_like(self.L1)
        self.prev_g2 = torch.zeros_like(self.L2)

        self._VV = None
        self._RR = None

        self.ev_calls = 0

    def _sigma(self):
        cap = float(self.args.logit_cap)
        l1 = torch.clamp(self.L1, -cap, cap)
        l2 = torch.clamp(self.L2, -cap, cap)
        return torch.softmax(l1, 0), torch.softmax(l2, 0)

    @torch.no_grad()
    def _policy_probs(self):
        P1 = sigmoid_probs(self.gen(self.Z1), self.args.tau, 1e-6)
        P2 = sigmoid_probs(self.gen(self.Z2), self.args.tau, 1e-6)
        return P1, P2

    def _simulate_episode(self, p1: np.ndarray, p2: np.ndarray) -> float:
        c = [0, 1, 2]
        random.shuffle(c)
        c1, c2 = c[0], c[1]
        b1 = p1[c1]
        c2_call = p2[c2]
        c1_call = p1[3 + c1]
        b2 = p2[3 + c2]

        if random.random() < b1:
            if random.random() < c2_call:
                return 2.0 if c1 > c2 else -2.0
            return 1.0

        if random.random() < b2:
            if random.random() < c1_call:
                return 2.0 if c1 > c2 else -2.0
            return -1.0

        return 1.0 if c1 > c2 else -1.0

    def mc_estimate_v_r(self):
        P1_t, P2_t = self._policy_probs()
        s1, s2 = self._sigma()

        P1 = P1_t.detach().cpu().numpy().astype(np.float64)
        P2 = P2_t.detach().cpu().numpy().astype(np.float64)
        s1n = safe_np_probs(s1.detach().cpu().numpy())
        s2n = safe_np_probs(s2.detach().cpu().numpy())

        K1, K2 = P1.shape[0], P2.shape[0]
        v1 = np.zeros(K1, dtype=np.float64)
        v2 = np.zeros(K2, dtype=np.float64)
        r = np.zeros(1, dtype=np.float64)

        for i in range(K1):
            acc = 0.0
            for _ in range(self.args.mc_n):
                j = int(np.random.choice(K2, p=s2n))
                for _ in range(self.args.mc_m):
                    self.ev_calls += 1
                    acc += self._simulate_episode(P1[i], P2[j])
            v1[i] = acc / (self.args.mc_n * self.args.mc_m)

        for j in range(K2):
            acc = 0.0
            for _ in range(self.args.mc_n):
                i = int(np.random.choice(K1, p=s1n))
                for _ in range(self.args.mc_m):
                    self.ev_calls += 1
                    acc += self._simulate_episode(P1[i], P2[j])
            v2[j] = acc / (self.args.mc_n * self.args.mc_m)

        acc = 0.0
        for _ in range(self.args.mc_B):
            i = int(np.random.choice(K1, p=s1n))
            j = int(np.random.choice(K2, p=s2n))
            self.ev_calls += 1
            acc += self._simulate_episode(P1[i], P2[j])
        r[0] = acc / self.args.mc_B

        if self.args.ema > 0.0:
            beta = float(self.args.ema)
            if self._VV is None:
                self._VV = [v1.copy(), v2.copy()]
                self._RR = r.copy()
            else:
                for pidx, vcur in enumerate([v1, v2]):
                    if self._VV[pidx].shape != vcur.shape:
                        new = np.zeros_like(vcur)
                        m = min(self._VV[pidx].shape[0], vcur.shape[0])
                        new[:m] = self._VV[pidx][:m]
                        if vcur.shape[0] > m:
                            new[m:] = vcur[m:]
                        self._VV[pidx] = new

            self._VV[0] = (1 - beta) * self._VV[0] + beta * v1
            self._VV[1] = (1 - beta) * self._VV[1] + beta * v2
            self._RR = (1 - beta) * self._RR + beta * r
            return [self._VV[0].copy(), self._VV[1].copy()], self._RR.copy()

        return [v1, v2], r

    def _eta(self, t: int) -> float:
        if self.args.eta_sched == "const":
            return float(self.args.eta)
        if self.args.eta_sched == "sqrt":
            return float(self.args.eta) / max(1.0, math.sqrt(t))
        if self.args.eta_sched == "harmonic":
            return float(self.args.eta) / (1.0 + 0.5 * t)
        return float(self.args.eta)

    def omwu_update(self, t: int, v: List[np.ndarray]):
        eta = self._eta(t)
        s1, s2 = self._sigma()
        g1 = torch.tensor(v[0], dtype=torch.float32, device=self.device)
        g2 = torch.tensor(-v[1], dtype=torch.float32, device=self.device)

        if self.prev_g1.shape[0] != g1.shape[0]:
            new = torch.zeros_like(g1)
            m = min(self.prev_g1.shape[0], g1.shape[0])
            new[:m] = self.prev_g1[:m]
            self.prev_g1 = new
        if self.prev_g2.shape[0] != g2.shape[0]:
            new = torch.zeros_like(g2)
            m = min(self.prev_g2.shape[0], g2.shape[0])
            new[:m] = self.prev_g2[:m]
            self.prev_g2 = new

        upd1 = 2 * g1 - self.prev_g1
        upd2 = 2 * g2 - self.prev_g2
        self.prev_g1 = g1.detach()
        self.prev_g2 = g2.detach()

        l1 = torch.log(torch.clamp(s1, 1e-12, 1.0))
        l2 = torch.log(torch.clamp(s2, 1e-12, 1.0))
        self.L1 = l1 + eta * upd1
        self.L2 = l2 + eta * upd2

    def abr_tr_train(self):
        self.gen.train()
        s1, s2 = self._sigma()

        K1, K2 = self.Z1.shape[0], self.Z2.shape[0]
        q_new = float(self.args.q_new_frac)

        def sample_anchor_idx(K: int, sigma: torch.Tensor, batch: int):
            sig = safe_np_probs(sigma.detach().cpu().numpy())
            if K == 1:
                return np.zeros(batch, dtype=np.int64)
            mix = (1 - q_new) * sig
            mix[-1] += q_new
            mix = safe_np_probs(mix)
            return np.random.choice(K, size=batch, p=mix)

        idx1 = sample_anchor_idx(K1, s1, self.args.abr_batch_z)
        idx2 = sample_anchor_idx(K2, s2, self.args.abr_batch_z)

        with torch.no_grad():
            prev1 = sigmoid_probs(self.gen(self.Z1[idx1]), self.args.tau, 1e-6)
            prev2 = sigmoid_probs(self.gen(self.Z2[idx2]), self.args.tau, 1e-6)

        def ev_torch(p1v: torch.Tensor, p2v: torch.Tensor) -> torch.Tensor:
            b1, c1 = p1v[:3], p1v[3:]
            c2, b2 = p2v[:3], p2v[3:]
            ev = torch.zeros([], dtype=torch.float32, device=self.device)
            for c1i in CARDS:
                for c2i in CARDS:
                    if c1i == c2i:
                        continue
                    s4 = 2.0 if c1i > c2i else -2.0
                    s2v = 1.0 if c1i > c2i else -1.0
                    term_B = b1[c1i] * (c2[c2i]*s4 + (1.0-c2[c2i])*1.0)
                    term_C = (1.0-b1[c1i]) * (
                        b2[c2i]*(c1[c1i]*s4 + (1.0-c1[c1i])*(-1.0)) + (1.0-b2[c2i])*s2v
                    )
                    ev = ev + term_B + term_C
            return ev / 6.0

        for _ in range(self.args.abr_steps):
            p1 = sigmoid_probs(self.gen(self.Z1[idx1]), self.args.tau, 1e-6)
            p2 = sigmoid_probs(self.gen(self.Z2[idx2]), self.args.tau, 1e-6)

            with torch.no_grad():
                P1_all = sigmoid_probs(self.gen(self.Z1), self.args.tau, 1e-6)
                P2_all = sigmoid_probs(self.gen(self.Z2), self.args.tau, 1e-6)
                mix1 = (s1[:, None] * P1_all).sum(0)
                mix2 = (s2[:, None] * P2_all).sum(0)

            ev1 = torch.stack([ev_torch(p1[b], mix2) for b in range(p1.shape[0])], 0).mean()
            ev2 = torch.stack([ev_torch(mix1, p2[b]) for b in range(p2.shape[0])], 0).mean()

            obj = ev1 - ev2
            loss = -obj

            if self.args.beta_kl > 0.0:
                kl1 = bernoulli_kl(p1, prev1, eps=1e-6).mean()
                kl2 = bernoulli_kl(p2, prev2, eps=1e-6).mean()
                loss = loss + float(self.args.beta_kl) * (kl1 + kl2)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.clip_grad and self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), float(self.args.clip_grad))
            self.opt.step()

        self.gen.eval()

    def eb_ucb_oracle(self, role: int):
        P1_t, P2_t = self._policy_probs()
        s1, s2 = self._sigma()

        P1 = P1_t.detach().cpu().numpy().astype(np.float64)
        P2 = P2_t.detach().cpu().numpy().astype(np.float64)
        s1n = safe_np_probs(s1.detach().cpu().numpy())
        s2n = safe_np_probs(s2.detach().cpu().numpy())

        def eb_width(var: float, n: int, delta: float) -> float:
            n = max(1, int(n))
            delta = max(1e-12, float(delta))
            return math.sqrt(2.0 * var * math.log(3.0 / delta) / n) + 3.0 * math.log(3.0 / delta) / max(1, n - 1)

        if role == 0:
            Z = self.Z1
            K = Z.shape[0]
            mut = Z[torch.randint(0, K, (self.args.cand_mut,), device=self.device)] \
                  + self.args.mut_sigma * torch.randn(self.args.cand_mut, self.args.zdim, device=self.device)
            rnd = torch.randn(self.args.cand_rand, self.args.zdim, device=self.device)
            C = torch.cat([mut, rnd], 0)

            scores = []
            for zc in C:
                with torch.no_grad():
                    pc = sigmoid_probs(self.gen(zc.unsqueeze(0)), self.args.tau, 1e-6)[0].detach().cpu().numpy().astype(np.float64)
                X = []
                for _ in range(self.args.oracle_n):
                    j = int(np.random.choice(P2.shape[0], p=s2n))
                    for _ in range(self.args.oracle_m):
                        self.ev_calls += 1
                        X.append(self._simulate_episode(pc, P2[j]))
                X = np.array(X, dtype=np.float64)
                mu = float(X.mean()) if X.size > 0 else 0.0
                var = float(X.var(ddof=1)) if X.size > 1 else 0.0
                delta = float(self.args.delta0) / max(2.0, K + 1.0)
                scores.append(mu + eb_width(var, X.size, delta))

            best = int(np.argmax(np.array(scores)))
            self.Z1 = torch.cat([self.Z1, C[best].unsqueeze(0)], 0)
            self.L1 = torch.cat([self.L1, torch.tensor([0.0], device=self.device)], 0)
            self.prev_g1 = torch.cat([self.prev_g1, torch.tensor([0.0], device=self.device)], 0)

        else:
            Z = self.Z2
            K = Z.shape[0]
            mut = Z[torch.randint(0, K, (self.args.cand_mut,), device=self.device)] \
                  + self.args.mut_sigma * torch.randn(self.args.cand_mut, self.args.zdim, device=self.device)
            rnd = torch.randn(self.args.cand_rand, self.args.zdim, device=self.device)
            C = torch.cat([mut, rnd], 0)

            scores = []
            for zc in C:
                with torch.no_grad():
                    pc = sigmoid_probs(self.gen(zc.unsqueeze(0)), self.args.tau, 1e-6)[0].detach().cpu().numpy().astype(np.float64)
                X = []
                for _ in range(self.args.oracle_n):
                    i = int(np.random.choice(P1.shape[0], p=s1n))
                    for _ in range(self.args.oracle_m):
                        self.ev_calls += 1
                        X.append(self._simulate_episode(P1[i], pc))
                X = np.array(X, dtype=np.float64)
                mu = float(X.mean()) if X.size > 0 else 0.0
                var = float(X.var(ddof=1)) if X.size > 1 else 0.0
                delta = float(self.args.delta0) / max(2.0, K + 1.0)
                scores.append(mu + eb_width(var, X.size, delta))

            best = int(np.argmin(np.array(scores)))
            self.Z2 = torch.cat([self.Z2, C[best].unsqueeze(0)], 0)
            self.L2 = torch.cat([self.L2, torch.tensor([0.0], device=self.device)], 0)
            self.prev_g2 = torch.cat([self.prev_g2, torch.tensor([0.0], device=self.device)], 0)

    def nashconv(self) -> Tuple[float, float]:
        P1_t, P2_t = self._policy_probs()
        s1, s2 = self._sigma()

        P1 = P1_t.detach().cpu().numpy().astype(np.float64)
        P2 = P2_t.detach().cpu().numpy().astype(np.float64)
        s1n = s1.detach().cpu().numpy().astype(np.float64)
        s2n = s2.detach().cpu().numpy().astype(np.float64)

        mix1 = (s1n[:, None] * P1).sum(0)
        mix2 = (s2n[:, None] * P2).sum(0)

        val = ev_p1_vs(mix1, mix2)
        br1 = max(ev_p1_vs(pi, mix2) for pi in PURE1)
        br2min = min(ev_p1_vs(mix1, pj) for pj in PURE2)
        return max(0.0, br1 - br2min), val

    def step(self, t: int) -> Tuple[float, float, float, float, int, int, str]:
        t0 = time.time()

        v, r = self.mc_estimate_v_r()

        self.omwu_update(t, v)

        self.abr_tr_train()

        if self.Z1.shape[0] < self.args.kmax:
            self.eb_ucb_oracle(role=0)
        if self.Z2.shape[0] < self.args.kmax:
            self.eb_ucb_oracle(role=1)

        if (t % self.args.eval_every) == 0:
            nc, val = self.nashconv()
        else:
            nc, val = float("nan"), float("nan")

        dt = time.time() - t0
        mem_mb, mem_type = process_mem_mb()
        K1 = int(self.Z1.shape[0])
        K2 = int(self.Z2.shape[0])
        mix_ev = float(val)

        return dt, float(mem_mb), mix_ev, float(nc), K1, K2, mem_type



def run_seed(args: Args, seed: int, device: torch.device, per_seed_csv: str) -> Tuple[np.ndarray, str]:
    runner = GEMSRunner(args, seed=seed, device=device)
    os.makedirs(os.path.dirname(per_seed_csv) or ".", exist_ok=True)

    hist = np.zeros((args.iters, 6), dtype=np.float64)
    mem_type_last = "n/a"

    with open(per_seed_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "dt", "mem_mb", "mix_ev", "nashconv", "n_strats_p1", "n_strats_p2", "mem_type"])
        for t in range(1, args.iters + 1):
            dt, mem_mb, mix_ev, nc, k1, k2, mem_type = runner.step(t)
            mem_type_last = mem_type
            hist[t - 1] = [dt, mem_mb, mix_ev, nc, k1, k2]

            print(f"[GEMS] seed={seed} iter {t}/{args.iters} | "
                  f"K1={k1} K2={k2} | NashConv={nc:.6f} mixEV={mix_ev:+.6f} | "
                  f"{dt:.2f}s mem={mem_mb:.1f}MB")

            w.writerow([t,
                        f"{dt:.6f}",
                        f"{mem_mb:.6f}",
                        f"{mix_ev:+.6f}",
                        f"{nc:.6f}",
                        k1, k2,
                        mem_type])
            f.flush()

    return hist, mem_type_last


def aggregate_and_write(out_csv: str, H: np.ndarray, mem_type: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    dt = H[:, :, 0]
    mem = H[:, :, 1]
    mix_ev = H[:, :, 2]
    nc = H[:, :, 3]
    k1 = H[:, :, 4]
    k2 = H[:, :, 5]

    def mean_std(x):
        return x.mean(0), x.std(0, ddof=1) if x.shape[0] > 1 else (x.mean(0), np.zeros_like(x.mean(0)))

    dt_m, dt_s = mean_std(dt)
    mem_m, mem_s = mean_std(mem)
    mix_m, mix_s = mean_std(mix_ev)
    nc_m, nc_s = mean_std(nc)
    k1_m, k1_s = mean_std(k1)
    k2_m, k2_s = mean_std(k2)

    T = H.shape[1]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter",
                    "dt_mean", "dt_std",
                    "mem_mb_mean", "mem_mb_std",
                    "mix_ev_mean", "mix_ev_std",
                    "nashconv_mean", "nashconv_std",
                    "n_strats_p1_mean", "n_strats_p1_std",
                    "n_strats_p2_mean", "n_strats_p2_std",
                    "mem_type"])
        for t in range(T):
            w.writerow([t + 1,
                        f"{dt_m[t]:.6f}", f"{dt_s[t]:.6f}",
                        f"{mem_m[t]:.6f}", f"{mem_s[t]:.6f}",
                        f"{mix_m[t]:+.6f}", f"{mix_s[t]:.6f}",
                        f"{nc_m[t]:.6f}", f"{nc_s[t]:.6f}",
                        f"{k1_m[t]:.6f}", f"{k1_s[t]:.6f}",
                        f"{k2_m[t]:.6f}", f"{k2_s[t]:.6f}",
                        mem_type
                        ])


def main():
    args = parse_args()
    device = pick_device(args.device)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    os.makedirs(args.outdir, exist_ok=True)

    per_seed_hist = []
    mem_type = "n/a"
    seed_csvs = []

    for s in seeds:
        csv_path = os.path.join(args.outdir, f"gems_seed{s}.csv")
        seed_csvs.append(csv_path)
        hist, mtype = run_seed(args, seed=s, device=device, per_seed_csv=csv_path)
        per_seed_hist.append(hist)
        mem_type = mtype

    H = np.stack(per_seed_hist, 0)
    agg_path = os.path.join(args.outdir, "gems_meanstd.csv")
    aggregate_and_write(agg_path, H, mem_type)
    print(f"[AGG] wrote: {agg_path}")

    if not args.no_plots:
        try:
            import pandas as pd
            df = pd.read_csv(agg_path)
            it = df["iter"].values
            nc_m = df["nashconv_mean"].values
            nc_s = df["nashconv_std"].values
            plt.figure()
            plt.plot(it, nc_m)
            plt.fill_between(it, nc_m - nc_s, nc_m + nc_s, alpha=0.2)
            plt.xlabel("Iteration")
            plt.ylabel("NashConv")
            plt.title("GEMS on Kuhn Poker (mean ± std over seeds)")
            plt.tight_layout()
            nash_png = os.path.join(args.outdir, "gems_nashconv_meanstd.png")
            plt.savefig(nash_png, dpi=200)
            plt.close()

            ev_m = df["mix_ev_mean"].values
            ev_s = df["mix_ev_std"].values
            plt.figure()
            plt.plot(it, ev_m)
            plt.fill_between(it, ev_m - ev_s, ev_m + ev_s, alpha=0.2)
            plt.xlabel("Iteration")
            plt.ylabel("Mixture EV (P1)")
            plt.title("GEMS on Kuhn Poker (mean ± std over seeds)")
            plt.tight_layout()
            ev_png = os.path.join(args.outdir, "gems_mixev_meanstd.png")
            plt.savefig(ev_png, dpi=200)
            plt.close()

            print(f"[PLOTS] saved: {nash_png}")
            print(f"[PLOTS] saved: {ev_png}")
        except Exception as e:
            print(f"[PLOTS] skipped due to error: {e}")


    if args.ttest_against_glob:
        base_paths = sorted(glob.glob(args.ttest_against_glob))
        if len(base_paths) == 0:
            print(f"[TTEST] no baseline files matched glob: {args.ttest_against_glob}")
        else:
            n = min(len(base_paths), len(seed_csvs))
            a = np.array([read_last_col(p, "nashconv") for p in seed_csvs[:n]], dtype=np.float64)
            b = np.array([read_last_col(p, "nashconv") for p in base_paths[:n]], dtype=np.float64)
            t, pval = welch_ttest(a, b)
            print(f"[TTEST] Welch t-test (final NashConv): "
                  f"GEMS mean={a.mean():.4f}±{a.std(ddof=1):.4f} vs baseline mean={b.mean():.4f}±{b.std(ddof=1):.4f} | "
                  f"t={t:.4f}, p={pval:.4g}")


if __name__ == "__main__":
    main()
