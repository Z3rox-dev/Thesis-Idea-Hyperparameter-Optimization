from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CopulaTPEConfig:
    # Elite definition (TPE-style)
    gamma: float = 0.2
    # Sampling / batching
    M: int = 128
    top_eval: int = 4
    # Exploration
    eps_explore: float = 0.05
    # Optional locality (0 = global elite)
    local_k: int = 0
    # Copula (CMA-like geometry) regularization
    alpha_corr: float = 0.1
    reg: float = 1e-6
    # Numerical safety for (u in (0,1))
    u_clip: float = 1e-6
    seed: int = 0


class CopulaTPE:
    """
    CopulaTPE (numpy-only).

    A single probabilistic model that combines:
      - TPE strength: flexible per-dimension marginals via empirical CDF/quantiles on elites
      - CMA strength: multivariate geometry via a Gaussian copula correlation matrix

    Model (in normalized space x in [0,1]^d):
        z ~ N(0, R)
        u = Phi(z)
        x_j = F_j^{-1}(u_j)

    where F_j are empirical elite marginals, and R is the (shrunk, PD) correlation matrix
    estimated after Gaussianizing elites via normal scores (rank -> u -> z).
    """

    def __init__(
        self,
        bounds,
        gamma: float = 0.2,
        M: int = 128,
        top_eval: int = 4,
        eps_explore: float = 0.05,
        local_k: int = 0,
        alpha_corr: float = 0.1,
        reg: float = 1e-6,
        u_clip: float = 1e-6,
        seed: int = 0,
    ):
        bounds_arr = np.asarray(bounds, dtype=float)
        if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
            raise ValueError("bounds must have shape (d, 2)")

        lower = bounds_arr[:, 0]
        upper = bounds_arr[:, 1]
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("bounds must be finite")
        if np.any(upper < lower):
            raise ValueError("bounds must satisfy upper >= lower")

        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must be in (0, 1)")
        if M < 1 or top_eval < 1:
            raise ValueError("M and top_eval must be >= 1")
        if not (0.0 <= eps_explore <= 1.0):
            raise ValueError("eps_explore must be in [0, 1]")
        if local_k < 0:
            raise ValueError("local_k must be >= 0")
        if not (0.0 <= alpha_corr <= 1.0):
            raise ValueError("alpha_corr must be in [0, 1]")
        if reg < 0.0:
            raise ValueError("reg must be >= 0")
        if not (0.0 < u_clip < 0.5):
            raise ValueError("u_clip must be in (0, 0.5)")

        self.lower = lower
        self.upper = upper
        self._range = np.where(upper > lower, upper - lower, 1.0)
        self.d = int(bounds_arr.shape[0])

        self.cfg = CopulaTPEConfig(
            gamma=gamma,
            M=M,
            top_eval=top_eval,
            eps_explore=eps_explore,
            local_k=local_k,
            alpha_corr=alpha_corr,
            reg=reg,
            u_clip=u_clip,
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)

        self.Xn_hist = np.empty((0, self.d), dtype=float)
        self.y_hist = np.empty((0,), dtype=float)
        self.best_x: np.ndarray | None = None
        self.best_y: float = float("inf")

        self._pending_Xn: list[np.ndarray] = []

        self.stats: dict[str, list] = {
            "best_y": [],
            "improved": [],
            "tau": [],
            "n_elite": [],
            "n_fit": [],
            "explore": [],
            "fallback": [],
            "local": [],
            "corr_cond": [],
            "from_pending": [],
        }
        self._last_ask_meta = {
            "tau": float("nan"),
            "n_elite": 0,
            "n_fit": 0,
            "explore": True,
            "fallback": True,
            "local": False,
            "corr_cond": float("nan"),
        }

    def ask(self) -> np.ndarray:
        if self._pending_Xn:
            xn = self._pending_Xn.pop(0)
            self._record_ask_meta(**self._last_ask_meta, from_pending=True)
            return self._denormalize(xn)

        n = int(self.y_hist.shape[0])
        n_min = max(10, 2 * self.d)
        if n < n_min or self.rng.random() < self.cfg.eps_explore:
            self._record_ask_meta(
                tau=float("nan"),
                n_elite=0,
                n_fit=0,
                explore=True,
                fallback=bool(n < n_min),
                local=False,
                corr_cond=float("nan"),
                from_pending=False,
            )
            return self._denormalize(self._random_Xn())

        tau = float(np.quantile(self.y_hist, self.cfg.gamma))
        elite_mask = self.y_hist <= tau
        elite_idx = np.flatnonzero(elite_mask)
        n_elite = int(elite_idx.size)
        if n_elite < 2:
            self._record_ask_meta(
                tau=tau,
                n_elite=n_elite,
                n_fit=0,
                explore=False,
                fallback=True,
                local=False,
                corr_cond=float("nan"),
                from_pending=False,
            )
            return self._denormalize(self._random_Xn())

        elite_Xn = self.Xn_hist[elite_idx]
        use_local = bool(self.cfg.local_k > 0 and elite_Xn.shape[0] >= 3)
        if use_local:
            anchor_Xn = self._choose_anchor_Xn(elite_idx)
            elite_Xn, used = self._knn_rows(elite_Xn, anchor_Xn, k=self.cfg.local_k)
            use_local = bool(used)

        model, corr_cond = self._fit_model(elite_Xn)
        if model is None:
            self._record_ask_meta(
                tau=tau,
                n_elite=n_elite,
                n_fit=int(elite_Xn.shape[0]),
                explore=False,
                fallback=True,
                local=use_local,
                corr_cond=float("nan"),
                from_pending=False,
            )
            return self._denormalize(self._random_Xn())

        Xn_cand = model.sample(self.rng, int(self.cfg.M))
        Xn_cand = np.clip(Xn_cand, 0.0, 1.0)

        top = int(min(self.cfg.top_eval, Xn_cand.shape[0]))
        selected: list[np.ndarray] = []
        seen = set()
        for i in range(Xn_cand.shape[0]):
            xn = Xn_cand[i]
            key = tuple(np.round(xn, 12))
            if key in seen:
                continue
            seen.add(key)
            selected.append(xn)
            if len(selected) >= top:
                break

        if not selected:
            self._record_ask_meta(
                tau=tau,
                n_elite=n_elite,
                n_fit=int(elite_Xn.shape[0]),
                explore=False,
                fallback=True,
                local=use_local,
                corr_cond=float(corr_cond),
                from_pending=False,
            )
            return self._denormalize(self._random_Xn())

        self._record_ask_meta(
            tau=tau,
            n_elite=n_elite,
            n_fit=int(elite_Xn.shape[0]),
            explore=False,
            fallback=False,
            local=use_local,
            corr_cond=float(corr_cond),
            from_pending=False,
        )
        self._pending_Xn.extend(selected[1:])
        return self._denormalize(selected[0])

    def tell(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x must have shape ({self.d},)")
        if not np.isfinite(y):
            raise ValueError("y must be finite")

        xn = self._normalize(x)
        self.Xn_hist = np.vstack([self.Xn_hist, xn[None, :]])
        self.y_hist = np.concatenate([self.y_hist, np.asarray([float(y)])])

        prev_best = self.best_y
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()

        self.stats["best_y"].append(self.best_y)
        self.stats["improved"].append(bool(y < prev_best))

    def _record_ask_meta(
        self,
        *,
        tau: float,
        n_elite: int,
        n_fit: int,
        explore: bool,
        fallback: bool,
        local: bool,
        corr_cond: float,
        from_pending: bool,
    ) -> None:
        self.stats["tau"].append(float(tau))
        self.stats["n_elite"].append(int(n_elite))
        self.stats["n_fit"].append(int(n_fit))
        self.stats["explore"].append(bool(explore))
        self.stats["fallback"].append(bool(fallback))
        self.stats["local"].append(bool(local))
        self.stats["corr_cond"].append(float(corr_cond))
        self.stats["from_pending"].append(bool(from_pending))
        self._last_ask_meta = {
            "tau": float(tau),
            "n_elite": int(n_elite),
            "n_fit": int(n_fit),
            "explore": bool(explore),
            "fallback": bool(fallback),
            "local": bool(local),
            "corr_cond": float(corr_cond),
        }

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xn = (x - self.lower) / self._range
        return np.clip(xn, 0.0, 1.0)

    def _denormalize(self, xn: np.ndarray) -> np.ndarray:
        xn = np.asarray(xn, dtype=float)
        x = self.lower + xn * self._range
        x = np.where(self.upper > self.lower, x, self.lower)
        return x

    def _random_Xn(self) -> np.ndarray:
        return self.rng.random(self.d)

    def _choose_anchor_Xn(self, elite_idx: np.ndarray) -> np.ndarray:
        if self.best_x is not None:
            best_Xn = self._normalize(self.best_x)
        else:
            best_Xn = self._random_Xn()

        if elite_idx.size < 2:
            return best_Xn

        # Mostly exploit best, sometimes jump to another elite (rank-weighted).
        if self.rng.random() >= 0.25:
            return best_Xn

        elite_y = self.y_hist[elite_idx]
        order = np.argsort(elite_y)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)
        weights = 1.0 / (1.0 + ranks.astype(float))
        weights = weights / weights.sum()
        pick = int(self.rng.choice(elite_idx.size, p=weights))
        return self.Xn_hist[elite_idx[pick]].copy()

    def _knn_rows(
        self, X: np.ndarray, anchor: np.ndarray, *, k: int
    ) -> tuple[np.ndarray, bool]:
        if k <= 0 or X.shape[0] == 0:
            return X, False
        diff = X - anchor[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        keep = dist2 > 1e-24
        Xk = X[keep]
        if Xk.shape[0] == 0:
            return X, False
        diff = Xk - anchor[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        kk = int(min(k, Xk.shape[0]))
        nn = np.argpartition(dist2, kk - 1)[:kk]
        return Xk[nn], True

    def _fit_model(self, elite_Xn: np.ndarray) -> tuple["_CopulaModel | None", float]:
        X = np.asarray(elite_Xn, dtype=float)
        n = int(X.shape[0])
        if n < 2:
            return None, float("nan")

        # 1) Empirical marginals (store sorted values per dim).
        sorted_vals = [np.sort(X[:, j].copy()) for j in range(self.d)]

        # 2) Gaussianize elites via normal scores (rank -> u -> z).
        U = np.empty_like(X)
        for j in range(self.d):
            U[:, j] = _rank_to_uniform(X[:, j], clip=self.cfg.u_clip)
        Z = _norm_ppf(U)
        Z = Z - Z.mean(axis=0, keepdims=True)

        # 3) Estimate correlation matrix (geometry) with shrinkage and PD clamp.
        cov = (Z.T @ Z) / max(n - 1, 1)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / (std[:, None] * std[None, :])
        corr = np.asarray(corr, dtype=float)
        corr[~np.isfinite(corr)] = 0.0
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        # Small-sample adaptive shrinkage: if n is small vs d, push more towards I.
        alpha = max(self.cfg.alpha_corr, min(0.95, self.d / max(n, 1)))
        corr = (1.0 - alpha) * corr + alpha * np.eye(self.d)
        corr = corr + float(self.cfg.reg) * np.eye(self.d)
        corr = _nearest_pd_corr(corr, min_eig=max(float(self.cfg.reg), 1e-12))

        w = np.linalg.eigvalsh(corr)
        wmin = float(np.min(w))
        wmax = float(np.max(w))
        corr_cond = float("inf") if wmin <= 0.0 else float(wmax / wmin)

        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            return None, corr_cond

        return _CopulaModel(sorted_vals=sorted_vals, chol=L, u_clip=self.cfg.u_clip), corr_cond


class _CopulaModel:
    def __init__(self, *, sorted_vals: list[np.ndarray], chol: np.ndarray, u_clip: float):
        self.sorted_vals = sorted_vals
        self.chol = np.asarray(chol, dtype=float)  # lower-triangular
        self.u_clip = float(u_clip)
        self.d = int(chol.shape[0])

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        z = rng.normal(0.0, 1.0, size=(int(n), self.d)) @ self.chol.T
        u = _norm_cdf(z)
        u = np.clip(u, self.u_clip, 1.0 - self.u_clip)
        X = np.empty_like(u)
        for j in range(self.d):
            X[:, j] = _quantile_from_sorted(self.sorted_vals[j], u[:, j])
        return X


def _rank_to_uniform(x: np.ndarray, *, clip: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if n <= 0:
        return np.empty((0,), dtype=float)
    if n == 1:
        return np.full((1,), 0.5, dtype=float)

    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]

    uniq, inv, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    starts = np.cumsum(np.concatenate([np.asarray([0]), counts[:-1]]))
    ends = starts + counts
    avg_rank = 0.5 * (starts + ends - 1).astype(float)

    ranks_sorted = avg_rank[inv]
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted

    u = (ranks + 0.5) / float(n)
    return np.clip(u, float(clip), 1.0 - float(clip))


def _quantile_from_sorted(sorted_x: np.ndarray, u: np.ndarray) -> np.ndarray:
    sorted_x = np.asarray(sorted_x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    n = int(sorted_x.size)
    if n == 0:
        return np.zeros_like(u)
    if n == 1:
        return np.full_like(u, sorted_x[0])

    pos = u * float(n - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0.astype(float)
    return sorted_x[i0] * (1.0 - frac) + sorted_x[i1] * frac


def _nearest_pd_corr(A: np.ndarray, *, min_eig: float) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, float(min_eig))
    B = (V * w[None, :]) @ V.T
    B = 0.5 * (B + B.T)

    d = np.sqrt(np.clip(np.diag(B), 1e-12, None))
    B = B / (d[:, None] * d[None, :])
    np.fill_diagonal(B, 1.0)
    return B


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Normal CDF approximation (Abramowitz-Stegun style), vectorized numpy-only.
    """
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = (
        0.319381530 * t
        + (-0.356563782) * t**2
        + 1.781477937 * t**3
        + (-1.821255978) * t**4
        + 1.330274429 * t**5
    )
    pdf = np.exp(-0.5 * ax * ax) / np.sqrt(2.0 * np.pi)
    cdf_pos = 1.0 - pdf * poly
    return np.where(x >= 0.0, cdf_pos, 1.0 - cdf_pos)


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Inverse Normal CDF (Acklam's approximation), vectorized numpy-only.
    p must be in (0, 1).
    """
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("p must be in (0, 1)")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ],
        dtype=float,
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ],
        dtype=float,
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ],
        dtype=float,
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ],
        dtype=float,
    )

    plow = 0.02425
    phigh = 1.0 - plow

    x = np.empty_like(p)

    # Lower tail
    m = p < plow
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(p[m]))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[m] = num / den

    # Central region
    m = (p >= plow) & (p <= phigh)
    if np.any(m):
        q = p[m] - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        x[m] = num / den

    # Upper tail
    m = p > phigh
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(1.0 - p[m]))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[m] = -(num / den)

    return x


def minimize(
    f: Callable[[np.ndarray], float], bounds, T: int, seed: int = 0, **kwargs
) -> dict:
    """
    Runs CopulaTPE for T evaluations and returns a result dict with:
      - best_x, best_y
      - X_hist (original bounds), y_hist
      - stats, config
    """
    kwargs = dict(kwargs)
    kwargs.pop("seed", None)
    opt = CopulaTPE(bounds, seed=seed, **kwargs)

    X_hist = np.empty((0, opt.d), dtype=float)
    y_hist = np.empty((0,), dtype=float)
    for _ in range(int(T)):
        x = opt.ask()
        y = float(f(x))
        opt.tell(x, y)
        X_hist = np.vstack([X_hist, x[None, :]])
        y_hist = np.concatenate([y_hist, np.asarray([y])])

    return {
        "best_x": None if opt.best_x is None else opt.best_x.copy(),
        "best_y": float(opt.best_y),
        "X_hist": X_hist,
        "y_hist": y_hist,
        "stats": opt.stats,
        "config": opt.cfg,
    }


def _sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def _rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def _random_search(
    f: Callable[[np.ndarray], float], bounds, T: int, seed: int = 0
) -> tuple[np.ndarray, float]:
    bounds_arr = np.asarray(bounds, dtype=float)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    span = np.where(upper > lower, upper - lower, 0.0)
    rng = np.random.default_rng(seed)
    best_x = lower.copy()
    best_y = float("inf")
    for _ in range(int(T)):
        x = lower + rng.random(lower.shape[0]) * span
        y = float(f(x))
        if y < best_y:
            best_y = y
            best_x = x.copy()
    return best_x, best_y


if __name__ == "__main__":
    d = 10
    problems = [
        ("Sphere", _sphere, [(-5.0, 5.0)] * d),
        ("Rosenbrock", _rosenbrock, [(-2.0, 2.0)] * d),
        ("Rastrigin", _rastrigin, [(-5.12, 5.12)] * d),
    ]
    T = 400

    for name, f, bounds in problems:
        res = minimize(f, bounds, T=T, seed=0, M=256, top_eval=8, local_k=0)
        _, best_random = _random_search(f, bounds, T=T, seed=0)
        print(
            f"{name:11s} | CopulaTPE best_y={res['best_y']:.6g} | random best_y={best_random:.6g}"
        )
