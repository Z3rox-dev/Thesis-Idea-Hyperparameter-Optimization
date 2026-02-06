from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ECFSConfig:
    gamma: float = 0.2
    k_elite: int = 30
    k_nonelite: int = 60
    M: int = 128
    top_eval: int = 2
    alpha_shrink: float = 0.1
    reg: float = 1e-6
    step_scale: float = 1.0
    eps_explore: float = 0.05
    seed: int = 0
    use_ratio: bool = True
    diag_cov: bool = False
    mu_zero: bool = False


class ECFS:
    """
    ECFS – Elite-Conditioned Flow Sampler (numpy-only).

    Ask/tell optimizer that proposes new candidates by sampling *steps* Δx
    conditioned on elite observations, and ranks steps by a contrastive
    log-likelihood ratio between elite/non-elite step models.
    """

    def __init__(
        self,
        bounds,
        gamma: float = 0.2,
        k_elite: int = 30,
        k_nonelite: int = 60,
        M: int = 128,
        top_eval: int = 2,
        alpha_shrink: float = 0.1,
        reg: float = 1e-6,
        step_scale: float = 1.0,
        eps_explore: float = 0.05,
        seed: int = 0,
        *,
        use_ratio: bool = True,
        diag_cov: bool = False,
        mu_zero: bool = False,
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

        self.lower = lower
        self.upper = upper
        self._range = np.where(upper > lower, upper - lower, 1.0)
        self.d = int(bounds_arr.shape[0])

        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must be in (0, 1)")
        if k_elite < 1 or k_nonelite < 1:
            raise ValueError("k_elite and k_nonelite must be >= 1")
        if M < 1 or top_eval < 1:
            raise ValueError("M and top_eval must be >= 1")
        if not (0.0 <= eps_explore <= 1.0):
            raise ValueError("eps_explore must be in [0, 1]")
        if alpha_shrink < 0.0 or alpha_shrink > 1.0:
            raise ValueError("alpha_shrink must be in [0, 1]")
        if reg < 0.0:
            raise ValueError("reg must be >= 0")
        if step_scale <= 0.0:
            raise ValueError("step_scale must be > 0")

        self.cfg = ECFSConfig(
            gamma=gamma,
            k_elite=k_elite,
            k_nonelite=k_nonelite,
            M=M,
            top_eval=top_eval,
            alpha_shrink=alpha_shrink,
            reg=reg,
            step_scale=step_scale,
            eps_explore=eps_explore,
            seed=seed,
            use_ratio=use_ratio,
            diag_cov=diag_cov,
            mu_zero=mu_zero,
        )

        self.rng = np.random.default_rng(seed)

        self.Xn_hist = np.empty((0, self.d), dtype=float)
        self.y_hist = np.empty((0,), dtype=float)

        self.best_x: np.ndarray | None = None
        self.best_y: float = float("inf")

        self._pending_Xn: list[np.ndarray] = []
        self._last_anchor_Xn: np.ndarray | None = None

        self.stats: dict[str, list] = {
            "best_y": [],
            "improved": [],
            "tau": [],
            "n_elite": [],
            "n_nonelite": [],
            "explore": [],
            "fallback": [],
            "cond_elite": [],
            "cond_nonelite": [],
            "batch_top_score": [],
            "from_pending": [],
        }
        self._last_ask_meta = {
            "tau": float("nan"),
            "n_elite": 0,
            "n_nonelite": 0,
            "explore": True,
            "fallback": True,
            "cond_elite": float("nan"),
            "cond_nonelite": float("nan"),
            "batch_top_score": float("nan"),
        }

    def _record_ask_meta(
        self,
        *,
        tau: float,
        n_elite: int,
        n_nonelite: int,
        explore: bool,
        fallback: bool,
        cond_elite: float,
        cond_nonelite: float,
        batch_top_score: float,
        from_pending: bool,
    ) -> None:
        self.stats["tau"].append(float(tau))
        self.stats["n_elite"].append(int(n_elite))
        self.stats["n_nonelite"].append(int(n_nonelite))
        self.stats["explore"].append(bool(explore))
        self.stats["fallback"].append(bool(fallback))
        self.stats["cond_elite"].append(float(cond_elite))
        self.stats["cond_nonelite"].append(float(cond_nonelite))
        self.stats["batch_top_score"].append(float(batch_top_score))
        self.stats["from_pending"].append(bool(from_pending))
        self._last_ask_meta = {
            "tau": float(tau),
            "n_elite": int(n_elite),
            "n_nonelite": int(n_nonelite),
            "explore": bool(explore),
            "fallback": bool(fallback),
            "cond_elite": float(cond_elite),
            "cond_nonelite": float(cond_nonelite),
            "batch_top_score": float(batch_top_score),
        }

    def ask(self) -> np.ndarray:
        """Return one candidate x in original bounds."""
        if self._pending_Xn:
            Xn = self._pending_Xn.pop(0)
            self._record_ask_meta(**self._last_ask_meta, from_pending=True)
            return self._denormalize(Xn)

        # Exploration (explicit) or cold start.
        n = int(self.y_hist.shape[0])
        n_min = max(10, 2 * self.d)
        if n < n_min or self.rng.random() < self.cfg.eps_explore:
            self._record_ask_meta(
                tau=float("nan"),
                n_elite=0,
                n_nonelite=0,
                explore=True,
                fallback=bool(n < n_min),
                cond_elite=float("nan"),
                cond_nonelite=float("nan"),
                batch_top_score=float("nan"),
                from_pending=False,
            )
            return self._denormalize(self._random_Xn())

        tau = float(np.quantile(self.y_hist, self.cfg.gamma))
        elite_mask = self.y_hist <= tau
        nonelite_mask = ~elite_mask
        elite_idx = np.flatnonzero(elite_mask)
        nonelite_idx = np.flatnonzero(nonelite_mask)

        # Anchor: best-so-far by default; sometimes sample within elite (rank-weighted).
        anchor_Xn = self._choose_anchor_Xn(elite_idx)
        self._last_anchor_Xn = anchor_Xn

        deltas_E, fallback_E = self._local_deltas(anchor_Xn, elite_idx, self.cfg.k_elite)
        deltas_N, fallback_N = self._local_deltas(
            anchor_Xn, nonelite_idx, self.cfg.k_nonelite
        )

        if deltas_E.shape[0] < 2:
            # Not enough local signal even after KNN (e.g. all points identical).
            self._record_ask_meta(
                tau=tau,
                n_elite=int(elite_idx.size),
                n_nonelite=int(nonelite_idx.size),
                explore=False,
                fallback=True,
                cond_elite=float("nan"),
                cond_nonelite=float("nan"),
                batch_top_score=float("nan"),
                from_pending=False,
            )
            return self._denormalize(self._fallback_isotropic(anchor_Xn))

        model_E, cond_E = self._fit_gaussian(deltas_E)

        if self.cfg.use_ratio and deltas_N.shape[0] >= 2:
            model_N, cond_N = self._fit_gaussian(deltas_N)
        else:
            # Contrast model fallback: isotropic around 0.
            model_N, cond_N = self._isotropic_model(scale=0.25), float("nan")
            fallback_N = True

        # Sample step candidates from the elite-conditioned model.
        deltas = model_E.sample(self.rng, self.cfg.M)
        Xn_cand = anchor_Xn[None, :] + self.cfg.step_scale * deltas
        Xn_cand = np.clip(Xn_cand, 0.0, 1.0)

        # Score by contrastive ratio (or elite-only).
        logp_E = model_E.logpdf(deltas)
        if self.cfg.use_ratio:
            logp_N = model_N.logpdf(deltas)
            score = logp_E - logp_N
        else:
            score = logp_E

        # Keep top_eval candidates; return one now, keep the rest pending.
        top = int(min(self.cfg.top_eval, self.cfg.M))
        order = np.argsort(score)[::-1]

        selected: list[np.ndarray] = []
        seen = set()
        for idx in order:
            xn = Xn_cand[idx]
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
                n_elite=int(elite_idx.size),
                n_nonelite=int(nonelite_idx.size),
                explore=False,
                fallback=True,
                cond_elite=cond_E,
                cond_nonelite=cond_N,
                batch_top_score=float("nan"),
                from_pending=False,
            )
            return self._denormalize(self._fallback_isotropic(anchor_Xn))

        self._record_ask_meta(
            tau=tau,
            n_elite=int(elite_idx.size),
            n_nonelite=int(nonelite_idx.size),
            explore=False,
            fallback=bool(fallback_E or fallback_N),
            cond_elite=cond_E,
            cond_nonelite=cond_N,
            batch_top_score=float(score[order[0]]),
            from_pending=False,
        )

        self._pending_Xn.extend(selected[1:])
        return self._denormalize(selected[0])

    def tell(self, x: np.ndarray, y: float) -> None:
        """Add observation."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x must have shape ({self.d},)")
        if not np.isfinite(y):
            raise ValueError("y must be finite")

        Xn = self._normalize(x)
        self.Xn_hist = np.vstack([self.Xn_hist, Xn[None, :]])
        self.y_hist = np.concatenate([self.y_hist, np.asarray([float(y)])])

        prev_best = self.best_y
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()

        improved = bool(y < prev_best)
        self.stats["best_y"].append(self.best_y)
        self.stats["improved"].append(improved)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xn = (x - self.lower) / self._range
        return np.clip(xn, 0.0, 1.0)

    def _denormalize(self, xn: np.ndarray) -> np.ndarray:
        xn = np.asarray(xn, dtype=float)
        x = self.lower + xn * self._range
        # For degenerate bounds, keep exactly the bound value.
        x = np.where(self.upper > self.lower, x, self.lower)
        return x

    def _random_Xn(self) -> np.ndarray:
        # Uniform in [0,1]^d (numpy-only). Swap in Sobol/Halton if desired.
        return self.rng.random(self.d)

    def _choose_anchor_Xn(self, elite_idx: np.ndarray) -> np.ndarray:
        # Default: best-so-far.
        if self.best_x is not None:
            best_Xn = self._normalize(self.best_x)
        else:
            best_Xn = self._random_Xn()

        if elite_idx.size < 2:
            return best_Xn

        # With some probability, sample from elite with stable rank-based weights.
        if self.rng.random() >= 0.30:
            return best_Xn

        elite_y = self.y_hist[elite_idx]
        order = np.argsort(elite_y)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.size)
        weights = 1.0 / (1.0 + ranks.astype(float))
        weights = weights / weights.sum()
        pick = int(self.rng.choice(elite_idx.size, p=weights))
        return self.Xn_hist[elite_idx[pick]].copy()

    def _local_deltas(
        self, anchor_Xn: np.ndarray, idx: np.ndarray, k: int
    ) -> tuple[np.ndarray, bool]:
        if idx.size == 0:
            return np.empty((0, self.d), dtype=float), True

        Xn = self.Xn_hist[idx]
        diff = Xn - anchor_Xn[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        # Exclude exact match (anchor itself) if present.
        keep = dist2 > 1e-24
        Xn = Xn[keep]
        if Xn.shape[0] == 0:
            return np.empty((0, self.d), dtype=float), True

        diff = Xn - anchor_Xn[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        kk = int(min(k, Xn.shape[0]))
        nn = np.argpartition(dist2, kk - 1)[:kk]
        deltas = diff[nn]
        fallback = Xn.shape[0] < k
        return deltas, fallback

    def _fallback_isotropic(self, anchor_Xn: np.ndarray) -> np.ndarray:
        # Isotropic local step around the anchor.
        sigma = 0.15
        Xn = anchor_Xn + self.cfg.step_scale * self.rng.normal(0.0, sigma, size=self.d)
        return np.clip(Xn, 0.0, 1.0)

    def _fit_gaussian(self, deltas: np.ndarray) -> tuple["_GaussianModel", float]:
        deltas = np.asarray(deltas, dtype=float)
        n = int(deltas.shape[0])
        if n < 2:
            return self._isotropic_model(scale=0.25), float("nan")

        if self.cfg.mu_zero:
            mu = np.zeros(self.d, dtype=float)
            centered = deltas
        else:
            mu = deltas.mean(axis=0)
            centered = deltas - mu[None, :]

        cov = (centered.T @ centered) / max(n - 1, 1)
        cov = cov + self.cfg.reg * np.eye(self.d)

        if self.cfg.diag_cov:
            cov = np.diag(np.diag(cov))

        # Shrinkage towards scaled identity (trace/d).
        tr = float(np.trace(cov))
        target = (tr / max(self.d, 1)) * np.eye(self.d)
        cov = (1.0 - self.cfg.alpha_shrink) * cov + self.cfg.alpha_shrink * target

        L, logdet = _chol_with_jitter(cov, self.cfg.reg)
        model = _GaussianModel(mu=mu, chol=L, logdet=logdet)
        cond = _cond_from_chol(L)
        return model, cond

    def _isotropic_model(self, scale: float) -> "_GaussianModel":
        mu = np.zeros(self.d, dtype=float)
        cov = (scale**2) * np.eye(self.d)
        L, logdet = _chol_with_jitter(cov, self.cfg.reg)
        return _GaussianModel(mu=mu, chol=L, logdet=logdet)


class _GaussianModel:
    def __init__(self, *, mu: np.ndarray, chol: np.ndarray, logdet: float):
        self.mu = np.asarray(mu, dtype=float)
        self.chol = np.asarray(chol, dtype=float)  # lower-triangular
        self.logdet = float(logdet)
        self.d = int(self.mu.shape[0])

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        z = rng.normal(0.0, 1.0, size=(int(n), self.d))
        return self.mu[None, :] + z @ self.chol.T

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        xc = x - self.mu[None, :]
        # Solve L v = xc^T  => v = L^{-1} xc^T
        v = np.linalg.solve(self.chol, xc.T)
        maha = np.sum(v * v, axis=0)
        return -0.5 * (maha + self.logdet + self.d * np.log(2.0 * np.pi))


def _chol_with_jitter(cov: np.ndarray, base_jitter: float) -> tuple[np.ndarray, float]:
    cov = np.asarray(cov, dtype=float)
    jitter = max(float(base_jitter), 1e-12)
    for _ in range(12):
        try:
            L = np.linalg.cholesky(cov)
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
            return L, logdet
        except np.linalg.LinAlgError:
            cov = cov + jitter * np.eye(cov.shape[0])
            jitter *= 10.0
    # Last resort: eigenvalue clamp
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    cov_pd = (V * w[None, :]) @ V.T
    L = np.linalg.cholesky(cov_pd)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return L, logdet


def _cond_from_chol(L: np.ndarray) -> float:
    # Approximate condition number from Cholesky diag (cheap proxy).
    d = np.abs(np.diag(L))
    dmin = float(np.min(d))
    dmax = float(np.max(d))
    if dmin <= 0.0:
        return float("inf")
    return float((dmax / dmin) ** 2)


def minimize(
    f: Callable[[np.ndarray], float], bounds, T: int, seed: int = 0, **ecfs_kwargs
) -> dict:
    """
    Runs ECFS for T evaluations and returns a result dict with:
      - best_x, best_y
      - X_hist (original bounds), y_hist
      - stats, config
    """
    ecfs_kwargs = dict(ecfs_kwargs)
    ecfs_kwargs.pop("seed", None)
    opt = ECFS(bounds, seed=seed, **ecfs_kwargs)
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
    d = 5
    problems = [
        ("Sphere", _sphere, [(-5.0, 5.0)] * d),
        ("Rosenbrock", _rosenbrock, [(-2.0, 2.0)] * d),
        ("Rastrigin", _rastrigin, [(-5.12, 5.12)] * d),
    ]
    T = 200

    for name, f, bounds in problems:
        res_ecfs = minimize(f, bounds, T=T, seed=0, top_eval=2, M=128)
        _, best_random = _random_search(f, bounds, T=T, seed=0)
        print(f"{name:11s} | ECFS best_y={res_ecfs['best_y']:.6g} | random best_y={best_random:.6g}")
