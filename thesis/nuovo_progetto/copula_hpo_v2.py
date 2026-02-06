#!/usr/bin/env python3
"""
CopulaHPO v2 - Unified Probabilistic HPO for Mixed Spaces.

A single probabilistic model that handles:
- Continuous variables (Gaussian marginal)
- Categorical variables (Categorical marginal)  
- Integer variables (Discrete marginal)

All correlations are captured in a Gaussian copula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Any
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CopulaHPOConfig:
    mode: Literal["elite", "elite_ratio", "latent_cma"] = "elite"
    gamma: float = 0.2
    M: int | None = None  # None = auto-select based on mode/dimension
    top_eval: int = 4
    eps_explore: float = 0.05
    alpha_corr: float = 0.1
    reg: float = 1e-6
    u_clip: float = 1e-6
    seed: int = 0
    cma_mu: int | None = None
    cma_sigma0: float = 1.0
    cma_active: bool = False  # Active/contrastive covariance update using worst samples (still 1 distribution).
    cma_active_eta: float = 0.3  # Strength of negative covariance update (0 disables).
    budget: int | None = None  # Optional budget hint for adaptive lambda
    cma_min_generations: int = 20  # Target minimum generations for latent_cma


# =============================================================================
# Marginali
# =============================================================================

class Marginal(ABC):
    """Base class for marginal distributions."""
    
    @abstractmethod
    def fit(self, values: np.ndarray) -> None:
        """Fit the marginal from elite samples."""
        pass
    
    @abstractmethod
    def to_uniform(self, values: np.ndarray) -> np.ndarray:
        """Transform values to uniform (0,1)."""
        pass
    
    @abstractmethod
    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        """Transform uniform (0,1) back to original space."""
        pass


class ContinuousMarginal(Marginal):
    """Empirical marginal for continuous variables via Gaussian fit with bounds."""
    
    def __init__(self, lower: float, upper: float, u_clip: float = 1e-6):
        self.lower = lower
        self.upper = upper
        self.u_clip = u_clip
        self.sorted_vals: np.ndarray | None = None
        # Gaussian parameters for extrapolation
        self.mu: float = 0.5 * (lower + upper)
        self.sigma: float = (upper - lower) / 4.0
    
    def fit(self, values: np.ndarray) -> None:
        if self.upper == self.lower:
            self.sorted_vals = np.sort(values.copy()) if len(values) > 0 else np.asarray([self.lower])
            self.mu = float(self.lower)
            self.sigma = 1.0
            return
        self.sorted_vals = np.sort(values.copy())
        # Fit Gaussian for extrapolation capability
        if len(values) >= 2:
            self.mu = float(np.mean(values))
            self.sigma = float(np.std(values))
            # Ensure sigma is not too small (for exploration)
            # Use at least 10% of the range
            span = float(self.upper - self.lower)
            min_sigma = max(span / 10.0, 1e-12)
            self.sigma = max(self.sigma, min_sigma)
    
    def to_uniform(self, values: np.ndarray) -> np.ndarray:
        # Use Gaussian CDF for consistent forward/inverse
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            self.sigma = 1.0
        u = stats.norm.cdf(values, loc=self.mu, scale=self.sigma)
        return np.clip(u, self.u_clip, 1.0 - self.u_clip)
    
    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        # Use Gaussian PPF - this CAN extrapolate beyond observed range!
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            self.sigma = 1.0
        x = stats.norm.ppf(u, loc=self.mu, scale=self.sigma)
        return np.clip(x, self.lower, self.upper)


class CategoricalMarginal(Marginal):
    """Marginal for categorical variables."""
    
    def __init__(self, categories: list[Any], u_clip: float = 1e-6):
        self.categories = list(categories)
        self.n_cat = len(categories)
        self.u_clip = u_clip
        self.probs: np.ndarray | None = None
        self.cdf: np.ndarray | None = None
    
    def fit(self, values: np.ndarray) -> None:
        # Count frequencies
        counts = np.zeros(self.n_cat)
        for i, cat in enumerate(self.categories):
            counts[i] = np.sum(values == cat)
        
        # Laplace smoothing
        counts = counts + 1
        self.probs = counts / counts.sum()
        self.cdf = np.cumsum(self.probs)
    
    def to_uniform(self, values: np.ndarray) -> np.ndarray:
        """Map category to middle of its probability interval."""
        if self.cdf is None:
            self.probs = np.ones(self.n_cat) / self.n_cat
            self.cdf = np.cumsum(self.probs)
        
        u = np.zeros(len(values), dtype=float)
        for i, val in enumerate(values):
            try:
                idx = self.categories.index(val)
            except ValueError:
                idx = 0
            # Middle of the interval
            lower = 0.0 if idx == 0 else self.cdf[idx - 1]
            upper = self.cdf[idx]
            u[i] = (lower + upper) / 2
        
        return np.clip(u, self.u_clip, 1.0 - self.u_clip)
    
    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        """Map uniform to category."""
        if self.cdf is None:
            self.probs = np.ones(self.n_cat) / self.n_cat
            self.cdf = np.cumsum(self.probs)
        
        # u falls in which interval?
        indices = np.searchsorted(self.cdf, u, side='right')
        indices = np.clip(indices, 0, self.n_cat - 1)
        return np.array([self.categories[i] for i in indices])


class IntegerMarginal(Marginal):
    """Marginal for integer/ordinal variables."""
    
    def __init__(self, lower: int, upper: int, u_clip: float = 1e-6):
        self.lower = int(lower)
        self.upper = int(upper)
        self.u_clip = u_clip
        self.values = list(range(self.lower, self.upper + 1))
        self.n_vals = len(self.values)
        self.probs: np.ndarray | None = None
        self.cdf: np.ndarray | None = None
    
    def fit(self, values: np.ndarray) -> None:
        counts = np.zeros(self.n_vals)
        for i, val in enumerate(self.values):
            counts[i] = np.sum(values == val)
        
        # Laplace smoothing
        counts = counts + 1
        self.probs = counts / counts.sum()
        self.cdf = np.cumsum(self.probs)
    
    def to_uniform(self, values: np.ndarray) -> np.ndarray:
        if self.cdf is None:
            self.probs = np.ones(self.n_vals) / self.n_vals
            self.cdf = np.cumsum(self.probs)
        
        u = np.zeros(len(values), dtype=float)
        for i, val in enumerate(values):
            idx = int(val) - self.lower
            idx = np.clip(idx, 0, self.n_vals - 1)
            lower = 0.0 if idx == 0 else self.cdf[idx - 1]
            upper = self.cdf[idx]
            u[i] = (lower + upper) / 2
        
        return np.clip(u, self.u_clip, 1.0 - self.u_clip)
    
    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        if self.cdf is None:
            self.probs = np.ones(self.n_vals) / self.n_vals
            self.cdf = np.cumsum(self.probs)
        
        indices = np.searchsorted(self.cdf, u, side='right')
        indices = np.clip(indices, 0, self.n_vals - 1)
        return np.array([self.values[i] for i in indices])


# =============================================================================
# Utilità
# =============================================================================

def _rank_to_uniform(x: np.ndarray, sorted_ref: np.ndarray, clip: float) -> np.ndarray:
    """Transform values to uniform via empirical CDF."""
    n = len(sorted_ref)
    if n == 0:
        return np.full_like(x, 0.5, dtype=float)
    
    # Rank of each x in sorted_ref
    ranks = np.searchsorted(sorted_ref, x, side='right')
    u = (ranks + 0.5) / (n + 1)
    return np.clip(u, clip, 1.0 - clip)


def _quantile_from_sorted(sorted_x: np.ndarray, u: np.ndarray, 
                          lower: float, upper: float) -> np.ndarray:
    """Inverse empirical CDF via linear interpolation."""
    n = len(sorted_x)
    if n == 0:
        return np.full_like(u, (lower + upper) / 2, dtype=float)
    if n == 1:
        return np.full_like(u, sorted_x[0], dtype=float)
    
    # Interpolate
    idx_float = u * (n - 1)
    idx_low = np.floor(idx_float).astype(int)
    idx_high = np.ceil(idx_float).astype(int)
    idx_low = np.clip(idx_low, 0, n - 1)
    idx_high = np.clip(idx_high, 0, n - 1)
    
    frac = idx_float - idx_low
    result = sorted_x[idx_low] * (1 - frac) + sorted_x[idx_high] * frac
    return np.clip(result, lower, upper)


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    """Standard normal quantile function."""
    return stats.norm.ppf(u)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return stats.norm.cdf(z)


def _nearest_pd(A: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Find nearest positive-definite matrix."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, min_eig)
    return (V * w[None, :]) @ V.T


@dataclass
class _LatentCMAState:
    """Minimal CMA-ES state in latent z-space."""

    m: np.ndarray
    C: np.ndarray
    sigma: float
    p_c: np.ndarray
    p_sigma: np.ndarray
    gen: int


# =============================================================================
# CopulaHPO
# =============================================================================

@dataclass
class HyperparameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    type: Literal["continuous", "categorical", "integer"]
    # For continuous: (lower, upper)
    # For categorical: list of categories
    # For integer: (lower, upper)
    bounds: tuple | list


class CopulaHPO:
    """
    CopulaHPO v2 - Unified optimizer for mixed hyperparameter spaces.
    
    Uses Gaussian copula to capture correlations across all variable types.
    """
    
    def __init__(
        self,
        param_specs: list[HyperparameterSpec],
        mode: Literal["elite", "elite_ratio", "latent_cma"] = "elite",
        gamma: float = 0.2,
        M: int | None = None,
        top_eval: int = 4,
        eps_explore: float = 0.05,
        alpha_corr: float = 0.1,
        reg: float = 1e-6,
        u_clip: float = 1e-6,
        seed: int = 0,
        cma_mu: int | None = None,
        cma_sigma0: float = 1.0,
        cma_active: bool = False,
        cma_active_eta: float = 0.3,
        budget: int | None = None,
        cma_min_generations: int = 20,
    ):
        self.param_specs = param_specs
        self.d = len(param_specs)
        
        # Compute adaptive M based on mode and dimension
        if M is None:
            M = self._compute_adaptive_M(mode, self.d, budget, cma_min_generations)
        
        self.cfg = CopulaHPOConfig(
            mode=mode,
            gamma=gamma,
            M=M,
            top_eval=top_eval,
            eps_explore=eps_explore,
            alpha_corr=alpha_corr,
            reg=reg,
            u_clip=u_clip,
            seed=seed,
            budget=budget,
            cma_min_generations=cma_min_generations,
            cma_mu=cma_mu,
            cma_sigma0=cma_sigma0,
            cma_active=bool(cma_active),
            cma_active_eta=float(cma_active_eta),
        )
        
        self.rng = np.random.default_rng(seed)
        
        # Initialize marginals
        self.marginals: list[Marginal] = []
        for spec in param_specs:
            if spec.type == "continuous":
                self.marginals.append(ContinuousMarginal(
                    spec.bounds[0], spec.bounds[1], u_clip
                ))
            elif spec.type == "categorical":
                self.marginals.append(CategoricalMarginal(
                    spec.bounds, u_clip
                ))
            elif spec.type == "integer":
                self.marginals.append(IntegerMarginal(
                    spec.bounds[0], spec.bounds[1], u_clip
                ))
        
        # History
        self.X_hist: list[dict] = []
        self.y_hist: list[float] = []
        
        self.best_x: dict | None = None
        self.best_y: float = float("inf")
        
        self._pending: list[dict] = []

        # Latent-CMA state (optional).
        self._cma: _LatentCMAState | None = None
        self._latent_by_key: dict[tuple, list[np.ndarray]] = {}
        self._gen_expected: int = 0
        self._gen_received: int = 0
        self._gen_z: list[np.ndarray] = []
        self._gen_y: list[float] = []
    
    @staticmethod
    def _compute_adaptive_M(
        mode: str, 
        d: int, 
        budget: int | None, 
        cma_min_generations: int = 20
    ) -> int:
        """
        Compute adaptive M (population size) based on mode, dimension, and budget.
        
        For 'elite' mode:
            M = 128 (best-of-M internal selection)
        
        For 'latent_cma' mode:
            Base: λ_dim = 4 + floor(3 * ln(d))  [standard CMA]
            Budget check: if budget / λ_dim < cma_min_generations,
                          reduce λ = max(4, budget // cma_min_generations)
        """
        if mode in ("elite", "elite_ratio"):
            # Elite mode: M is for internal candidate generation
            return 128
        
        # Latent-CMA mode: dimension-driven lambda
        lambda_dim = 4 + int(np.floor(3 * np.log(max(d, 1))))
        
        if budget is None:
            # No budget hint, use dimension-based default
            return lambda_dim
        
        # Check if we have enough generations
        n_generations = budget // lambda_dim
        
        if n_generations < cma_min_generations:
            # Budget too small for standard lambda, reduce it
            lambda_budget = max(4, budget // cma_min_generations)
            return lambda_budget
        
        return lambda_dim
    
    def _random_sample(self) -> dict:
        """Generate a random sample."""
        x = {}
        for spec in self.param_specs:
            if spec.type == "continuous":
                x[spec.name] = self.rng.uniform(spec.bounds[0], spec.bounds[1])
            elif spec.type == "categorical":
                x[spec.name] = self.rng.choice(spec.bounds)
            elif spec.type == "integer":
                x[spec.name] = self.rng.integers(spec.bounds[0], spec.bounds[1] + 1)
        return x
    
    def _to_array(self, x: dict) -> np.ndarray:
        """Convert dict to array (for internal use)."""
        arr = np.zeros(self.d)
        for i, spec in enumerate(self.param_specs):
            val = x[spec.name]
            if spec.type == "categorical":
                # Store index
                arr[i] = spec.bounds.index(val) if val in spec.bounds else 0
            else:
                arr[i] = val
        return arr
    
    def _get_elite_arrays(self) -> tuple[list[np.ndarray], int]:
        """Get elite observations as list of arrays per dimension."""
        if len(self.y_hist) < 2:
            return [np.empty(0) for _ in range(self.d)], 0
        
        tau = np.quantile(self.y_hist, self.cfg.gamma)
        elite_mask = np.array(self.y_hist) <= tau
        elite_idx = np.where(elite_mask)[0]
        
        if len(elite_idx) < 2:
            return [np.empty(0) for _ in range(self.d)], 0
        
        return self._arrays_from_indices(elite_idx)

    def _arrays_from_indices(self, indices: np.ndarray) -> tuple[list[np.ndarray], int]:
        """Build per-dimension arrays for the given history indices (keeps original types)."""
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return [np.empty(0) for _ in range(self.d)], 0

        arrays: list[np.ndarray] = []
        for j, spec in enumerate(self.param_specs):
            col = [self.X_hist[int(i)][spec.name] for i in indices]
            if spec.type == "categorical":
                arrays.append(np.array(col, dtype=object))
            elif spec.type == "integer":
                arrays.append(np.array(col, dtype=int))
            else:
                arrays.append(np.array(col, dtype=float))
        return arrays, int(indices.size)

    def _make_marginal(self, spec: HyperparameterSpec) -> Marginal:
        """Construct a fresh marginal instance for a spec (used for two-model fitting)."""
        if spec.type == "continuous":
            return ContinuousMarginal(spec.bounds[0], spec.bounds[1], self.cfg.u_clip)
        if spec.type == "categorical":
            return CategoricalMarginal(spec.bounds, self.cfg.u_clip)
        return IntegerMarginal(spec.bounds[0], spec.bounds[1], self.cfg.u_clip)

    def _fit_copula_model(self, arrays: list[np.ndarray], n: int) -> tuple[list[Marginal], np.ndarray, float] | None:
        """
        Fit marginals + Gaussian copula correlation from a dataset.

        Returns: (marginals, L, logdet_corr) where corr = L L^T.
        """
        if n < 3:
            return None
        marginals = [self._make_marginal(spec) for spec in self.param_specs]

        U = np.zeros((n, self.d), dtype=float)
        for j in range(self.d):
            marginals[j].fit(arrays[j])
            U[:, j] = marginals[j].to_uniform(arrays[j])

        U = np.clip(U, self.cfg.u_clip, 1.0 - self.cfg.u_clip)
        Z = _norm_ppf(U)
        Z = Z - Z.mean(axis=0, keepdims=True)

        cov = np.cov(Z.T)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]], dtype=float)

        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        alpha = max(self.cfg.alpha_corr, min(0.9, self.d / max(n, 1)))
        corr = (1.0 - alpha) * corr + alpha * np.eye(self.d)
        corr = corr + self.cfg.reg * np.eye(self.d)
        corr = _nearest_pd(corr, min_eig=self.cfg.reg)

        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            return None

        logdet = float(2.0 * np.sum(np.log(np.clip(np.diag(L), 1e-300, None))))
        return marginals, L, logdet

    def _log_copula_density(self, z: np.ndarray, L: np.ndarray, logdet: float) -> float:
        """Gaussian copula density log c_R(u) using a Cholesky factor L of corr."""
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or z.shape[0] != self.d:
            return float("-inf")
        try:
            v = np.linalg.solve(L, z)
        except Exception:
            return float("-inf")
        quad = float(np.dot(v, v))
        zz = float(np.dot(z, z))
        return float(-0.5 * logdet - 0.5 * (quad - zz))

    def _log_marginal_density(self, marginal: Marginal, spec: HyperparameterSpec, value: Any) -> float:
        """Log of marginal density/mass for a single value under the fitted marginal."""
        if spec.type == "continuous":
            m = marginal
            if isinstance(m, ContinuousMarginal):
                if m.upper == m.lower:
                    return 0.0 if float(value) == float(m.lower) else float("-inf")
                sigma = float(m.sigma) if np.isfinite(m.sigma) and m.sigma > 0.0 else 1.0
                return float(stats.norm.logpdf(float(value), loc=float(m.mu), scale=sigma))
            return float("-inf")

        if spec.type == "categorical":
            m = marginal
            if isinstance(m, CategoricalMarginal):
                if m.probs is None:
                    return float("-inf")
                try:
                    idx = m.categories.index(value)
                except ValueError:
                    return float("-inf")
                p = float(m.probs[idx])
                return float(np.log(max(p, 1e-300)))
            return float("-inf")

        # integer
        m = marginal
        if isinstance(m, IntegerMarginal):
            if m.probs is None:
                return float("-inf")
            try:
                idx = int(value) - int(m.lower)
            except Exception:
                return float("-inf")
            if idx < 0 or idx >= m.n_vals:
                return float("-inf")
            p = float(m.probs[idx])
            return float(np.log(max(p, 1e-300)))
        return float("-inf")

    def _log_joint_density(self, x: dict, marginals: list[Marginal], L: np.ndarray, logdet: float) -> float:
        """Log joint density under the copula model: log c_R(u) + sum log f_j(x_j)."""
        u = np.zeros(self.d, dtype=float)
        logm = 0.0
        for j, spec in enumerate(self.param_specs):
            v = x[spec.name]
            if spec.type == "categorical":
                arr = np.asarray([v], dtype=object)
            elif spec.type == "integer":
                arr = np.asarray([int(v)], dtype=int)
            else:
                arr = np.asarray([float(v)], dtype=float)

            uj = float(marginals[j].to_uniform(arr)[0])
            uj = float(np.clip(uj, self.cfg.u_clip, 1.0 - self.cfg.u_clip))
            u[j] = uj
            logm += self._log_marginal_density(marginals[j], spec, v)

        z = _norm_ppf(u)
        z = np.where(np.isfinite(z), z, 0.0)
        logc = self._log_copula_density(z, L, logdet)
        out = float(logc + logm)
        return out if np.isfinite(out) else float("-inf")
    
    def ask(self) -> dict:
        """Return one candidate configuration."""
        if self._pending:
            return self._pending.pop(0)

        if self.cfg.mode == "elite":
            return self._ask_elite()
        if self.cfg.mode == "elite_ratio":
            return self._ask_elite_ratio()
        return self._ask_latent_cma()

    def _ask_elite(self) -> dict:
        """Original v2 behavior: fit on elite, sample, return a small pending batch."""
        n = len(self.y_hist)
        n_min = max(10, 2 * self.d)

        # Exploration
        if n < n_min or self.rng.random() < self.cfg.eps_explore:
            return self._random_sample()

        # Get elite
        elite_arrays, n_elite = self._get_elite_arrays()
        if n_elite < 3:
            return self._random_sample()

        # Fit marginals and transform to uniform
        U = np.zeros((n_elite, self.d))
        for j in range(self.d):
            self.marginals[j].fit(elite_arrays[j])
            U[:, j] = self.marginals[j].to_uniform(elite_arrays[j])

        # Gaussianize
        U = np.clip(U, self.cfg.u_clip, 1.0 - self.cfg.u_clip)
        Z = _norm_ppf(U)
        Z = Z - Z.mean(axis=0, keepdims=True)

        # Estimate correlation matrix
        if n_elite > 1:
            cov = np.cov(Z.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
        else:
            cov = np.eye(self.d)

        # Standardize to correlation
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        # Shrinkage
        alpha = max(self.cfg.alpha_corr, min(0.9, self.d / max(n_elite, 1)))
        corr = (1.0 - alpha) * corr + alpha * np.eye(self.d)
        corr = corr + self.cfg.reg * np.eye(self.d)
        corr = _nearest_pd(corr, min_eig=self.cfg.reg)

        # Cholesky
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            return self._random_sample()

        # Sample candidates
        candidates = []
        for _ in range(self.cfg.M):
            z = self.rng.normal(0, 1, self.d) @ L.T
            u = _norm_cdf(z)
            u = np.clip(u, self.cfg.u_clip, 1.0 - self.cfg.u_clip)

            # Transform back
            x = {}
            for j, spec in enumerate(self.param_specs):
                val = self.marginals[j].from_uniform(np.array([u[j]]))[0]
                x[spec.name] = val
            candidates.append(x)

        # Deduplicate and select top
        seen = set()
        selected = []
        for x in candidates:
            key = tuple(sorted(x.items()))
            if key not in seen:
                seen.add(key)
                selected.append(x)
                if len(selected) >= self.cfg.top_eval:
                    break

        if not selected:
            return self._random_sample()

        self._pending.extend(selected[1:])
        return selected[0]

    def _ask_elite_ratio(self) -> dict:
        """
        Copula-TPE style:
          - fit l(x)=p(x|elite) and g(x)=p(x|non-elite) as two copula models
          - sample candidates from l
          - select by score log l(x) - log g(x)
        """
        n = len(self.y_hist)
        n_min = max(10, 2 * self.d)

        if n < n_min or self.rng.random() < self.cfg.eps_explore:
            return self._random_sample()

        tau = np.quantile(self.y_hist, self.cfg.gamma)
        y_arr = np.asarray(self.y_hist, dtype=float)
        elite_idx = np.flatnonzero(y_arr <= tau)
        non_idx = np.flatnonzero(y_arr > tau)

        elite_arrays, n_elite = self._arrays_from_indices(elite_idx)
        non_arrays, n_non = self._arrays_from_indices(non_idx)
        if n_elite < 3 or n_non < 3:
            # Not enough data to contrast; fall back to elite sampling.
            return self._ask_elite()

        model_E = self._fit_copula_model(elite_arrays, n_elite)
        model_N = self._fit_copula_model(non_arrays, n_non)
        if model_E is None or model_N is None:
            return self._ask_elite()
        mE, LE, logdetE = model_E
        mN, LN, logdetN = model_N

        # Sample candidates from elite model and score with log-ratio.
        candidates: list[dict] = []
        scores: list[float] = []
        for _ in range(int(self.cfg.M)):
            z = self.rng.normal(0, 1, self.d) @ LE.T
            u = _norm_cdf(z)
            u = np.clip(u, self.cfg.u_clip, 1.0 - self.cfg.u_clip)

            x = {}
            for j, spec in enumerate(self.param_specs):
                val = mE[j].from_uniform(np.asarray([u[j]], dtype=float))[0]
                if spec.type == "continuous":
                    x[spec.name] = float(val)
                elif spec.type == "integer":
                    x[spec.name] = int(val)
                else:
                    x[spec.name] = val

            lpE = self._log_joint_density(x, mE, LE, logdetE)
            lpN = self._log_joint_density(x, mN, LN, logdetN)
            score = float(lpE - lpN)
            if not np.isfinite(score):
                score = float("-inf")
            candidates.append(x)
            scores.append(score)

        if not candidates:
            return self._random_sample()

        order = np.argsort(np.asarray(scores, dtype=float))[::-1]

        seen = set()
        selected: list[dict] = []
        for idx in order:
            x = candidates[int(idx)]
            key = self._x_key(x)
            if key in seen:
                continue
            seen.add(key)
            selected.append(x)
            if len(selected) >= int(self.cfg.top_eval):
                break

        if not selected:
            return self._random_sample()

        self._pending.extend(selected[1:])
        return selected[0]

    def _ask_latent_cma(self) -> dict:
        """
        Latent-CMA: CMA-ES-style state (m, C, sigma, paths) in z-space, while
        keeping mixed-space marginal handling via per-dimension marginals.

        This runs in generations: we sample a (deduplicated) batch of candidates,
        then update CMA once all are evaluated through tell().
        """
        n = len(self.y_hist)
        n_min = max(10, 2 * self.d)

        # Cold start + explicit exploration.
        if n < n_min or self.rng.random() < self.cfg.eps_explore:
            return self._random_sample()

        # If a generation is mid-flight but we ran out of pending, fall back.
        if self._gen_expected > 0 and self._gen_received < self._gen_expected:
            return self._random_sample()

        # Fit marginals from elite history.
        n_elite = self._fit_marginals_from_elite()
        if n_elite < 3:
            return self._random_sample()

        # Initialize CMA state once we have a reasonable model.
        if self._cma is None:
            m0 = np.zeros(self.d, dtype=float)
            if self.best_x is not None:
                try:
                    m0 = self._x_to_z(self.best_x)
                except Exception:
                    m0 = np.zeros(self.d, dtype=float)
            self._cma = _LatentCMAState(
                m=m0,
                C=np.eye(self.d, dtype=float),
                sigma=float(self.cfg.cma_sigma0),
                p_c=np.zeros(self.d, dtype=float),
                p_sigma=np.zeros(self.d, dtype=float),
                gen=0,
            )

        self._sample_generation()
        if self._pending:
            return self._pending.pop(0)
        return self._random_sample()

    def _x_key(self, x: dict) -> tuple:
        """Stable key for a configuration (used to match ask()/tell())."""
        items: list[tuple[str, Any]] = []
        for spec in self.param_specs:
            v = x.get(spec.name)
            if spec.type == "continuous":
                try:
                    v = float(v)
                except Exception:
                    v = float("nan")
                v = float(np.round(v, 12))
            elif spec.type == "integer":
                try:
                    v = int(v)
                except Exception:
                    v = 0
            items.append((spec.name, v))
        return tuple(items)

    def _x_to_z(self, x: dict) -> np.ndarray:
        """Map a config x to its latent z via current fitted marginals."""
        z = np.zeros(self.d, dtype=float)
        for j, spec in enumerate(self.param_specs):
            v = x[spec.name]
            if spec.type == "categorical":
                arr = np.asarray([v], dtype=object)
            elif spec.type == "integer":
                arr = np.asarray([int(v)], dtype=int)
            else:
                arr = np.asarray([float(v)], dtype=float)

            u = float(self.marginals[j].to_uniform(arr)[0])
            u = float(np.clip(u, self.cfg.u_clip, 1.0 - self.cfg.u_clip))
            z_j = float(_norm_ppf(np.asarray([u], dtype=float))[0])
            z[j] = 0.0 if not np.isfinite(z_j) else z_j
        return z

    def _fit_marginals_from_elite(self) -> int:
        elite_arrays, n_elite = self._get_elite_arrays()
        if n_elite < 1:
            return 0
        for j in range(self.d):
            self.marginals[j].fit(elite_arrays[j])
        return int(n_elite)

    def _sample_generation(self) -> None:
        """Sample a generation of candidates from the current latent-CMA state."""
        if self._cma is None:
            self._gen_expected = 0
            return

        self._pending.clear()
        self._latent_by_key.clear()
        self._gen_z.clear()
        self._gen_y.clear()
        self._gen_received = 0

        lambda_target = int(max(2, self.cfg.M))

        C = np.asarray(self._cma.C, dtype=float)
        C = 0.5 * (C + C.T)
        C = C + float(self.cfg.reg) * np.eye(self.d)

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = _nearest_pd(C, min_eig=float(self.cfg.reg))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                self._gen_expected = 0
                return

        seen = set()
        tries = 0
        max_tries = max(500, lambda_target * 50)

        while len(self._pending) < lambda_target and tries < max_tries:
            tries += 1
            arz = self.rng.normal(0.0, 1.0, size=self.d)
            z = self._cma.m + float(self._cma.sigma) * (arz @ L.T)
            u = _norm_cdf(z)
            u = np.clip(u, self.cfg.u_clip, 1.0 - self.cfg.u_clip)

            x = {}
            z_adj = np.asarray(z, dtype=float).copy()
            for j, spec in enumerate(self.param_specs):
                val = self.marginals[j].from_uniform(np.asarray([u[j]], dtype=float))[0]
                if spec.type == "continuous":
                    val_f = float(val)
                    x[spec.name] = val_f
                    m = self.marginals[j]
                    if isinstance(m, ContinuousMarginal) and np.isfinite(m.sigma) and m.sigma > 0.0:
                        z_adj[j] = (val_f - float(m.mu)) / float(m.sigma)
                elif spec.type == "integer":
                    x[spec.name] = int(val)
                else:
                    x[spec.name] = val

            key = self._x_key(x)
            if key in seen:
                continue
            seen.add(key)
            self._pending.append(x)
            self._latent_by_key.setdefault(key, []).append(z_adj)

        self._gen_expected = int(len(self._pending))
        if self._gen_expected < 2:
            self._pending.clear()
            self._latent_by_key.clear()
            self._gen_expected = 0

    def _cma_update(self) -> None:
        """Update latent-CMA state after finishing a generation."""
        if self._cma is None:
            return

        y = np.asarray(self._gen_y, dtype=float)
        Z = np.asarray(self._gen_z, dtype=float)
        if y.size < 2 or Z.shape[0] < 2:
            self._cma.gen += 1
            return

        order = np.argsort(y)
        lam = int(y.size)
        mu = int(self.cfg.cma_mu) if self.cfg.cma_mu is not None else max(1, lam // 2)
        mu = int(np.clip(mu, 1, lam))

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = weights / float(np.sum(weights))
        mu_eff = 1.0 / float(np.sum(weights * weights))

        d = int(self.d)
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        c1 = 2.0 / (((d + 1.3) ** 2) + mu_eff)
        c_mu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (((d + 2.0) ** 2) + mu_eff),
        )
        d_sigma = 1.0 + 2.0 * max(0.0, float(np.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0)) + c_sigma
        chi_n = float(np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d)))

        m_old = np.asarray(self._cma.m, dtype=float)
        sigma = float(self._cma.sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(self.cfg.cma_sigma0)

        C = np.asarray(self._cma.C, dtype=float)
        C = 0.5 * (C + C.T)
        C = C + float(self.cfg.reg) * np.eye(d)
        C = _nearest_pd(C, min_eig=float(self.cfg.reg))

        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(_nearest_pd(C, min_eig=float(self.cfg.reg)))

        Z_sel = Z[order[:mu]]
        m_new = np.sum(Z_sel * weights[:, None], axis=0)
        y_w = (m_new - m_old) / sigma

        inv_sqrt_yw = np.linalg.solve(L, y_w)
        p_sigma = (1.0 - c_sigma) * self._cma.p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * inv_sqrt_yw

        norm_p_sigma = float(np.linalg.norm(p_sigma))
        gen = int(self._cma.gen)
        denom = float(np.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (gen + 1))))
        h_sigma = 1.0 if (norm_p_sigma / denom) < ((1.4 + 2.0 / (d + 1.0)) * chi_n) else 0.0

        p_c = (1.0 - c_c) * self._cma.p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w

        y_k = (Z_sel - m_old[None, :]) / sigma
        rank_mu = np.zeros((d, d), dtype=float)
        for i in range(mu):
            yi = y_k[i]
            rank_mu += weights[i] * np.outer(yi, yi)

        rank_mu_neg = None
        eta_neg = float(self.cfg.cma_active_eta)
        if bool(self.cfg.cma_active) and np.isfinite(eta_neg) and eta_neg > 0.0:
            mu_neg = int(min(mu, max(0, lam - mu)))
            if mu_neg >= 1:
                weights_neg = np.log(mu_neg + 0.5) - np.log(np.arange(1, mu_neg + 1, dtype=float))
                weights_neg = weights_neg / float(np.sum(weights_neg))
                Z_worst = Z[order[-mu_neg:]]
                y_k_neg = (Z_worst - m_old[None, :]) / sigma
                rank_mu_neg = np.zeros((d, d), dtype=float)
                max_norm2 = float(2.0 * d)
                for i in range(mu_neg):
                    yi = y_k_neg[i]
                    try:
                        wyi = np.linalg.solve(L, yi)
                        norm2 = float(np.dot(wyi, wyi))
                    except Exception:
                        norm2 = float("nan")
                    if np.isfinite(norm2) and norm2 > max_norm2 and norm2 > 0.0:
                        yi = yi * float(np.sqrt(max_norm2 / norm2))
                    rank_mu_neg += weights_neg[i] * np.outer(yi, yi)

        C_new = (1.0 - c1 - c_mu) * C
        C_new += c1 * (np.outer(p_c, p_c) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * C)
        C_new += c_mu * rank_mu
        if rank_mu_neg is not None:
            C_new -= (c_mu * eta_neg) * rank_mu_neg
        C_new = 0.5 * (C_new + C_new.T)
        C_new = _nearest_pd(C_new, min_eig=float(self.cfg.reg))

        sigma_new = sigma * float(np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1.0)))
        sigma_new = float(np.clip(sigma_new, 1e-12, 1e6))

        self._cma.m = m_new
        self._cma.C = C_new
        self._cma.sigma = sigma_new
        self._cma.p_c = p_c
        self._cma.p_sigma = p_sigma
        self._cma.gen = gen + 1
    
    def tell(self, x: dict, y: float) -> None:
        """Record an observation."""
        self.X_hist.append(x.copy())
        self.y_hist.append(float(y))
        
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()

        if self.cfg.mode != "latent_cma":
            return

        key = self._x_key(x)
        z_list = self._latent_by_key.get(key)
        if not z_list:
            return

        z = z_list.pop(0)
        if not z_list:
            self._latent_by_key.pop(key, None)

        self._gen_z.append(z)
        self._gen_y.append(float(y))
        self._gen_received += 1

        if self._gen_expected > 0 and self._gen_received >= self._gen_expected:
            self._cma_update()
            self._gen_expected = 0
            self._gen_received = 0
            self._gen_z.clear()
            self._gen_y.clear()
            self._latent_by_key.clear()


# =============================================================================
# Convenience wrapper for pure-continuous problems
# =============================================================================

class CopulaHPO_Continuous:
    """Wrapper for pure-continuous problems with bounds-only interface."""
    
    def __init__(self, bounds, seed: int = 0, **kwargs):
        bounds_arr = np.asarray(bounds, dtype=float)
        self.d = bounds_arr.shape[0]
        self.lower = bounds_arr[:, 0]
        self.upper = bounds_arr[:, 1]
        
        specs = [
            HyperparameterSpec(
                name=f"x{i}",
                type="continuous",
                bounds=(self.lower[i], self.upper[i])
            )
            for i in range(self.d)
        ]
        
        self.opt = CopulaHPO(specs, seed=seed, **kwargs)
        self.best_x = None
        self.best_y = float("inf")
    
    def ask(self) -> np.ndarray:
        x_dict = self.opt.ask()
        return np.array([x_dict[f"x{i}"] for i in range(self.d)])
    
    def tell(self, x: np.ndarray, y: float) -> None:
        x_dict = {f"x{i}": x[i] for i in range(self.d)}
        self.opt.tell(x_dict, y)
        
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing CopulaHPO v2...")
    
    # Test on mixed space
    specs = [
        HyperparameterSpec("lr", "continuous", (1e-5, 1e-1)),
        HyperparameterSpec("optimizer", "categorical", ["adam", "sgd", "rmsprop"]),
        HyperparameterSpec("n_layers", "integer", (1, 5)),
        HyperparameterSpec("dropout", "continuous", (0.0, 0.5)),
    ]
    
    def dummy_objective(x: dict) -> float:
        # Best: lr=0.001, adam, 3 layers, dropout=0.2
        score = 0.0
        score += (np.log10(x["lr"]) + 3) ** 2  # best at lr=0.001
        score += {"adam": 0, "sgd": 1, "rmsprop": 0.5}[x["optimizer"]]
        score += (x["n_layers"] - 3) ** 2
        score += (x["dropout"] - 0.2) ** 2
        return score
    
    opt = CopulaHPO(specs, seed=42)
    
    for i in range(100):
        x = opt.ask()
        y = dummy_objective(x)
        opt.tell(x, y)
    
    print(f"Best y: {opt.best_y:.4f}")
    print(f"Best x: {opt.best_x}")
    print("Expected: lr≈0.001, optimizer=adam, n_layers=3, dropout≈0.2")
