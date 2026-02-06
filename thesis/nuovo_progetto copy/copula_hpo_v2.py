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
    gamma: float = 0.2
    M: int = 128
    top_eval: int = 4
    eps_explore: float = 0.05
    alpha_corr: float = 0.1
    reg: float = 1e-6
    u_clip: float = 1e-6
    seed: int = 0


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
        # Degenerate bounds are common in conditional spaces: keep it robust.
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
        gamma: float = 0.2,
        M: int = 128,
        top_eval: int = 4,
        eps_explore: float = 0.05,
        alpha_corr: float = 0.1,
        reg: float = 1e-6,
        u_clip: float = 1e-6,
        seed: int = 0,
    ):
        self.param_specs = param_specs
        self.d = len(param_specs)
        
        self.cfg = CopulaHPOConfig(
            gamma=gamma,
            M=M,
            top_eval=top_eval,
            eps_explore=eps_explore,
            alpha_corr=alpha_corr,
            reg=reg,
            u_clip=u_clip,
            seed=seed,
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
        
        # Build arrays per dimension (keep original types)
        n_elite = len(elite_idx)
        arrays = []
        for j, spec in enumerate(self.param_specs):
            col = []
            for idx in elite_idx:
                val = self.X_hist[idx][spec.name]
                col.append(val)
            # For continuous/integer: numpy array; for categorical: keep as list
            if spec.type == "categorical":
                arrays.append(np.array(col, dtype=object))
            else:
                arrays.append(np.array(col, dtype=float))
        
        return arrays, n_elite
    
    def ask(self) -> dict:
        """Return one candidate configuration."""
        if self._pending:
            return self._pending.pop(0)
        
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
    
    def tell(self, x: dict, y: float) -> None:
        """Record an observation."""
        self.X_hist.append(x.copy())
        self.y_hist.append(float(y))
        
        if y < self.best_y:
            self.best_y = float(y)
            self.best_x = x.copy()


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
