"""
ALBA Framework (Copula variant) - Local Copula Model

Replaces LGS (Local Gradient Surrogate) with a Copula-based model.
This model:
- Uses elite sampling (top gamma%) instead of gradient direction
- Fits Gaussian copula to capture correlations
- Samples from the copula to generate candidates
- Does NOT depend on smooth/differentiable surfaces

Works well on piecewise-constant surfaces (RF/XGBoost surrogates).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .cube import Cube


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


def fit_copula_model(
    cube: "Cube",
    gamma: float,
    dim: int,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict]:
    """Fit a Copula model for a given cube.
    
    Instead of linear regression (LGS), we:
    1. Take elite samples (top gamma% by score)
    2. Fit empirical marginals for each dimension
    3. Estimate Gaussian copula correlation matrix
    
    Returns a dict model for candidate generation and uncertainty estimation.
    """
    pairs = list(cube.tested_pairs)
    
    # Copula needs MORE points than LGS - aggressive backfill from ancestors
    min_points_copula = max(5 * dim, 30)  # Copula needs more data
    
    # Backfill from parent chain (not just direct parent)
    current = cube.parent
    while current is not None and len(pairs) < min_points_copula:
        parent_pairs = getattr(current, "_tested_pairs", [])
        extra = [pp for pp in parent_pairs if cube.contains(pp[0])]
        needed = min_points_copula - len(pairs)
        if needed > 0 and extra:
            if rng is not None:
                extra = list(extra)
                rng.shuffle(extra)
            pairs = pairs + extra[:needed]
        current = current.parent  # Go up the tree
    
    if len(pairs) < dim + 2:
        return None
    
    all_pts = np.array([p for p, s in pairs])
    all_scores = np.array([s for p, s in pairs])
    
    # Elite selection (top gamma%)
    n = len(all_scores)
    n_elite = max(3, int(np.ceil(gamma * n)))
    elite_idx = np.argsort(all_scores)[:n_elite]  # Lower is better (minimization)
    elite_pts = all_pts[elite_idx]
    elite_scores = all_scores[elite_idx]
    
    # Also keep top-k for perturbation strategy
    k = max(3, len(pairs) // 5)
    top_k_idx = np.argsort(all_scores)[:k]
    top_k_pts = all_pts[top_k_idx]
    
    # Fit empirical marginals per dimension
    widths = np.maximum(cube.widths(), 1e-9)
    center = cube.center()
    bounds = cube.bounds
    
    marginal_params = []  # (mu, sigma) for each dimension
    for d in range(dim):
        col = elite_pts[:, d]
        mu = np.mean(col)
        sigma = np.std(col)
        # Ensure sigma is not too small - use ~0.15 to match top-k exploration
        # Previously: widths[d] / 10.0 = 0.1 (too tight for noisy functions)
        # Now: widths[d] / 6.5 ≈ 0.154 (balanced exploitation/exploration)
        min_sigma = widths[d] / 6.5
        sigma = max(sigma, min_sigma)
        marginal_params.append((mu, sigma))
    
    # Transform elite to uniform via Gaussian CDF
    U = np.zeros((n_elite, dim))
    u_clip = 1e-6
    for d in range(dim):
        mu, sigma = marginal_params[d]
        U[:, d] = stats.norm.cdf(elite_pts[:, d], loc=mu, scale=sigma)
        U[:, d] = np.clip(U[:, d], u_clip, 1.0 - u_clip)
    
    # Gaussianize
    Z = _norm_ppf(U)
    Z = Z - Z.mean(axis=0, keepdims=True)
    
    # Estimate correlation matrix
    if n_elite > 1:
        cov = np.cov(Z.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
    else:
        cov = np.eye(dim)
    
    # Standardize to correlation
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    
    # Shrinkage towards identity
    alpha = max(0.1, min(0.9, dim / max(n_elite, 1)))
    reg = 1e-6
    corr = (1.0 - alpha) * corr + alpha * np.eye(dim)
    corr = corr + reg * np.eye(dim)
    corr = _nearest_pd(corr, min_eig=reg)
    
    # Cholesky for sampling
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.eye(dim)
    
    # Uncertainty: use spread of elite
    y_mean = np.mean(all_scores)
    noise_var = np.var(elite_scores) + 1e-6
    
    # ADDED: Compute gradient direction as elite_center - non_elite_center
    # This gives a direction pointing TOWARDS the good region
    non_elite_mask = np.ones(len(all_pts), dtype=bool)
    non_elite_mask[elite_idx] = False
    
    if non_elite_mask.sum() > 0:
        non_elite_pts = all_pts[non_elite_mask]
        elite_center = np.mean(elite_pts, axis=0)
        non_elite_center = np.mean(non_elite_pts, axis=0)
        
        # Direction from non-elite center to elite center
        gradient_dir = elite_center - non_elite_center
        grad_norm = np.linalg.norm(gradient_dir)
        if grad_norm > 1e-8:
            gradient_dir = gradient_dir / grad_norm  # Normalize
        else:
            gradient_dir = None
    else:
        gradient_dir = None
    
    return {
        "all_pts": all_pts,
        "all_scores": all_scores,  # Added for predict_copula
        "top_k_pts": top_k_pts,
        "elite_pts": elite_pts,
        "elite_scores": elite_scores,
        "marginal_params": marginal_params,  # [(mu, sigma), ...]
        "corr": corr,
        "L": L,  # Cholesky factor for sampling
        "y_mean": y_mean,
        "noise_var": noise_var,
        "widths": widths,
        "center": center,
        "bounds": bounds,
        # Gradient direction (elite - non_elite centroid)
        "gradient_dir": gradient_dir,
        "grad": gradient_dir,  # Alias for compatibility with LGS
        "inv_cov": None,
    }


def sample_from_copula(
    model: Optional[Dict],
    n_samples: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Sample n_samples points from the fitted copula model."""
    if model is None:
        return []
    
    dim = len(model["marginal_params"])
    L = model["L"]
    marginal_params = model["marginal_params"]
    bounds = model["bounds"]
    u_clip = 1e-6
    
    samples = []
    for _ in range(n_samples):
        # Sample from correlated Gaussian
        z = rng.normal(0, 1, dim) @ L.T
        u = _norm_cdf(z)
        u = np.clip(u, u_clip, 1.0 - u_clip)
        
        # Transform back via marginal inverse CDF
        x = np.zeros(dim)
        for d in range(dim):
            mu, sigma = marginal_params[d]
            x[d] = stats.norm.ppf(u[d], loc=mu, scale=sigma)
            # Clip to cube bounds
            x[d] = np.clip(x[d], bounds[d][0], bounds[d][1])
        
        samples.append(x)
    
    return samples


def predict_copula(
    model: Optional[Dict],
    candidates: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty for candidate points.
    
    For copula model, we use k-nearest neighbors from ALL points (not just elite):
    - mu: weighted average of k nearest neighbors
    - sigma: standard deviation of k nearest neighbors + distance penalty
    
    This is more robust on piecewise-constant surfaces (RF) than LGS linear model.
    """
    if model is None or model.get("all_pts") is None or len(model["all_pts"]) == 0:
        return np.zeros(len(candidates)), np.ones(len(candidates))
    
    all_pts = model["all_pts"]
    all_scores = model.get("all_scores", np.zeros(len(all_pts)))
    if "all_scores" not in model:
        # Reconstruct from pairs if needed
        all_scores = np.array([model.get("y_mean", 0.0)] * len(all_pts))
    
    widths = model["widths"]
    noise_var = model["noise_var"]
    
    candidates_arr = np.array(candidates)
    n_cand = len(candidates)
    n_pts = len(all_pts)
    
    # Use k nearest neighbors
    k = min(5, n_pts)
    
    # Get global score stats for normalization
    score_std = np.std(all_scores) if len(all_scores) > 1 else 1.0
    score_std = max(score_std, 0.01)  # Avoid division by zero
    
    mu = np.zeros(n_cand)
    sigma = np.zeros(n_cand)
    
    for i, cand in enumerate(candidates_arr):
        # Distance to each point (normalized)
        dists = np.linalg.norm((all_pts - cand) / widths, axis=1)
        
        # Get k nearest
        knn_idx = np.argsort(dists)[:k]
        knn_scores = all_scores[knn_idx]
        knn_dists = dists[knn_idx]
        
        # Weighted average (closer = higher weight)
        weights = 1.0 / (knn_dists + 0.01)
        weights = weights / weights.sum()
        
        mu[i] = np.dot(weights, knn_scores)
        
        # FIXED: Sigma should be SMALLER for explored regions, not bigger for unexplored
        # Use local variance of neighbors (capped) + small noise term
        if k > 1:
            local_var = np.std(knn_scores)
            # Cap sigma to prevent exploration dominating
            sigma[i] = min(local_var, 0.5 * score_std) + 0.01
        else:
            sigma[i] = np.sqrt(noise_var)
        
        sigma[i] = max(sigma[i], 0.01)  # Minimum uncertainty
    
    return mu, sigma
