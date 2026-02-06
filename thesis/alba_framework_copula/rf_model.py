"""ALBA Framework - Random Forest Surrogate Model

Uses a local Random Forest instead of LGS for piecewise-constant surfaces.
This should work better on RF surrogate benchmarks like ParamNet.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Lightweight RF from sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError


def fit_rf_model(
    cube: "Cube",
    gamma: float,
    dim: int,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict]:
    """Fit a Random Forest model for a given cube.
    
    Uses RF instead of linear regression for better handling of
    piecewise-constant surfaces (like RF surrogates in HPOBench).
    """
    pairs = list(cube.tested_pairs)
    
    # RF needs more data than LGS
    min_points = max(3 * dim, 15)
    
    # Backfill from parent chain
    current = cube.parent
    while current is not None and len(pairs) < min_points:
        parent_pairs = getattr(current, "_tested_pairs", [])
        extra = [pp for pp in parent_pairs if cube.contains(pp[0])]
        needed = min_points - len(pairs)
        if needed > 0 and extra:
            if rng is not None:
                extra = list(extra)
                rng.shuffle(extra)
            pairs = pairs + extra[:needed]
        current = current.parent
    
    if len(pairs) < dim + 2:
        return None
    
    all_pts = np.array([p for p, s in pairs])
    all_scores = np.array([s for p, s in pairs])
    
    # Elite selection
    n = len(all_scores)
    n_elite = max(3, int(np.ceil(gamma * n)))
    elite_idx = np.argsort(all_scores)[:n_elite]
    elite_pts = all_pts[elite_idx]
    elite_scores = all_scores[elite_idx]
    
    # Top-k for perturbation
    k = max(3, len(pairs) // 5)
    top_k_idx = np.argsort(all_scores)[:k]
    top_k_pts = all_pts[top_k_idx]
    
    widths = np.maximum(cube.widths(), 1e-9)
    center = cube.center()
    bounds = cube.bounds
    
    # Fit Random Forest
    seed = int(rng.integers(0, 2**31)) if rng is not None else 42
    rf = RandomForestRegressor(
        n_estimators=10,  # Small for speed
        max_depth=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )
    
    # Normalize inputs to [0, 1] for stability
    X_norm = (all_pts - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-9)
    rf.fit(X_norm, all_scores)
    
    # Compute gradient direction (elite - non_elite centroid)
    non_elite_mask = np.ones(len(all_pts), dtype=bool)
    non_elite_mask[elite_idx] = False
    
    if non_elite_mask.sum() > 0:
        non_elite_pts = all_pts[non_elite_mask]
        elite_center = np.mean(elite_pts, axis=0)
        non_elite_center = np.mean(non_elite_pts, axis=0)
        gradient_dir = elite_center - non_elite_center
        grad_norm = np.linalg.norm(gradient_dir)
        if grad_norm > 1e-8:
            gradient_dir = gradient_dir / grad_norm
        else:
            gradient_dir = None
    else:
        gradient_dir = None
    
    return {
        "rf": rf,
        "all_pts": all_pts,
        "all_scores": all_scores,
        "top_k_pts": top_k_pts,
        "elite_pts": elite_pts,
        "elite_scores": elite_scores,
        "y_mean": np.mean(all_scores),
        "noise_var": np.var(elite_scores) + 1e-6,
        "widths": widths,
        "center": center,
        "bounds": bounds,
        "gradient_dir": gradient_dir,
        "grad": gradient_dir,
        "inv_cov": None,
        # Copula compatibility
        "marginal_params": [(center[d], widths[d]) for d in range(dim)],
        "L": None,
    }


def predict_rf(
    model: Optional[Dict],
    candidates: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty using Random Forest."""
    if model is None or len(candidates) == 0:
        n = len(candidates) if candidates else 0
        return np.zeros(n), np.ones(n)
    
    rf = model["rf"]
    bounds = model["bounds"]
    noise_var = model["noise_var"]
    
    candidates_arr = np.array(candidates)
    n_cand = len(candidates_arr)
    
    # Normalize
    X_norm = (candidates_arr - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-9)
    
    # Predict with individual trees for uncertainty
    mu = rf.predict(X_norm)
    
    # Uncertainty from tree variance
    tree_preds = np.array([tree.predict(X_norm) for tree in rf.estimators_])
    sigma = np.std(tree_preds, axis=0)
    
    # Minimum uncertainty
    sigma = np.maximum(sigma, 0.01)
    
    return mu, sigma
