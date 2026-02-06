"""\
ALBA Framework - Local Gradient Surrogate (LGS) Module

This module contains the Local Gradient Surrogate (LGS) implementation.

The LGS model is a lightweight local surrogate used by ALBA inside each Cube:
- Weighted linear regression in normalized cube coordinates
- Provides a local gradient direction
- Provides Bayesian-style uncertainty via (X^T W X + λI)^{-1}

This module is intentionally kept numerically identical to the original
ALBA_V1 implementation (same formulas, same thresholds).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cube import Cube


def fit_lgs_model(
    cube: "Cube",
    gamma: float,
    dim: int,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict]:
    """Fit the LGS model for a given cube.

    Notes
    -----
    - This is a direct extraction of the original Cube.fit_lgs_model logic.
    - Returns a dict model compatible with existing code.
    """

    pairs = list(cube.tested_pairs)

    # Parent backfill with shuffle for diversity
    if cube.parent and len(pairs) < 3 * dim:
        parent_pairs = getattr(cube.parent, "_tested_pairs", [])
        extra = [pp for pp in parent_pairs if cube.contains(pp[0])]
        needed = 3 * dim - len(pairs)
        if needed > 0 and extra:
            if rng is not None:
                extra = list(extra)
                rng.shuffle(extra)
            pairs = pairs + extra[:needed]

    if len(pairs) < dim + 2:
        return None

    all_pts = np.array([p for p, s in pairs])
    all_scores = np.array([s for p, s in pairs])

    k = max(3, len(pairs) // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]

    gradient_dir = None
    grad = None
    inv_cov = None
    y_mean = 0.0
    y_std = 1.0  # Default for when we don't have enough points
    noise_var = 1.0

    widths = np.maximum(cube.widths(), 1e-9)
    center = cube.center()

    if len(pairs) >= dim + 3:
        X_norm = (all_pts - center) / widths
        y_mean = all_scores.mean()
        y_std = all_scores.std() + 1e-6
        y_centered = (all_scores - y_mean) / y_std

        try:
            dists_sq = np.sum(X_norm**2, axis=1)
            sigma_sq = np.mean(dists_sq) + 1e-6
            weights = np.exp(-dists_sq / (2 * sigma_sq))

            # Boost weights for top performers
            rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / (
                all_scores.ptp() + 1e-9
            )
            weights = weights * rank_weights
            W = np.diag(weights)

            # Adaptive regularization
            n_pts = len(pairs)
            lambda_base = 0.1 * (1 + dim / max(n_pts - dim, 1))
            XtWX = X_norm.T @ W @ X_norm

            try:
                cond = np.linalg.cond(XtWX + lambda_base * np.eye(dim))
                if cond > 1e6:
                    lambda_base *= 10
            except Exception:
                lambda_base *= 5

            XtWX_reg = XtWX + lambda_base * np.eye(dim)
            inv_cov = np.linalg.inv(XtWX_reg)
            # grad is in NORMALIZED space (unit variance) - no y_std multiplication!
            # This prevents explosion on ill-conditioned functions
            grad = inv_cov @ (X_norm.T @ W @ y_centered)

            y_pred = X_norm @ grad
            residuals = y_centered - y_pred
            # noise_var in normalized space (will be scaled in predict_bayesian)
            noise_var = np.clip(
                np.average(residuals**2, weights=weights) + 1e-6,
                1e-4,
                10.0,
            )

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                gradient_dir = grad / grad_norm
        except Exception:
            pass

    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,  # In normalized space!
        "inv_cov": inv_cov,
        "y_mean": y_mean,
        "y_std": y_std,  # Needed for de-normalization in predict_bayesian
        "noise_var": noise_var,  # In normalized space
        "widths": widths,
        "center": center,
    }


def predict_bayesian(model: Optional[Dict], candidates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty for candidate points given an LGS model.
    
    All internal computations are in normalized space, then de-normalized
    at the end using y_mean and y_std. This prevents gradient explosion
    on ill-conditioned functions.
    """
    if model is None or model.get("inv_cov") is None:
        return np.zeros(len(candidates)), np.ones(len(candidates))

    widths = model["widths"]
    center = model["center"]
    grad = model["grad"]  # In normalized space
    inv_cov = model["inv_cov"]
    noise_var = model["noise_var"]  # In normalized space
    y_mean = model["y_mean"]
    y_std = model.get("y_std", 1.0)  # Fallback for old models

    C_norm = (np.array(candidates) - center) / widths
    
    # Prediction in normalized space, then de-normalize
    mu_normalized = C_norm @ grad
    mu = y_mean + mu_normalized * y_std

    # Variance in normalized space, then scale by y_std^2
    model_var = np.clip(np.sum((C_norm @ inv_cov) * C_norm, axis=1), 0, 10.0)
    total_var_normalized = noise_var * (1.0 + model_var)
    sigma = np.sqrt(total_var_normalized) * y_std

    return mu, sigma
