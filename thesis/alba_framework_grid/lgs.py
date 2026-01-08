"""\
ALBA Framework - Local Gradient Surrogate (LGS) Module

This module contains the Local Gradient Surrogate (LGS) implementation.

The LGS model is a lightweight local surrogate used by ALBA inside each Cube:
- Weighted linear regression in normalized cube coordinates
- Provides a local gradient direction
- Provides Bayesian-style uncertainty via (X^T W X + λI)^{-1}

This implementation follows the original ALBA_V1 philosophy, but includes a few
practical improvements for local ranking quality and numerical stability:
- Fit around a reference point near the local optimum (not cube center)
- Explicit intercept in the linear model
- Avoid materializing W=diag(weights)
- Fit only on the most informative dimensions when dim is large
- Optional scalar sigma calibration (per leaf) for better UCB scaling
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

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

    n_pts = len(pairs)
    if n_pts < 4:
        return None

    all_pts = np.array([p for p, _ in pairs], dtype=float)
    all_scores = np.array([s for _, s in pairs], dtype=float)

    k = max(3, n_pts // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]

    gradient_dir = None
    grad = None
    inv_cov = None
    bias = 0.0
    noise_var = 1.0

    widths = np.maximum(cube.widths(), 1e-9)
    dim = int(dim)
    lo = np.array([a for a, _ in cube.bounds], dtype=float)
    hi = np.array([b for _, b in cube.bounds], dtype=float)

    # Reference point: weighted mean of top-k points (more robust than cube center).
    top_scores = all_scores[top_k_idx]
    w_ref = top_scores - float(np.min(top_scores)) + 1e-12
    if not np.all(np.isfinite(w_ref)) or float(np.sum(w_ref)) <= 0.0:
        w_ref = np.ones_like(top_scores, dtype=float)
    x_ref = np.average(top_k_pts, axis=0, weights=w_ref)
    x_ref = np.minimum(np.maximum(x_ref, lo), hi)

    # Normalize around x_ref (not cube center).
    X_norm_full = (all_pts - x_ref) / widths

    y_mean_raw = float(np.mean(all_scores))
    y_std = float(np.std(all_scores)) + 1e-6
    y_centered = (all_scores - y_mean_raw) / y_std
    bias = y_mean_raw

    try:
        # Active dims: reduce overfitting when points are few vs dim.
        if n_pts >= 3 * dim:
            active_dims = np.arange(dim, dtype=np.int64)
        else:
            target_p = min(dim, max(5, int(np.floor(np.sqrt(max(n_pts, 1))))))
            p = int(min(target_p, max(1, n_pts - 2)))
            if p >= dim:
                active_dims = np.arange(dim, dtype=np.int64)
            else:
                yc = y_centered
                y_std_c = float(np.std(yc))
                scores = np.zeros(dim, dtype=float)
                if y_std_c > 1e-12:
                    for j in range(dim):
                        xj = X_norm_full[:, j]
                        x_std = float(np.std(xj))
                        if x_std <= 1e-12:
                            continue
                        cov = float(np.mean((xj - float(np.mean(xj))) * yc))
                        scores[j] = abs(cov / (x_std * y_std_c + 1e-12))
                active_dims = np.argsort(-scores)[:p].astype(np.int64)

        X_act = X_norm_full[:, active_dims]

        # Distance-based weights around x_ref (in active space).
        dists_sq = np.sum(X_act * X_act, axis=1)
        sigma_sq = float(np.mean(dists_sq)) + 1e-6
        weights = np.exp(-dists_sq / (2.0 * sigma_sq))

        # Boost weights for top performers.
        ptp = float(np.ptp(all_scores))
        rank_weights = 1.0 + 0.5 * (all_scores - float(np.min(all_scores))) / (ptp + 1e-9)
        weights = weights * rank_weights
        weights = np.clip(weights, 1e-12, np.inf)

        # Design matrix with explicit intercept.
        Phi = np.concatenate([np.ones((n_pts, 1), dtype=float), X_act], axis=1)
        n_feat = int(Phi.shape[1])

        # Weighted ridge without forming diag(W).
        PhiT_W_Phi = Phi.T @ (Phi * weights[:, None])
        PhiT_W_y = Phi.T @ (weights * y_centered)

        lambda_base = 0.1 * (1.0 + float(n_feat) / max(float(n_pts - n_feat), 1.0))
        reg = lambda_base * np.eye(n_feat, dtype=float)
        reg[0, 0] = lambda_base * 1e-3  # very light intercept regularization for stability

        try:
            cond = np.linalg.cond(PhiT_W_Phi + reg)
            if cond > 1e6:
                lambda_base *= 10.0
                reg = lambda_base * np.eye(n_feat, dtype=float)
                reg[0, 0] = lambda_base * 1e-3
        except Exception:
            lambda_base *= 5.0
            reg = lambda_base * np.eye(n_feat, dtype=float)
            reg[0, 0] = lambda_base * 1e-3

        inv_cov = np.linalg.inv(PhiT_W_Phi + reg)
        theta = inv_cov @ PhiT_W_y  # in y_centered units

        bias = y_mean_raw + y_std * float(theta[0])
        grad_act = y_std * theta[1:]
        grad_full = np.zeros(dim, dtype=float)
        grad_full[active_dims] = grad_act
        grad = grad_full

        y_pred_centered = Phi @ theta
        residuals = y_centered - y_pred_centered
        noise_var = float(
            np.clip(
                np.average(residuals**2, weights=weights) * (y_std**2) + 1e-6,
                1e-4,
                10.0,
            )
        )

        grad_norm = float(np.linalg.norm(grad_full))
        if grad_norm > 1e-9:
            gradient_dir = grad_full / grad_norm

        # Bonus: scalar sigma calibration via EMA of (err^2)/(sigma^2).
        model_var = np.clip(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0, 10.0)
        pred_var = noise_var * (1.0 + model_var)
        mu_train = bias + X_act @ grad_act
        err2 = (all_scores - mu_train) ** 2

        num = float(np.mean(err2))
        den = float(np.mean(pred_var))
        var_y = float(np.var(all_scores))
        rel_mse = float(num / (var_y + 1e-12))

        prev = getattr(cube, "lgs_model", None)
        prev_num = float(prev.get("sigma_ema_num")) if isinstance(prev, dict) and "sigma_ema_num" in prev else None
        prev_den = float(prev.get("sigma_ema_den")) if isinstance(prev, dict) and "sigma_ema_den" in prev else None

        ema_alpha = 0.10
        if prev_num is None or prev_den is None:
            ema_num = num
            ema_den = den
        else:
            ema_num = (1.0 - ema_alpha) * prev_num + ema_alpha * num
            ema_den = (1.0 - ema_alpha) * prev_den + ema_alpha * den

        sigma_scale = float(np.sqrt(max(ema_num, 1e-12) / (max(ema_den, 1e-12))))
        sigma_scale = float(np.clip(sigma_scale, 0.2, 5.0))
    except Exception:
        active_dims = np.arange(dim, dtype=np.int64)
        sigma_scale = 1.0
        ema_num = None
        ema_den = None
        rel_mse = None
        var_y = None

    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,
        "inv_cov": inv_cov,
        "y_mean": float(bias),
        "noise_var": noise_var,
        "widths": widths,
        "center": x_ref,
        "active_dims": active_dims,
        "sigma_scale": float(sigma_scale),
        "sigma_ema_num": (float(ema_num) if ema_num is not None else None),
        "sigma_ema_den": (float(ema_den) if ema_den is not None else None),
        "rel_mse": (float(rel_mse) if rel_mse is not None else None),
        "var_y": (float(var_y) if var_y is not None else None),
    }


CandidatesLike = Union[Sequence[np.ndarray], np.ndarray]


def predict_bayesian(model: Optional[Dict], candidates: CandidatesLike) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty for candidate points given an LGS model."""
    C = np.asarray(candidates, dtype=float)
    if C.size == 0:
        return np.zeros(0, dtype=float), np.ones(0, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape {tuple(C.shape)}")

    if model is None or model.get("inv_cov") is None:
        return np.zeros(C.shape[0], dtype=float), np.ones(C.shape[0], dtype=float)

    widths = model["widths"]
    center = model["center"]
    grad = model["grad"]
    inv_cov = model["inv_cov"]
    noise_var = model["noise_var"]
    y_mean = model["y_mean"]
    active_dims = model.get("active_dims")
    sigma_scale = float(model.get("sigma_scale", 1.0))

    C_norm = (C - center) / widths
    mu = y_mean + C_norm @ grad

    if active_dims is None:
        # Backward-compatible path (no intercept, full-dim covariance).
        model_var = np.clip(np.sum((C_norm @ inv_cov) * C_norm, axis=1), 0.0, 10.0)
    else:
        idx = np.asarray(active_dims, dtype=np.int64)
        X_act = C_norm[:, idx] if idx.size else np.zeros((C_norm.shape[0], 0), dtype=float)
        Phi = np.concatenate([np.ones((C_norm.shape[0], 1), dtype=float), X_act], axis=1)
        model_var = np.clip(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0, 10.0)

    total_var = noise_var * (1.0 + model_var)
    sigma = np.sqrt(total_var) * sigma_scale

    return mu, sigma
