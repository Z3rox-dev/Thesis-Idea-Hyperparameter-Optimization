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


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    """Simple rank data for Spearman correlation (ties get arbitrary stable order)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(order.size, dtype=float)
    return ranks


def _spearmanr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or b.size != a.size:
        return 0.0
    ra = _rankdata_1d(a)
    rb = _rankdata_1d(b)
    ra = ra - float(np.mean(ra))
    rb = rb - float(np.mean(rb))
    denom = float(np.sqrt(float(np.sum(ra * ra)) * float(np.sum(rb * rb))))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def _loo_rank_metrics(
    y_centered: np.ndarray,
    y_pred_centered: np.ndarray,
    weights: np.ndarray,
    model_var: np.ndarray,
    scores_raw: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute LOOCV-based ranking metrics for deterministic model selection.

    Returns
    -------
    (loo_regret, loo_topk_overlap, loo_spearman)
        - loo_regret: (best_true - true_of_argmax(loo_pred)) in raw score units.
        - loo_topk_overlap: overlap fraction between top-k true and top-k loo_pred.
        - loo_spearman: Spearman correlation between loo_pred and true (centered units).
    """
    yc = np.asarray(y_centered, dtype=float).reshape(-1)
    yp = np.asarray(y_pred_centered, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mv = np.asarray(model_var, dtype=float).reshape(-1)
    yr = np.asarray(scores_raw, dtype=float).reshape(-1)

    n = int(yc.size)
    if n < 3 or yp.size != n or w.size != n or mv.size != n or yr.size != n:
        return float("inf"), 0.0, 0.0
    if (
        (not np.all(np.isfinite(yc)))
        or (not np.all(np.isfinite(yp)))
        or (not np.all(np.isfinite(w)))
        or (not np.all(np.isfinite(mv)))
        or (not np.all(np.isfinite(yr)))
    ):
        return float("inf"), 0.0, 0.0

    # Hat diagonal for weighted ridge:
    #   H_ii = w_i * phi_i^T (Phi^T W Phi + λI)^(-1) phi_i
    h = np.clip(w * mv, 0.0, 0.99)
    denom = np.maximum(1.0 - h, 1e-6)
    resid = yc - yp
    y_loo = yc - resid / denom

    i_best = int(np.argmax(y_loo))
    best_true = float(np.max(yr))
    chosen_true = float(yr[i_best])
    loo_regret = float(max(0.0, best_true - chosen_true))

    k = int(max(3, n // 5))
    true_top = set(np.argsort(yr)[-k:].tolist())
    pred_top = set(np.argsort(y_loo)[-k:].tolist())
    loo_topk_overlap = float(len(true_top & pred_top)) / float(k) if k > 0 else 0.0

    loo_spearman = _spearmanr_1d(y_loo, yc)
    return loo_regret, loo_topk_overlap, loo_spearman


def _poly2_features(X: np.ndarray) -> np.ndarray:
    """Degree-2 polynomial features with intercept; X shape (n, d)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X.shape
    feats: List[np.ndarray] = [np.ones((n, 1), dtype=float)]
    if d <= 0:
        return feats[0]
    feats.append(X)
    feats.append(X * X)
    if d >= 2:
        cross: List[np.ndarray] = []
        for i in range(d):
            for j in range(i + 1, d):
                cross.append((X[:, i] * X[:, j]).reshape(n, 1))
        if cross:
            feats.append(np.concatenate(cross, axis=1))
    return np.concatenate(feats, axis=1)


def _diag2_features(X: np.ndarray) -> np.ndarray:
    """Diagonal degree-2 features with intercept; X shape (n, d).

    Features are: [1, x_1..x_d, x_1^2..x_d^2] (no cross terms).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X.shape
    if d <= 0:
        return np.ones((n, 1), dtype=float)
    return np.concatenate([np.ones((n, 1), dtype=float), X, X * X], axis=1)


def _discretize_cat(x_val: float, n_choices: int) -> int:
    n = int(n_choices)
    if n <= 1:
        return 0
    idx = int(np.round(float(x_val) * float(n - 1)))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


def _cat_keys(X: np.ndarray, categorical_dims: Sequence[Tuple[int, int]]) -> List[Tuple[int, ...]]:
    X = np.asarray(X, dtype=float)
    n = int(X.shape[0])
    if not categorical_dims:
        return [tuple()] * n
    keys: List[Tuple[int, ...]] = []
    for i in range(n):
        key_parts: List[int] = []
        for dim_idx, n_choices in categorical_dims:
            key_parts.append(_discretize_cat(float(X[i, int(dim_idx)]), int(n_choices)))
        keys.append(tuple(key_parts))
    return keys


def _fit_poly2_model_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, float]],
    dim: int,
    bounds: Sequence[Tuple[float, float]],
    *,
    prev_model: Optional[Dict] = None,
    candidate_dims: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    pairs_l = list(pairs)
    n_pts = int(len(pairs_l))
    if n_pts < 4:
        return None

    all_pts = np.array([p for p, _ in pairs_l], dtype=float)
    all_scores = np.array([s for _, s in pairs_l], dtype=float)

    k = max(3, n_pts // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]

    gradient_dir = None
    grad = None
    inv_cov = None
    theta = None
    bias = 0.0
    noise_var = 1.0

    widths = np.maximum(np.array([abs(hi - lo) for lo, hi in bounds], dtype=float), 1e-9)
    dim = int(dim)
    lo = np.array([a for a, _ in bounds], dtype=float)
    hi = np.array([b for _, b in bounds], dtype=float)

    # Reference point: weighted mean of top-k points (more robust than cube center).
    top_scores = all_scores[top_k_idx]
    w_ref = top_scores - float(np.min(top_scores)) + 1e-12
    if not np.all(np.isfinite(w_ref)) or float(np.sum(w_ref)) <= 0.0:
        w_ref = np.ones_like(top_scores, dtype=float)
    x_ref = np.average(top_k_pts, axis=0, weights=w_ref)
    x_ref = np.minimum(np.maximum(x_ref, lo), hi)

    # Normalize around x_ref (not cube center).
    X_norm_full = (all_pts - x_ref) / widths

    # Global scale (used for robust dimension scoring); the actual fit uses a
    # weighted/local scale computed after weights are known.
    y_mean_global = float(np.mean(all_scores))
    y_std_global = float(np.std(all_scores)) + 1e-6
    y_centered_global = (all_scores - y_mean_global) / y_std_global

    y_mean_raw = float(y_mean_global)
    y_std = float(y_std_global)
    y_centered = y_centered_global
    bias = float(y_mean_raw)

    cand: Optional[np.ndarray]
    if candidate_dims is None:
        cand = None
    else:
        cd = np.asarray(candidate_dims, dtype=np.int64).ravel()
        cd = cd[(cd >= 0) & (cd < dim)]
        if cd.size:
            # Preserve order and uniqueness.
            seen = set()
            uniq: List[int] = []
            for j in cd.tolist():
                jj = int(j)
                if jj in seen:
                    continue
                seen.add(jj)
                uniq.append(jj)
            cand = np.asarray(uniq, dtype=np.int64)
        else:
            cand = np.zeros(0, dtype=np.int64)

    try:
        # Active dims: keep the quadratic feature count reasonable vs n_pts.
        yc = y_centered_global
        y_std_c = float(np.std(yc))

        dims_pool = np.arange(dim, dtype=np.int64) if cand is None else cand
        scores_pool = np.zeros(int(dims_pool.size), dtype=float)
        if y_std_c > 1e-12 and dims_pool.size:
            for i, j in enumerate(dims_pool.tolist()):
                xj = X_norm_full[:, int(j)]
                x_std = float(np.std(xj))
                if x_std <= 1e-12:
                    continue
                cov = float(np.mean((xj - float(np.mean(xj))) * yc))
                scores_pool[i] = abs(cov / (x_std * y_std_c + 1e-12))

        ranked_pool = dims_pool[np.argsort(-scores_pool)] if dims_pool.size else np.zeros(0, dtype=np.int64)

        def n_feat_poly2(p: int) -> int:
            # 1 (bias) + p (linear) + p (squares) + p*(p-1)/2 (cross)
            return int(1 + 2 * p + (p * (p - 1)) // 2)

        p_max = int(ranked_pool.size)
        if p_max <= 0:
            active_dims = np.zeros(0, dtype=np.int64)
        else:
            target_p = int(min(p_max, max(2, int(np.floor(np.sqrt(max(n_pts, 1)))))))
            p = int(min(p_max, max(1, target_p)))
            cap = int(max(6, n_pts))
            while p > 1 and n_feat_poly2(p) > cap:
                p -= 1
            active_dims = ranked_pool[:p].astype(np.int64)

        X_act = X_norm_full[:, active_dims] if active_dims.size else np.zeros((n_pts, 0), dtype=float)

        # Distance-based weights around x_ref (in active space).
        dists_sq = np.sum(X_act * X_act, axis=1) if X_act.size else np.zeros(n_pts, dtype=float)
        sigma_sq = float(np.mean(dists_sq)) + 1e-6
        weights = np.exp(-dists_sq / (2.0 * sigma_sq))

        # Boost weights for top performers.
        ptp = float(np.ptp(all_scores))
        rank_weights = 1.0 + 0.5 * (all_scores - float(np.min(all_scores))) / (ptp + 1e-9)
        weights = weights * rank_weights
        weights = np.clip(weights, 1e-12, np.inf)

        # Fit in a local score scale (weighted mean/std) to avoid inflated sigma on
        # objectives with large global range (e.g., Rosenbrock).
        y_mean_w = float(np.average(all_scores, weights=weights))
        y_var_w = float(np.average((all_scores - y_mean_w) ** 2, weights=weights))
        y_std_w = float(np.sqrt(max(y_var_w, 1e-12)))
        if (not np.isfinite(y_std_w)) or y_std_w <= 1e-12:
            y_mean_raw = float(y_mean_global)
            y_std = float(y_std_global)
        else:
            y_mean_raw = float(y_mean_w)
            y_std = float(y_std_w) + 1e-6
        y_centered = (all_scores - y_mean_raw) / y_std

        # Quadratic design matrix (poly2) with intercept.
        Phi = _poly2_features(X_act)
        n_feat = int(Phi.shape[1])

        # Weighted ridge without forming diag(W).
        PhiT_W_Phi = Phi.T @ (Phi * weights[:, None])
        PhiT_W_y = Phi.T @ (weights * y_centered)

        # Ridge regularization (scale-free):
        # Keep the base very small so that truly quadratic objectives can be
        # represented without systematic underfitting, while still allowing
        # the condition-number guard below to increase it when needed.
        lambda_base = 1e-3 * (1.0 + float(n_feat) / max(float(n_pts - n_feat), 1.0))
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

        bias = float(y_mean_raw + y_std * float(theta[0]))
        grad_act = y_std * theta[1 : 1 + int(active_dims.size)]
        grad_full = np.zeros(dim, dtype=float)
        if active_dims.size:
            grad_full[active_dims] = grad_act
        grad = grad_full

        y_pred_centered = Phi @ theta
        residuals = y_centered - y_pred_centered
        # Estimate residual variance in standardized units, then map back to raw scale.
        # Clipping in standardized units makes this robust across objectives with very
        # different raw scales (e.g., accuracy in [0,1] vs large-valued synthetic losses).
        noise_var_c = float(np.average(residuals**2, weights=weights))
        noise_var_c = float(np.clip(noise_var_c + 1e-12, 1e-6, 10.0))
        noise_var = float(noise_var_c * (y_std**2))

        grad_norm = float(np.linalg.norm(grad_full))
        if grad_norm > 1e-9:
            gradient_dir = grad_full / grad_norm

        # Scalar sigma calibration via EMA of (err^2)/(sigma^2).
        # Note: we keep a capped version for predictive variance, but LOOCV needs the *uncapped*
        # phi_i^T inv_cov phi_i (hat diagonal uses it).
        hat_comp = np.maximum(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0)
        model_var = np.clip(hat_comp, 0.0, 10.0)
        pred_var = noise_var * (1.0 + model_var)
        mu_train = y_mean_raw + y_std * (Phi @ theta)
        err2 = (all_scores - mu_train) ** 2

        num = float(np.mean(err2))
        den = float(np.mean(pred_var))
        var_y = float(np.var(all_scores))
        rel_mse = float(num / (var_y + 1e-12))

        loo_regret, loo_topk_overlap, loo_spearman = _loo_rank_metrics(
            y_centered=y_centered,
            y_pred_centered=y_pred_centered,
            weights=weights,
            model_var=hat_comp,
            scores_raw=all_scores,
        )

        # Generalization proxy (GCV) for deterministic model selection.
        try:
            df = float(np.trace(PhiT_W_Phi @ inv_cov))
            denom = 1.0 - df / float(max(n_pts, 1))
            gcv = float(num / max(denom * denom, 1e-12))
        except Exception:
            df = 0.0
            gcv = float("inf")

        prev_num = (
            float(prev_model.get("sigma_ema_num"))
            if isinstance(prev_model, dict) and prev_model.get("sigma_ema_num") is not None
            else None
        )
        prev_den = (
            float(prev_model.get("sigma_ema_den"))
            if isinstance(prev_model, dict) and prev_model.get("sigma_ema_den") is not None
            else None
        )

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
        active_dims = np.asarray(cand, dtype=np.int64) if cand is not None else np.arange(dim, dtype=np.int64)
        sigma_scale = 1.0
        ema_num = None
        ema_den = None
        rel_mse = None
        var_y = None
        loo_regret = float("inf")
        loo_topk_overlap = 0.0
        loo_spearman = 0.0
        df = 0.0
        gcv = float("inf")

    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,
        "inv_cov": inv_cov,
        "theta": (theta.copy() if isinstance(theta, np.ndarray) else None),
        "feature_kind": "poly2",
        "gcv": float(gcv),
        "df": float(df),
        "n_feat": (int(inv_cov.shape[0]) if isinstance(inv_cov, np.ndarray) else None),
        "y_mean_raw": float(y_mean_raw),
        "y_std": float(y_std),
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
        "loo_regret": float(loo_regret),
        "loo_topk_overlap": float(loo_topk_overlap),
        "loo_spearman": float(loo_spearman),
    }


def _fit_diag2_model_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, float]],
    dim: int,
    bounds: Sequence[Tuple[float, float]],
    *,
    prev_model: Optional[Dict] = None,
    candidate_dims: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Fit a diagonal quadratic surrogate (no cross terms)."""
    pairs_l = list(pairs)
    n_pts = int(len(pairs_l))
    if n_pts < 4:
        return None

    all_pts = np.array([p for p, _ in pairs_l], dtype=float)
    all_scores = np.array([s for _, s in pairs_l], dtype=float)

    k = max(3, n_pts // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]

    gradient_dir = None
    grad = None
    inv_cov = None
    theta = None
    bias = 0.0
    noise_var = 1.0

    widths = np.maximum(np.array([abs(hi - lo) for lo, hi in bounds], dtype=float), 1e-9)
    dim = int(dim)
    lo = np.array([a for a, _ in bounds], dtype=float)
    hi = np.array([b for _, b in bounds], dtype=float)

    # Reference point: weighted mean of top-k points.
    top_scores = all_scores[top_k_idx]
    w_ref = top_scores - float(np.min(top_scores)) + 1e-12
    if not np.all(np.isfinite(w_ref)) or float(np.sum(w_ref)) <= 0.0:
        w_ref = np.ones_like(top_scores, dtype=float)
    x_ref = np.average(top_k_pts, axis=0, weights=w_ref)
    x_ref = np.minimum(np.maximum(x_ref, lo), hi)

    X_norm_full = (all_pts - x_ref) / widths

    y_mean_global = float(np.mean(all_scores))
    y_std_global = float(np.std(all_scores)) + 1e-6
    y_centered_global = (all_scores - y_mean_global) / y_std_global

    y_mean_raw = float(y_mean_global)
    y_std = float(y_std_global)
    y_centered = y_centered_global
    bias = float(y_mean_raw)

    cand: Optional[np.ndarray]
    if candidate_dims is None:
        cand = None
    else:
        cd = np.asarray(candidate_dims, dtype=np.int64).ravel()
        cd = cd[(cd >= 0) & (cd < dim)]
        if cd.size:
            seen = set()
            uniq: List[int] = []
            for j in cd.tolist():
                jj = int(j)
                if jj in seen:
                    continue
                seen.add(jj)
                uniq.append(jj)
            cand = np.asarray(uniq, dtype=np.int64)
        else:
            cand = np.zeros(0, dtype=np.int64)

    try:
        yc = y_centered_global
        y_std_c = float(np.std(yc))

        dims_pool = np.arange(dim, dtype=np.int64) if cand is None else cand
        scores_pool = np.zeros(int(dims_pool.size), dtype=float)
        if y_std_c > 1e-12 and dims_pool.size:
            for i, j in enumerate(dims_pool.tolist()):
                xj = X_norm_full[:, int(j)]
                x_std = float(np.std(xj))
                if x_std <= 1e-12:
                    continue
                cov = float(np.mean((xj - float(np.mean(xj))) * yc))
                scores_pool[i] = abs(cov / (x_std * y_std_c + 1e-12))

        ranked_pool = dims_pool[np.argsort(-scores_pool)] if dims_pool.size else np.zeros(0, dtype=np.int64)

        def n_feat_diag2(p: int) -> int:
            return int(1 + 2 * p)

        p_max = int(ranked_pool.size)
        if p_max <= 0:
            active_dims = np.zeros(0, dtype=np.int64)
        else:
            base = int(max(2, int(np.floor(np.sqrt(max(n_pts, 1)))))) if n_pts > 0 else 2
            target_p = int(min(p_max, max(2, 2 * base)))
            p = int(min(p_max, max(1, target_p)))
            cap = int(max(6, n_pts))
            while p > 1 and n_feat_diag2(p) > cap:
                p -= 1
            active_dims = ranked_pool[:p].astype(np.int64)

        X_act = X_norm_full[:, active_dims] if active_dims.size else np.zeros((n_pts, 0), dtype=float)

        dists_sq = np.sum(X_act * X_act, axis=1) if X_act.size else np.zeros(n_pts, dtype=float)
        sigma_sq = float(np.mean(dists_sq)) + 1e-6
        weights = np.exp(-dists_sq / (2.0 * sigma_sq))

        ptp = float(np.ptp(all_scores))
        rank_weights = 1.0 + 0.5 * (all_scores - float(np.min(all_scores))) / (ptp + 1e-9)
        weights = weights * rank_weights
        weights = np.clip(weights, 1e-12, np.inf)

        y_mean_w = float(np.average(all_scores, weights=weights))
        y_var_w = float(np.average((all_scores - y_mean_w) ** 2, weights=weights))
        y_std_w = float(np.sqrt(max(y_var_w, 1e-12)))
        if (not np.isfinite(y_std_w)) or y_std_w <= 1e-12:
            y_mean_raw = float(y_mean_global)
            y_std = float(y_std_global)
        else:
            y_mean_raw = float(y_mean_w)
            y_std = float(y_std_w) + 1e-6
        y_centered = (all_scores - y_mean_raw) / y_std

        Phi = _diag2_features(X_act)
        n_feat = int(Phi.shape[1])

        PhiT_W_Phi = Phi.T @ (Phi * weights[:, None])
        PhiT_W_y = Phi.T @ (weights * y_centered)

        # See _fit_poly2_model_from_pairs for rationale.
        lambda_base = 1e-3 * (1.0 + float(n_feat) / max(float(n_pts - n_feat), 1.0))
        reg = lambda_base * np.eye(n_feat, dtype=float)
        reg[0, 0] = lambda_base * 1e-3

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
        theta = inv_cov @ PhiT_W_y

        bias = float(y_mean_raw + y_std * float(theta[0]))
        p = int(active_dims.size)
        grad_act = y_std * theta[1 : 1 + p]
        grad_full = np.zeros(dim, dtype=float)
        if p:
            grad_full[active_dims] = grad_act
        grad = grad_full

        y_pred_centered = Phi @ theta
        residuals = y_centered - y_pred_centered
        noise_var_c = float(np.average(residuals**2, weights=weights))
        noise_var_c = float(np.clip(noise_var_c + 1e-12, 1e-6, 10.0))
        noise_var = float(noise_var_c * (y_std**2))

        grad_norm = float(np.linalg.norm(grad_full))
        if grad_norm > 1e-9:
            gradient_dir = grad_full / grad_norm

        hat_comp = np.maximum(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0)
        model_var = np.clip(hat_comp, 0.0, 10.0)
        pred_var = noise_var * (1.0 + model_var)
        mu_train = y_mean_raw + y_std * (Phi @ theta)
        err2 = (all_scores - mu_train) ** 2

        num = float(np.mean(err2))
        den = float(np.mean(pred_var))
        var_y = float(np.var(all_scores))
        rel_mse = float(num / (var_y + 1e-12))

        loo_regret, loo_topk_overlap, loo_spearman = _loo_rank_metrics(
            y_centered=y_centered,
            y_pred_centered=y_pred_centered,
            weights=weights,
            model_var=hat_comp,
            scores_raw=all_scores,
        )

        try:
            df = float(np.trace(PhiT_W_Phi @ inv_cov))
            denom = 1.0 - df / float(max(n_pts, 1))
            gcv = float(num / max(denom * denom, 1e-12))
        except Exception:
            df = 0.0
            gcv = float("inf")

        prev_num = (
            float(prev_model.get("sigma_ema_num"))
            if isinstance(prev_model, dict) and prev_model.get("sigma_ema_num") is not None
            else None
        )
        prev_den = (
            float(prev_model.get("sigma_ema_den"))
            if isinstance(prev_model, dict) and prev_model.get("sigma_ema_den") is not None
            else None
        )

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
        active_dims = np.asarray(cand, dtype=np.int64) if cand is not None else np.arange(dim, dtype=np.int64)
        sigma_scale = 1.0
        ema_num = None
        ema_den = None
        rel_mse = None
        var_y = None
        loo_regret = float("inf")
        loo_topk_overlap = 0.0
        loo_spearman = 0.0
        df = 0.0
        gcv = float("inf")

    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,
        "inv_cov": inv_cov,
        "theta": (theta.copy() if isinstance(theta, np.ndarray) else None),
        "feature_kind": "diag2",
        "gcv": float(gcv),
        "df": float(df),
        "n_feat": (int(inv_cov.shape[0]) if isinstance(inv_cov, np.ndarray) else None),
        "y_mean_raw": float(y_mean_raw),
        "y_std": float(y_std),
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
        "loo_regret": float(loo_regret),
        "loo_topk_overlap": float(loo_topk_overlap),
        "loo_spearman": float(loo_spearman),
    }


def _fit_diag2_pca_model_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, float]],
    dim: int,
    bounds: Sequence[Tuple[float, float]],
    *,
    prev_model: Optional[Dict] = None,
    candidate_dims: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Fit a diagonal quadratic in a low-rank PCA subspace (rotated low-rank quadratic)."""
    pairs_l = list(pairs)
    n_pts = int(len(pairs_l))
    if n_pts < 12:
        return None

    all_pts = np.array([p for p, _ in pairs_l], dtype=float)
    all_scores = np.array([s for _, s in pairs_l], dtype=float)

    k = max(3, n_pts // 5)
    top_k_idx = np.argsort(all_scores)[-k:]
    top_k_pts = all_pts[top_k_idx]

    gradient_dir = None
    grad = None
    inv_cov = None
    theta = None
    bias = 0.0
    noise_var = 1.0

    widths = np.maximum(np.array([abs(hi - lo) for lo, hi in bounds], dtype=float), 1e-9)
    dim = int(dim)
    lo = np.array([a for a, _ in bounds], dtype=float)
    hi = np.array([b for _, b in bounds], dtype=float)

    top_scores = all_scores[top_k_idx]
    w_ref = top_scores - float(np.min(top_scores)) + 1e-12
    if not np.all(np.isfinite(w_ref)) or float(np.sum(w_ref)) <= 0.0:
        w_ref = np.ones_like(top_scores, dtype=float)
    x_ref = np.average(top_k_pts, axis=0, weights=w_ref)
    x_ref = np.minimum(np.maximum(x_ref, lo), hi)

    X_norm_full = (all_pts - x_ref) / widths

    y_mean_global = float(np.mean(all_scores))
    y_std_global = float(np.std(all_scores)) + 1e-6
    y_centered_global = (all_scores - y_mean_global) / y_std_global

    y_mean_raw = float(y_mean_global)
    y_std = float(y_std_global)
    y_centered = y_centered_global
    bias = float(y_mean_raw)

    cand: Optional[np.ndarray]
    if candidate_dims is None:
        cand = None
    else:
        cd = np.asarray(candidate_dims, dtype=np.int64).ravel()
        cd = cd[(cd >= 0) & (cd < dim)]
        if cd.size:
            seen = set()
            uniq: List[int] = []
            for j in cd.tolist():
                jj = int(j)
                if jj in seen:
                    continue
                seen.add(jj)
                uniq.append(jj)
            cand = np.asarray(uniq, dtype=np.int64)
        else:
            cand = np.zeros(0, dtype=np.int64)

    try:
        yc = y_centered_global
        y_std_c = float(np.std(yc))

        dims_pool = np.arange(dim, dtype=np.int64) if cand is None else cand
        scores_pool = np.zeros(int(dims_pool.size), dtype=float)
        if y_std_c > 1e-12 and dims_pool.size:
            for i, j in enumerate(dims_pool.tolist()):
                xj = X_norm_full[:, int(j)]
                x_std = float(np.std(xj))
                if x_std <= 1e-12:
                    continue
                cov = float(np.mean((xj - float(np.mean(xj))) * yc))
                scores_pool[i] = abs(cov / (x_std * y_std_c + 1e-12))

        ranked_pool = dims_pool[np.argsort(-scores_pool)] if dims_pool.size else np.zeros(0, dtype=np.int64)

        base = int(max(2, int(np.floor(np.sqrt(max(n_pts, 1)))))) if n_pts > 0 else 2
        # Allow PCA even in 2D: a diagonal quadratic in a rotated 2D basis can represent
        # a full quadratic form (i.e., captures cross-terms) with the same feature count.
        p_pca = int(min(int(ranked_pool.size), max(2, 3 * base)))
        if p_pca < 2:
            return None
        active_dims = ranked_pool[:p_pca].astype(np.int64)

        X_act = X_norm_full[:, active_dims]

        dists_sq = np.sum(X_act * X_act, axis=1)
        sigma_sq = float(np.mean(dists_sq)) + 1e-6
        weights = np.exp(-dists_sq / (2.0 * sigma_sq))

        ptp = float(np.ptp(all_scores))
        rank_weights = 1.0 + 0.5 * (all_scores - float(np.min(all_scores))) / (ptp + 1e-9)
        weights = weights * rank_weights
        weights = np.clip(weights, 1e-12, np.inf)

        w_sum = float(np.sum(weights))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            return None

        # Weighted covariance around 0 (origin is x_ref in normalized coordinates).
        cov = (X_act.T * weights) @ X_act / w_sum
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(-evals)
        evals = evals[order]
        evecs = evecs[:, order]

        evals_pos = np.maximum(evals, 0.0)
        total = float(np.sum(evals_pos))
        if not np.isfinite(total) or total <= 1e-12:
            return None

        r_target = int(min(int(np.floor(np.sqrt(max(n_pts, 1)))), p_pca))
        r_target = max(2, r_target)
        cap = int(max(6, n_pts))
        r_max_feat = int(max(1, (cap - 1) // 2))
        r = int(min(p_pca, r_target, r_max_feat))
        if r < 2:
            return None

        comps = evecs[:, :r].astype(float)
        Z = X_act @ comps

        y_mean_w = float(np.average(all_scores, weights=weights))
        y_var_w = float(np.average((all_scores - y_mean_w) ** 2, weights=weights))
        y_std_w = float(np.sqrt(max(y_var_w, 1e-12)))
        if (not np.isfinite(y_std_w)) or y_std_w <= 1e-12:
            y_mean_raw = float(y_mean_global)
            y_std = float(y_std_global)
        else:
            y_mean_raw = float(y_mean_w)
            y_std = float(y_std_w) + 1e-6
        y_centered = (all_scores - y_mean_raw) / y_std

        Phi = _diag2_features(Z)
        n_feat = int(Phi.shape[1])

        PhiT_W_Phi = Phi.T @ (Phi * weights[:, None])
        PhiT_W_y = Phi.T @ (weights * y_centered)

        # See _fit_poly2_model_from_pairs for rationale.
        lambda_base = 1e-3 * (1.0 + float(n_feat) / max(float(n_pts - n_feat), 1.0))
        reg = lambda_base * np.eye(n_feat, dtype=float)
        reg[0, 0] = lambda_base * 1e-3

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
        theta = inv_cov @ PhiT_W_y

        bias = float(y_mean_raw + y_std * float(theta[0]))
        grad_z = y_std * theta[1 : 1 + r]
        grad_act = comps @ grad_z
        grad_full = np.zeros(dim, dtype=float)
        grad_full[active_dims] = grad_act
        grad = grad_full

        y_pred_centered = Phi @ theta
        residuals = y_centered - y_pred_centered
        noise_var_c = float(np.average(residuals**2, weights=weights))
        noise_var_c = float(np.clip(noise_var_c + 1e-12, 1e-6, 10.0))
        noise_var = float(noise_var_c * (y_std**2))

        grad_norm = float(np.linalg.norm(grad_full))
        if grad_norm > 1e-9:
            gradient_dir = grad_full / grad_norm

        hat_comp = np.maximum(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0)
        model_var = np.clip(hat_comp, 0.0, 10.0)
        pred_var = noise_var * (1.0 + model_var)
        mu_train = y_mean_raw + y_std * (Phi @ theta)
        err2 = (all_scores - mu_train) ** 2

        num = float(np.mean(err2))
        den = float(np.mean(pred_var))
        var_y = float(np.var(all_scores))
        rel_mse = float(num / (var_y + 1e-12))

        loo_regret, loo_topk_overlap, loo_spearman = _loo_rank_metrics(
            y_centered=y_centered,
            y_pred_centered=y_pred_centered,
            weights=weights,
            model_var=hat_comp,
            scores_raw=all_scores,
        )

        try:
            df = float(np.trace(PhiT_W_Phi @ inv_cov))
            denom = 1.0 - df / float(max(n_pts, 1))
            gcv = float(num / max(denom * denom, 1e-12))
        except Exception:
            df = 0.0
            gcv = float("inf")

        prev_num = (
            float(prev_model.get("sigma_ema_num"))
            if isinstance(prev_model, dict) and prev_model.get("sigma_ema_num") is not None
            else None
        )
        prev_den = (
            float(prev_model.get("sigma_ema_den"))
            if isinstance(prev_model, dict) and prev_model.get("sigma_ema_den") is not None
            else None
        )

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
        sigma_scale = 1.0
        ema_num = None
        ema_den = None
        rel_mse = None
        var_y = None
        loo_regret = float("inf")
        loo_topk_overlap = 0.0
        loo_spearman = 0.0
        df = 0.0
        gcv = float("inf")
        active_dims = np.asarray(cand, dtype=np.int64) if cand is not None else np.arange(dim, dtype=np.int64)
        comps = None
        r = None

    if comps is None or r is None:
        return None

    return {
        "all_pts": all_pts,
        "top_k_pts": top_k_pts,
        "gradient_dir": gradient_dir,
        "grad": grad,
        "inv_cov": inv_cov,
        "theta": (theta.copy() if isinstance(theta, np.ndarray) else None),
        "feature_kind": "diag2_pca",
        "pca_components": comps,
        "pca_r": int(r),
        "gcv": float(gcv),
        "df": float(df),
        "n_feat": (int(inv_cov.shape[0]) if isinstance(inv_cov, np.ndarray) else None),
        "y_mean_raw": float(y_mean_raw),
        "y_std": float(y_std),
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
        "loo_regret": float(loo_regret),
        "loo_topk_overlap": float(loo_topk_overlap),
        "loo_spearman": float(loo_spearman),
    }


def _fit_best_local_model_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, float]],
    dim: int,
    bounds: Sequence[Tuple[float, float]],
    *,
    prev_model: Optional[Dict] = None,
    candidate_dims: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Fit multiple local surrogate variants and pick the best deterministically (GCV).

    We still compute LOOCV ranking metrics per candidate and expose them in traces;
    empirically they were not stable enough to use as the primary selector across tasks.
    """
    prev_state: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(prev_model, dict):
        st = prev_model.get("_candidate_state")
        if isinstance(st, dict):
            for k, v in st.items():
                if isinstance(k, str) and isinstance(v, dict):
                    prev_state[k] = {
                        "sigma_ema_num": (float(v["sigma_ema_num"]) if v.get("sigma_ema_num") is not None else None),
                        "sigma_ema_den": (float(v["sigma_ema_den"]) if v.get("sigma_ema_den") is not None else None),
                    }
        fk = prev_model.get("feature_kind")
        if isinstance(fk, str) and fk and fk not in prev_state:
            prev_state[fk] = {
                "sigma_ema_num": (
                    float(prev_model.get("sigma_ema_num")) if prev_model.get("sigma_ema_num") is not None else None
                ),
                "sigma_ema_den": (
                    float(prev_model.get("sigma_ema_den")) if prev_model.get("sigma_ema_den") is not None else None
                ),
            }

    def prev_for(kind: str) -> Optional[Dict]:
        st = prev_state.get(kind)
        if not st:
            return None
        return {
            "sigma_ema_num": st.get("sigma_ema_num"),
            "sigma_ema_den": st.get("sigma_ema_den"),
        }

    candidates: Dict[str, Dict] = {}
    poly = _fit_poly2_model_from_pairs(
        pairs,
        dim=int(dim),
        bounds=bounds,
        prev_model=prev_for("poly2"),
        candidate_dims=candidate_dims,
    )
    if isinstance(poly, dict) and poly.get("inv_cov") is not None:
        candidates["poly2"] = poly

    diag = _fit_diag2_model_from_pairs(
        pairs,
        dim=int(dim),
        bounds=bounds,
        prev_model=prev_for("diag2"),
        candidate_dims=candidate_dims,
    )
    if isinstance(diag, dict) and diag.get("inv_cov") is not None:
        candidates["diag2"] = diag

    pca = _fit_diag2_pca_model_from_pairs(
        pairs,
        dim=int(dim),
        bounds=bounds,
        prev_model=prev_for("diag2_pca"),
        candidate_dims=candidate_dims,
    )
    if isinstance(pca, dict) and pca.get("inv_cov") is not None:
        candidates["diag2_pca"] = pca

    if not candidates:
        return None

    best_kind = None
    best_gcv = float("inf")
    for kind, model in candidates.items():
        try:
            gcv = float(model.get("gcv", float("inf")))
        except Exception:
            gcv = float("inf")
        if not np.isfinite(gcv):
            continue
        if gcv < best_gcv:
            best_gcv = gcv
            best_kind = kind

    if best_kind is None:
        best_kind = "poly2" if "poly2" in candidates else sorted(candidates.keys())[0]

    chosen = candidates[best_kind]

    # Persist per-candidate EMA state across refits without storing whole candidate models.
    new_state: Dict[str, Dict[str, Optional[float]]] = {}
    for kind, m in candidates.items():
        new_state[kind] = {
            "sigma_ema_num": (float(m["sigma_ema_num"]) if m.get("sigma_ema_num") is not None else None),
            "sigma_ema_den": (float(m["sigma_ema_den"]) if m.get("sigma_ema_den") is not None else None),
        }

    chosen["_candidate_state"] = new_state
    chosen["selected_kind"] = str(best_kind)
    chosen["candidates_gcv"] = {k: float(v.get("gcv", float("inf"))) for k, v in candidates.items()}
    chosen["candidates_rel_mse"] = {k: v.get("rel_mse") for k, v in candidates.items()}
    chosen["candidates_n_feat"] = {k: v.get("n_feat") for k, v in candidates.items()}
    chosen["candidates_loo_regret"] = {k: v.get("loo_regret") for k, v in candidates.items()}
    chosen["candidates_loo_topk_overlap"] = {k: v.get("loo_topk_overlap") for k, v in candidates.items()}
    chosen["candidates_loo_spearman"] = {k: v.get("loo_spearman") for k, v in candidates.items()}
    return chosen


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

    prev_global = getattr(cube, "lgs_model", None)

    categorical_dims = getattr(cube, "categorical_dims", None)
    cat_info: List[Tuple[int, int]] = []
    if categorical_dims is not None:
        try:
            cat_info = [(int(i), int(n)) for i, n in list(categorical_dims)]
        except Exception:
            cat_info = []

    cat_dim_set = {int(i) for i, _ in cat_info}
    cont_dims = np.asarray([i for i in range(int(dim)) if i not in cat_dim_set], dtype=np.int64)

    global_model = _fit_best_local_model_from_pairs(
        pairs,
        dim=int(dim),
        bounds=cube.bounds,
        prev_model=(prev_global if isinstance(prev_global, dict) else None),
        candidate_dims=cont_dims,
    )
    if global_model is None:
        return None

    if not cat_info:
        return global_model

    all_pts = np.asarray([p for p, _ in pairs], dtype=float)
    all_scores = np.asarray([s for _, s in pairs], dtype=float)
    mu_train, _ = predict_bayesian(global_model, all_pts)
    resid = all_scores - mu_train

    keys = _cat_keys(all_pts, cat_info)
    alpha = 2.0
    cat_bias: Dict[Tuple[int, ...], float] = {}
    sums: Dict[Tuple[int, ...], float] = {}
    counts: Dict[Tuple[int, ...], int] = {}
    for k, r in zip(keys, resid):
        if not np.isfinite(r):
            continue
        sums[k] = float(sums.get(k, 0.0) + float(r))
        counts[k] = int(counts.get(k, 0) + 1)
    for k, c in counts.items():
        cat_bias[k] = float(sums.get(k, 0.0) / (float(c) + float(alpha)))

    by_cat_key: Dict[Tuple[int, ...], Dict] = {}
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(int(i))

    prev_by_key = prev_global.get("by_cat_key", {}) if isinstance(prev_global, dict) else {}
    min_pts_key = 6
    for k, idxs in groups.items():
        if len(idxs) < min_pts_key:
            continue
        sub_pairs = [pairs[int(i)] for i in idxs]
        prev_sub = prev_by_key.get(k) if isinstance(prev_by_key, dict) else None
        sub = _fit_best_local_model_from_pairs(
            sub_pairs,
            dim=int(dim),
            bounds=cube.bounds,
            prev_model=(prev_sub if isinstance(prev_sub, dict) else None),
            candidate_dims=cont_dims,
        )
        if sub is None or sub.get("inv_cov") is None:
            continue
        by_cat_key[k] = sub

    global_model["kind"] = "cat_mixture"
    global_model["categorical_dims"] = list(cat_info)
    global_model["cat_bias"] = cat_bias
    global_model["by_cat_key"] = by_cat_key
    return global_model


CandidatesLike = Union[Sequence[np.ndarray], np.ndarray]

def _predict_single_model(model: Optional[Dict], candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    active_dims = model.get("active_dims")
    sigma_scale = float(model.get("sigma_scale", 1.0))

    C_norm = (C - center) / widths

    theta = model.get("theta")
    feature_kind = model.get("feature_kind")
    y_mean_raw = model.get("y_mean_raw")
    y_std = model.get("y_std")

    if (
        isinstance(theta, np.ndarray)
        and theta.size > 0
        and isinstance(feature_kind, str)
        and feature_kind in {"poly2", "diag2", "diag2_pca"}
        and y_mean_raw is not None
        and y_std is not None
    ):
        idx = np.asarray(active_dims, dtype=np.int64) if active_dims is not None else np.arange(C_norm.shape[1])
        X_act = C_norm[:, idx] if idx.size else np.zeros((C_norm.shape[0], 0), dtype=float)
        if feature_kind == "poly2":
            Phi = _poly2_features(X_act)
        elif feature_kind == "diag2":
            Phi = _diag2_features(X_act)
        else:  # "diag2_pca"
            comps = model.get("pca_components")
            if isinstance(comps, np.ndarray) and comps.ndim == 2 and X_act.shape[1] == comps.shape[0]:
                Z = X_act @ comps
                Phi = _diag2_features(Z)
            else:
                # Fallback to diagonal quadratic in the original active subspace.
                Phi = _diag2_features(X_act)
        mu = float(y_mean_raw) + float(y_std) * (Phi @ theta)
        model_var = np.clip(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0, 10.0)
    else:
        # Fallback: linear model (backward-compatible).
        y_mean = model["y_mean"]
        mu = y_mean + C_norm @ grad
        if active_dims is None:
            model_var = np.clip(np.sum((C_norm @ inv_cov) * C_norm, axis=1), 0.0, 10.0)
        else:
            idx = np.asarray(active_dims, dtype=np.int64)
            X_act = C_norm[:, idx] if idx.size else np.zeros((C_norm.shape[0], 0), dtype=float)
            Phi = np.concatenate([np.ones((C_norm.shape[0], 1), dtype=float), X_act], axis=1)
            model_var = np.clip(np.sum((Phi @ inv_cov) * Phi, axis=1), 0.0, 10.0)

    total_var = noise_var * (1.0 + model_var)
    sigma = np.sqrt(total_var) * sigma_scale
    return mu, sigma


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

    cat_info = model.get("categorical_dims")
    cat_bias = model.get("cat_bias")
    by_cat_key = model.get("by_cat_key")

    use_cat = bool(cat_info) and (isinstance(cat_bias, dict) or isinstance(by_cat_key, dict))
    if not use_cat:
        return _predict_single_model(model, C)

    mu, sigma = _predict_single_model(model, C)
    keys = _cat_keys(C, list(cat_info))

    if isinstance(cat_bias, dict) and cat_bias:
        b = np.zeros(C.shape[0], dtype=float)
        for i, k in enumerate(keys):
            v = cat_bias.get(k)
            if v is None:
                continue
            try:
                b[i] = float(v)
            except Exception:
                continue
        mu = mu + b

    if isinstance(by_cat_key, dict) and by_cat_key:
        groups: Dict[Tuple[int, ...], List[int]] = {}
        for i, k in enumerate(keys):
            if k in by_cat_key:
                groups.setdefault(k, []).append(int(i))
        for k, idxs in groups.items():
            sub = by_cat_key.get(k)
            if not isinstance(sub, dict) or sub.get("inv_cov") is None:
                continue
            mu_k, sigma_k = _predict_single_model(sub, C[np.asarray(idxs, dtype=np.int64)])
            mu[np.asarray(idxs, dtype=np.int64)] = mu_k
            sigma[np.asarray(idxs, dtype=np.int64)] = sigma_k

    return mu, sigma
