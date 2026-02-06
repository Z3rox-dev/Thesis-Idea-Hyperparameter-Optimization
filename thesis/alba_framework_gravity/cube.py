"""
ALBA Framework - Cube Module

This module implements the Cube class, representing a hyperrectangle region
of the search space with a local surrogate model (Local Gradient Surrogate).

The Cube class is the fundamental building block of ALBA's space partitioning
strategy, enabling adaptive refinement of promising regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .lgs import fit_lgs_model as _fit_lgs_model
from .lgs import predict_bayesian as _predict_bayesian

if TYPE_CHECKING:
    pass


@dataclass(eq=False)
class Cube:
    """
    A hyperrectangle region of the search space with a local surrogate model.

    The cube maintains statistics about evaluated points, fits a Local Gradient
    Surrogate (LGS) model for prediction, and supports adaptive splitting based
    on performance characteristics.

    Attributes
    ----------
    bounds : List[Tuple[float, float]]
        Lower and upper bounds for each dimension.
    parent : Optional[Cube]
        Parent cube (None for root).
    n_trials : int
        Number of points evaluated in this cube.
    n_good : int
        Number of points above the gamma threshold.
    best_score : float
        Best internal score observed in this cube.
    best_x : Optional[np.ndarray]
        Configuration corresponding to best_score.
    depth : int
        Depth in the partition tree (0 for root).
    cat_stats : dict
        Per-dimension categorical statistics: {dim_idx: {val_idx: (n_good, n_total)}}.
    """

    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    lgs_model: Optional[Dict] = field(default=None, init=False)
    depth: int = 0
    cat_stats: Dict[int, Dict[int, Tuple[int, int]]] = field(default_factory=dict)
    categorical_dims: Tuple[int, ...] = field(default_factory=tuple, repr=False)
    warp_dim_sensitivity: Optional[np.ndarray] = field(default=None, repr=False)
    warp_history: List[Tuple[np.ndarray, float]] = field(default_factory=list, repr=False)
    warp_updates: int = 0

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.warp_dim_sensitivity is None:
            self.warp_dim_sensitivity = np.ones(len(self.bounds), dtype=float)

    def widths(self) -> np.ndarray:
        """Return the width of each dimension."""
        return np.array([abs(hi - lo) for lo, hi in self.bounds], dtype=float)

    def center(self) -> np.ndarray:
        """Return the center point of the cube."""
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds], dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is inside the cube (with small tolerance)."""
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo - 1e-9 or x[i] > hi + 1e-9:
                return False
        return True

    def volume(self) -> float:
        """Return the volume of the cube."""
        return float(np.prod(self.widths()))

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def good_ratio(self) -> float:
        """
        Beta prior estimation of the proportion of good points.

        Uses (good + 1) / (trials + 2) to avoid extreme ratios when
        there are few observations.

        Returns
        -------
        float
            Estimated proportion of good points in [0, 1].
        """
        return (self.n_good + 1) / (self.n_trials + 2)

    @property
    def tested_pairs(self) -> List[Tuple[np.ndarray, float]]:
        """Return list of (point, score) pairs tested in this cube."""
        return self._tested_pairs

    def add_observation(self, x: np.ndarray, score: float, gamma: float) -> None:
        """
        Add an observation to this cube.

        Parameters
        ----------
        x : np.ndarray
            The evaluated point.
        score : float
            The internal score (higher is better).
        gamma : float
            Current threshold for "good" points.
        """
        self._tested_pairs.append((x.copy(), score))
        self.n_trials += 1

        if score >= gamma:
            self.n_good += 1

        if score > self.best_score:
            self.best_score = score
            self.best_x = x.copy()

    # -------------------------------------------------------------------------
    # Local warp ("elastic" per-leaf metric)
    # -------------------------------------------------------------------------

    def get_warp_multipliers(self) -> np.ndarray:
        """
        Return per-dimension multipliers used to warp distances and steps.

        Multipliers > 1 stretch that dimension (treat as farther / step smaller).
        Multipliers < 1 compress that dimension (treat as closer / step larger).
        """
        dim = len(self.bounds)
        s = self.warp_dim_sensitivity
        if s is None or getattr(s, "shape", None) != (dim,):
            return np.ones(dim, dtype=float)

        cat_dims = {int(i) for i in self.categorical_dims}
        cont_dims = [i for i in range(dim) if i not in cat_dims]
        if not cont_dims:
            return np.ones(dim, dtype=float)

        s_cont = np.asarray([float(s[i]) for i in cont_dims], dtype=float)
        s_cont = np.where(np.isfinite(s_cont), s_cont, 1.0)
        mean_s = float(np.mean(s_cont))
        if not np.isfinite(mean_s) or mean_s <= 1e-12:
            return np.ones(dim, dtype=float)

        rel = np.asarray(s, dtype=float) / mean_s
        rel = np.where(np.isfinite(rel), rel, 1.0)
        w = np.sqrt(np.clip(rel, 0.25, 4.0))
        for i in cat_dims:
            if 0 <= i < dim:
                w[i] = 1.0
        return w.astype(float)

    def update_warp(self, x: np.ndarray, score: float) -> None:
        """
        Update per-dimension sensitivity from (x, score) using nearby pairs.

        Uses only existing observations (no extra evaluations).
        """
        x = np.asarray(x, dtype=float)
        y = float(score)
        dim = len(self.bounds)
        if x.shape[0] != dim:
            return
        if self.warp_dim_sensitivity is None or getattr(self.warp_dim_sensitivity, "shape", None) != (dim,):
            self.warp_dim_sensitivity = np.ones(dim, dtype=float)

        widths = np.maximum(self.widths(), 1e-9)
        cat_dims = {int(i) for i in self.categorical_dims}
        cont_dims = [i for i in range(dim) if i not in cat_dims]
        if not cont_dims:
            self.warp_history.append((x.copy(), y))
            self.warp_updates += 1
            if len(self.warp_history) > 80:
                self.warp_history = self.warp_history[-80:]
            return

        alpha = 0.20
        min_diff = 0.02
        align_thresh = 0.40
        window = 30

        grads: List[List[float]] = [[] for _ in range(dim)]
        try:
            w = self.get_warp_multipliers()
        except Exception:
            w = None
        if w is None or getattr(w, "shape", None) != (dim,):
            w = np.ones(dim, dtype=float)

        for px, py in self.warp_history[-window:]:
            dx = (x - px) / widths
            abs_dx = np.abs(dx)
            abs_dx_w = abs_dx * np.asarray(w, dtype=float)
            sum_abs_w = float(np.sum(abs_dx_w[cont_dims]))
            if sum_abs_w < 1e-12:
                continue
            dy = float(abs(y - float(py)))
            if not np.isfinite(dy):
                continue

            # Attribute this pair to the single most-changed dim in warped coordinates.
            i_star = int(cont_dims[int(np.argmax(abs_dx_w[cont_dims]))])
            di = float(abs_dx[i_star])
            if di < min_diff:
                continue
            alignment = float(abs_dx_w[i_star]) / (sum_abs_w + 1e-12)
            if alignment < align_thresh:
                continue
            g = dy / (di + 1e-12)
            if np.isfinite(g):
                grads[i_star].append(float(g))

        for i in cont_dims:
            if grads[i]:
                med = float(np.median(np.asarray(grads[i], dtype=float)))
                if np.isfinite(med) and med > 0.0:
                    self.warp_dim_sensitivity[i] = float(
                        (1.0 - alpha) * float(self.warp_dim_sensitivity[i]) + alpha * med
                    )

        self.warp_history.append((x.copy(), y))
        self.warp_updates += 1
        if len(self.warp_history) > 80:
            self.warp_history = self.warp_history[-80:]

    def rebuild_warp(self) -> None:
        """Rebuild warp statistics from this cube's tested_pairs."""
        dim = len(self.bounds)
        self.warp_dim_sensitivity = np.ones(dim, dtype=float)
        self.warp_history = []
        self.warp_updates = 0
        for pt, sc in self._tested_pairs:
            self.update_warp(np.asarray(pt, dtype=float), float(sc))

    # -------------------------------------------------------------------------
    # Local Gradient Surrogate (LGS) Model
    # -------------------------------------------------------------------------

    def fit_lgs_model(
        self,
        gamma: float,
        dim: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.lgs_model = _fit_lgs_model(self, gamma, dim, rng)

    def predict_bayesian(
        self, candidates: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _predict_bayesian(self.lgs_model, candidates)

    # -------------------------------------------------------------------------
    # Split logic
    # -------------------------------------------------------------------------

    def get_split_axis(self) -> int:
        """
        Choose the axis along which to split this cube.

        Selection priority:
        1. Gradient direction (if available and reliable)
        2. Variance of good points (split where good configs spread most)
        3. Widest dimension (fallback)

        Returns
        -------
        int
            Index of the dimension to split along.
        """
        widths = self.widths()
        cat_dims = set(int(i) for i in self.categorical_dims)
        cont_dims = [i for i in range(len(widths)) if i not in cat_dims]

        # Primary: gradient direction (if available and reliable)
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            grad_dir = np.abs(self.lgs_model["gradient_dir"])
            # Only trust gradient if it's reasonably strong in one direction
            if grad_dir.max() > 0.3:
                if cont_dims:
                    grad_dir = grad_dir.copy()
                    for i in cat_dims:
                        if 0 <= i < grad_dir.shape[0]:
                            grad_dir[i] = 0.0
                    if grad_dir.max() > 1e-12:
                        return int(np.argmax(grad_dir))
                else:
                    return int(np.argmax(grad_dir))

        # Secondary: variance of good points (split where good configs spread most)
        good_pts = np.array(
            [p for p, s in self._tested_pairs if s >= self.best_score * 0.95]
        )
        if len(good_pts) >= 3:
            # Normalize by widths to compare fairly
            var_per_dim = np.var(good_pts / (widths + 1e-9), axis=0)
            # Prefer dimensions with high variance AND reasonable width
            score = var_per_dim * (widths / (widths.max() + 1e-9))
            if score.max() > 0.01:
                if cont_dims:
                    score = score.copy()
                    for i in cat_dims:
                        if 0 <= i < score.shape[0]:
                            score[i] = 0.0
                    if score.max() > 1e-12:
                        return int(np.argmax(score))
                else:
                    return int(np.argmax(score))

        # Fallback: widest dimension
        if cont_dims:
            widths = widths.copy()
            for i in cat_dims:
                if 0 <= i < widths.shape[0]:
                    widths[i] = 0.0
            if widths.max() > 1e-12:
                return int(np.argmax(widths))
        return int(np.argmax(self.widths()))

    def split(
        self,
        gamma: float,
        dim: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List["Cube"]:
        """
        Split this cube into two children along the chosen axis.

        The split point is determined by a weighted median of good points,
        giving more weight to higher-scoring configurations.

        Parameters
        ----------
        gamma : float
            Current threshold for "good" points.
        dim : int
            Dimensionality of the search space.
        rng : Optional[np.random.Generator]
            Random generator for model fitting.

        Returns
        -------
        List[Cube]
            Two child cubes [lower, upper].
        """
        axis = self.get_split_axis()
        lo, hi = self.bounds[axis]
        good_pairs = [(p[axis], s) for p, s in self._tested_pairs if s >= gamma]

        if len(good_pairs) >= 3:
            # Weighted median: weight by how much above gamma
            positions = np.array([pos for pos, _ in good_pairs])
            scores = np.array([s for _, s in good_pairs])
            weights = scores - gamma + 1e-6  # Higher score = more weight
            weights = weights / weights.sum()

            # Weighted median via sorting
            sorted_idx = np.argsort(positions)
            cumsum = np.cumsum(weights[sorted_idx])
            median_idx = np.searchsorted(cumsum, 0.5)
            median_idx = min(median_idx, len(positions) - 1)
            cut = float(positions[sorted_idx[median_idx]])

            margin = 0.12 * (hi - lo)  # Slightly smaller margin
            cut = np.clip(cut, lo + margin, hi - margin)
        elif len(good_pairs) >= 1:
            # Few good points: use their mean
            cut = float(np.mean([pos for pos, _ in good_pairs]))
            margin = 0.15 * (hi - lo)
            cut = np.clip(cut, lo + margin, hi - margin)
        else:
            cut = (lo + hi) / 2

        bounds_lo = list(self.bounds)
        bounds_hi = list(self.bounds)
        bounds_lo[axis] = (lo, cut)
        bounds_hi[axis] = (cut, hi)

        child_lo = Cube(bounds=bounds_lo, parent=self)
        child_hi = Cube(bounds=bounds_hi, parent=self)
        child_lo.depth = self.depth + 1
        child_hi.depth = self.depth + 1
        child_lo.categorical_dims = tuple(self.categorical_dims)
        child_hi.categorical_dims = tuple(self.categorical_dims)

        for pt, sc in self._tested_pairs:
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1
            # Update child's best score
            if sc > child.best_score:
                child.best_score = sc
                child.best_x = pt.copy()

        for ch in (child_lo, child_hi):
            ch.rebuild_warp()
            ch.fit_lgs_model(gamma, dim, rng)

        return [child_lo, child_hi]
