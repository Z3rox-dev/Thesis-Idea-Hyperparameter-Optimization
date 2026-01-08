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
    from .grid import LeafGridState


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
    grid_state: Optional["LeafGridState"] = field(default=None, repr=False)
    tr_scale: float = 1.0  # in (0,1], local trust-region scale inside the cube

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------

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

        # Primary: gradient direction (if available and reliable)
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            grad_dir = np.abs(self.lgs_model["gradient_dir"])
            # Only trust gradient if it's reasonably strong in one direction
            if grad_dir.max() > 0.3:
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
                return int(np.argmax(score))

        # Fallback: widest dimension
        return int(np.argmax(widths))

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
            ch.fit_lgs_model(gamma, dim, rng)

        return [child_lo, child_hi]
