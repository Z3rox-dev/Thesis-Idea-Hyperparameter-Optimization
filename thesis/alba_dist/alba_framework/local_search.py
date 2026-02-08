"""ALBA Framework - Local Search

This module defines how ALBA samples around the incumbent best solution
in the local-search phase.

Default implementation matches ALBA_V1: Gaussian perturbation with a
radius that shrinks as progress increases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List, Optional, Tuple

import numpy as np


class LocalSearchSampler(Protocol):
    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
    progress: float,
        rng: np.random.Generator,
        X_history: Optional[List[np.ndarray]] = None,
        y_history: Optional[List[float]] = None,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GaussianLocalSearchSampler:
    """Default local-search sampler (matches ALBA_V1)."""

    radius_start: float = 0.15
    radius_end: float = 0.03

    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
        progress: float,
        rng: np.random.Generator,
        X_history: Optional[List[np.ndarray]] = None,
        y_history: Optional[List[float]] = None,
    ) -> np.ndarray:
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        # FIX Finding 29: Sanitize inputs
        # Handle NaN in progress
        if not np.isfinite(progress):
            progress = 0.5  # Default mid-progress
        progress = float(np.clip(progress, 0.0, 1.0))
        
        # Handle NaN in best_x - replace with bounds center
        best_x = np.array(best_x, dtype=float)
        for i in range(len(bounds)):
            if not np.isfinite(best_x[i]):
                best_x[i] = (bounds[i][0] + bounds[i][1]) / 2
        
        radius = self.radius_start * (1 - progress) + self.radius_end
        radius = max(radius, 1e-6)  # Ensure non-negative
        noise = rng.normal(0, radius, len(bounds)) * global_widths
        x = best_x + noise

        return np.array([np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(bounds))], dtype=float)


@dataclass(frozen=True)
class CovarianceLocalSearchSampler:
    """
    Local search sampler that adapts to the geometry of the problem
    using the covariance of the best points found so far.
    
    Acts as a simplified CMA-ES: learns the 'shape' of the valley.
    
    Key findings from extensive research:
    1. Use adaptive top_k_fraction: more points needed in high dimensions
    2. Use dimension-proportional regularization: condition number explodes in high-D
    3. Scale multiplier ~5.0 works best across dimensions
    4. Center on best_x, not on weighted mean mu_w
    """
    
    radius_start: float = 0.15
    radius_end: float = 0.01
    base_top_k_fraction: float = 0.15  # Base fraction, scales up with dimension
    min_points_fit: int = 10
    scale_multiplier: float = 5.0  # Research shows 5.0 is optimal across dimensions
    base_regularization: float = 1e-2  # Higher than before to handle high-D
    
    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
        progress: float,
        rng: np.random.Generator,
        X_history: Optional[List[np.ndarray]] = None,
        y_history: Optional[List[float]] = None,
    ) -> np.ndarray:
        dim = len(bounds)
        
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        # Sanitize inputs
        if not np.isfinite(progress):
            progress = 0.5
        progress = float(np.clip(progress, 0.0, 1.0))
        
        best_x = np.array(best_x, dtype=float)
        for i in range(dim):
            if not np.isfinite(best_x[i]):
                best_x[i] = (bounds[i][0] + bounds[i][1]) / 2
        
        scale = self.radius_start * (1 - progress) + self.radius_end
        scale = max(scale, 1e-6)

        can_fit = False
        x_candidate = None

        if X_history is not None and y_history is not None:
            n = len(X_history)
            min_needed = max(self.min_points_fit, dim + 2)
            
            if n >= min_needed:
                # IMPROVEMENT 1: Adaptive top_k_fraction
                # In high dimensions we need more points to estimate covariance reliably
                # Formula: fraction = min(0.5, base + 0.02 * dim)
                # - dim=3:  0.15 + 0.06 = 0.21
                # - dim=10: 0.15 + 0.20 = 0.35
                # - dim=20: 0.15 + 0.40 = 0.50 (capped)
                adaptive_fraction = min(0.5, self.base_top_k_fraction + 0.02 * dim)
                k = max(min_needed, int(n * adaptive_fraction))
                
                # Select top-k by fitness (y_history is assumed higher=better)
                indices = np.argsort(y_history)
                top_indices = indices[-k:][::-1]  # Best first
                
                top_X = np.array([X_history[i] for i in top_indices])
                
                # Weighted covariance (CMA-ES style)
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X, axis=0, weights=weights)
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                
                # IMPROVEMENT 2: Dimension-proportional regularization
                # Condition number grows ~O(dim^2), so regularization must scale
                # eps = base * (1 + 0.1 * dim)
                # - dim=3:  0.01 * 1.3 = 0.013
                # - dim=10: 0.01 * 2.0 = 0.02
                # - dim=20: 0.01 * 3.0 = 0.03
                eps = self.base_regularization * (1 + 0.1 * dim)
                C += eps * np.eye(dim)
                
                try:
                    # Generate sample from learned covariance
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    
                    # Center on best_x (not mu_w!) to stay near the true best
                    cov_scale = scale * self.scale_multiplier
                    x_candidate = best_x + (z * cov_scale)
                    can_fit = True
                except Exception:
                    can_fit = False
        
        if not can_fit:
            # Fallback to Gaussian
            noise = rng.normal(0, scale, dim) * global_widths
            x_candidate = best_x + noise

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)
