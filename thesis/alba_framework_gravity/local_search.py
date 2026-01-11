"""ALBA Framework - Local Search

This module defines how ALBA samples around the incumbent best solution
in the local-search phase.

Implementations:
- GaussianLocalSearchSampler: Default Gaussian perturbation (ALBA_V1)
- LLRGradientLocalSearchSampler: Uses Local Linear Regression gradient estimation
  to guide local search direction (experimental, improves convergence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ) -> np.ndarray:
        ...
    
    def update_history(self, x: np.ndarray, y: float) -> None:
        """Optional: update internal observation history."""
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
    ) -> np.ndarray:
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        radius = self.radius_start * (1 - float(progress)) + self.radius_end
        noise = rng.normal(0, radius, len(bounds)) * global_widths
        x = best_x + noise

        return np.array([np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(bounds))], dtype=float)

    def update_history(self, x: np.ndarray, y: float) -> None:
        """No-op for Gaussian sampler (doesn't use history)."""
        pass


@dataclass
class LLRGradientLocalSearchSampler:
    """
    Local search sampler using Local Linear Regression (LLR) gradient estimation.
    
    Fits a weighted local plane to nearby observations to estimate the gradient
    direction, then combines with random perturbation for robustness.
    
    Formula:
        1. Find K nearest neighbors to best_x
        2. Fit weighted linear model: min Σ w_i·(f_i - a·x_i - b)²
        3. Gradient direction = -a / |a| (descent direction for minimization)
        4. Final direction = (1-gw)·random + gw·gradient
    
    Parameters
    ----------
    gradient_weight : float
        Weight for gradient direction vs random [0, 1]. Default 0.7.
    n_neighbors_factor : float
        Neighbors = factor × dim. Default 2.0.
    min_neighbors : int
        Minimum neighbors for LLR fit. Default 5.
    radius_start : float
        Initial step size (fraction of width). Default 0.12.
    radius_end : float
        Final step size. Default 0.02.
    """
    
    gradient_weight: float = 0.7
    n_neighbors_factor: float = 2.0
    min_neighbors: int = 5
    radius_start: float = 0.12
    radius_end: float = 0.02
    
    # Mutable fields for observation history
    X_history: List[np.ndarray] = field(default_factory=list)
    y_history: List[float] = field(default_factory=list)
    
    def update_history(self, x: np.ndarray, y: float) -> None:
        """Record a new observation for gradient estimation."""
        self.X_history.append(x.copy())
        self.y_history.append(y)
    
    def _estimate_llr_gradient(
        self,
        x: np.ndarray,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        n_neighbors: int,
    ) -> Optional[np.ndarray]:
        """
        Estimate local gradient using weighted linear regression.
        
        Returns None if not enough neighbors or fit fails.
        """
        if len(X_obs) < self.min_neighbors:
            return None
        
        # Compute distances
        dists = np.linalg.norm(X_obs - x, axis=1)
        
        # Select K nearest neighbors
        k = min(n_neighbors, len(X_obs))
        if k < self.min_neighbors:
            return None
        
        neighbor_idx = np.argsort(dists)[:k]
        X_neighbors = X_obs[neighbor_idx]
        y_neighbors = y_obs[neighbor_idx]
        dist_neighbors = dists[neighbor_idx]
        
        # Compute Gaussian weights (adaptive sigma = median distance)
        sigma = np.median(dist_neighbors) + 1e-8
        weights = np.exp(-dist_neighbors**2 / (2 * sigma**2))
        
        # Weighted least squares: min Σ w_i·(y_i - a·x_i - b)²
        # Using matrix form: (X^T W X)^{-1} X^T W y
        X_aug = np.hstack([X_neighbors, np.ones((len(X_neighbors), 1))])  # Add bias column
        W = np.diag(weights)
        
        try:
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ W @ y_neighbors
            
            # Add small regularization for numerical stability
            XtWX += 1e-6 * np.eye(XtWX.shape[0])
            
            coeffs = np.linalg.solve(XtWX, XtWy)
            gradient = coeffs[:-1]  # Exclude bias term
            
            norm = np.linalg.norm(gradient)
            if norm < 1e-10:
                return None
            
            return gradient / norm  # Normalized gradient (points uphill)
            
        except np.linalg.LinAlgError:
            return None
    
    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
        progress: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample next point using LLR gradient + random perturbation.
        """
        dim = len(bounds)
        
        # Fallback: pure random
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
        
        # Calculate adaptive step radius
        radius = self.radius_start * (1 - progress) + self.radius_end
        
        # Estimate LLR gradient if we have enough history
        llr_direction: Optional[np.ndarray] = None
        if len(self.X_history) >= self.min_neighbors:
            X_obs = np.array(self.X_history)
            y_obs = np.array(self.y_history)
            n_neighbors = max(self.min_neighbors, int(self.n_neighbors_factor * dim))
            
            llr_direction = self._estimate_llr_gradient(best_x, X_obs, y_obs, n_neighbors)
        
        # Generate random direction (normalized)
        random_dir = rng.normal(0, 1, dim)
        random_dir /= (np.linalg.norm(random_dir) + 1e-10)
        
        # Combine gradient with random
        if llr_direction is not None:
            # For minimization: move against gradient (descent)
            # Note: y_history is "higher is better" in ALBA, so gradient points to improvement
            descent_dir = llr_direction  # Already points toward improvement
            
            gw = self.gradient_weight
            direction = (1 - gw) * random_dir + gw * descent_dir
            direction /= (np.linalg.norm(direction) + 1e-10)
        else:
            direction = random_dir
        
        # Apply perturbation
        step = radius * global_widths * direction
        x_new = best_x + step
        
        # Clip to bounds
        return np.array([np.clip(x_new[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)
