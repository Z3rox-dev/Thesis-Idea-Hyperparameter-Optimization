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

        progress = float(np.clip(progress, 0.0, 1.0))
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
    """
    
    radius_start: float = 0.15
    radius_end: float = 0.01
    top_k_fraction: float = 0.15  # Increased slightly to allow more points for weighting
    min_points_fit: int = 10
    
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

        progress = float(np.clip(progress, 0.0, 1.0))
        scale = self.radius_start * (1 - progress) + self.radius_end
        scale = max(scale, 1e-6)

        can_fit = False
        x_candidate = None

        if X_history is not None and y_history is not None:
            n = len(X_history)
            min_needed = max(self.min_points_fit, dim + 2)
            
            if n >= min_needed:
                # Select top K points
                k = max(min_needed, int(n * self.top_k_fraction))
                indices = np.argsort(y_history)[-k:] # y is fitness (negative cost?), check optimizer. ALBA maximizes?
                # Wait, ALBA minimizes cost usually, but internally might use fitness.
                # In optimizer.py: "loss" is passed to tell(). 
                # Let's assume y_history is what is stored in ALBA.y_all.
                # If ALBA.y_all stores NEGATIVE cost (fitness), then argsort is correct (highest is best).
                # If ALBA stores COST, then lowest is best.
                # Checking optimizer.py: self.y_all.append(-transformed_y) usually?
                # Actually, in ALBA usually maximization=False means we minimize.
                # But tell() usually flips sign if maximize=False.
                # Let's be safe: The BEST points are those with highest y values in y_all (since ALBA maximizes internal fitness).
                # Assuming y_all is consistent with "higher is better".
                
                # Sort indices: last ones are best
                indices = np.argsort(y_history)
                # Take last k (highest values)
                top_indices = indices[-k:]
                # Reverse so best is first
                top_indices = top_indices[::-1]
                
                top_X = np.array([X_history[i] for i in top_indices])
                
                # --- WEIGHTED COVARIANCE (CMA-Style) ---
                # Weights: log(k+1/2) - log(i+1)
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights) # Normalize to sum 1
                
                # Weighted Mean
                mu_w = np.average(top_X, axis=0, weights=weights)

                # Weighted Covariance
                # C = sum(w_i * (x_i - mu_w)(x_i - mu_w)^T)
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                
                # Regularization to prevent singularity
                C += 1e-6 * np.eye(dim)

                try:
                    # Generate sample
                    # We center the new sample around mu_w (weighted center) or best_x?
                    # CMA centers on mu_w (updated mean).
                    # ALBA's "best_x" is just the single 1-best.
                    # mu_w is a more robust estimator of the "center of excellence".
                    # Let's use mu_w as the mean.
                    
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    
                    # Scale handling:
                    # If C comes from historical data, it already has the "scale" of the cloud.
                    # We might want to contract it slightly?
                    # "scale" parameter is 0.15 -> 0.01.
                    # If we multiply z by scale, we shrink the cloud significantly.
                    # If the cloud is large (exploration), shrinking is good (exploitation).
                    
                    # NOTE: If we use covariance from history, that covariance represents
                    # the region we HAVE visited.
                    # To drill down, we usually want to search WITHIN that region or slightly smaller.
                    # If scale=1.0, we replicate the distribution.
                    # If scale < 1.0, we refine.
                    
                    # For safety, let's mix mu_w and best_x
                    # But mu_w is likely close to best_x.
                    
                    x = mu_w + (z * 1.0) # Use the natural scale of the covariance
                    
                    # But we must apply the external "progress" decay?
                    # If the cloud is static, C is static.
                    # We want to shrink over time?
                    # ALBA handles global contraction via partitioning.
                    # Here we want to match the "current" shape.
                    
                    # Let's apply a mild shrinkage to encourage convergence
                    x = mu_w + (z * 0.9) 
                    
                    can_fit = True
                    x_candidate = x
                except Exception:
                    can_fit = False
        
        if not can_fit:
            noise = rng.normal(0, scale, dim) * global_widths
            x_candidate = best_x + noise

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)
