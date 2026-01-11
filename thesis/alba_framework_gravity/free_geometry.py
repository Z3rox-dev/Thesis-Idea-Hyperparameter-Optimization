"""
Free Geometry Estimation Module for ALBA

Estimates per-dimension sensitivity from observed points WITHOUT extra evaluations.
Uses pairs of nearby points to estimate |∂f/∂x_i| for each dimension.

This information is used to MODULATE the gravitational drift:
- On high-sensitivity dimensions: smaller drift steps
- On low-sensitivity dimensions: larger drift steps

Key advantage: ZERO extra function evaluations - uses only existing history.
"""

import numpy as np
from typing import List, Tuple


class FreeGeometryEstimator:
    """
    Estimates per-dimension sensitivity from observation history.
    
    The estimation uses pairs of nearby points that differ primarily
    along one dimension to estimate partial derivatives.
    """
    
    def __init__(
        self,
        n_dims: int,
        ema_alpha: float = 0.2,
        min_diff: float = 0.02,
        alignment_threshold: float = 0.4,
        history_window: int = 50,
    ):
        """
        Args:
            n_dims: Number of dimensions
            ema_alpha: EMA smoothing for sensitivity updates
            min_diff: Minimum difference required in a dimension
            alignment_threshold: Minimum fraction of movement in target dim
            history_window: Number of recent points to consider
        """
        self.n_dims = n_dims
        self.ema_alpha = ema_alpha
        self.min_diff = min_diff
        self.alignment_threshold = alignment_threshold
        self.history_window = history_window
        
        # Per-dimension sensitivity (higher = more sensitive)
        self.dim_sensitivity = np.ones(n_dims)
        
        # History of (x, f) pairs
        self.history: List[Tuple[np.ndarray, float]] = []
        
        # Statistics
        self.n_updates = 0
    
    def update(self, x: np.ndarray, f: float) -> None:
        """
        Update sensitivity estimates with new observation.
        
        Args:
            x: Point in normalized [0,1] space
            f: Objective value (raw scale)
        """
        self.history.append((x.copy(), f))
        self.n_updates += 1
        
        # Keep only recent history
        if len(self.history) > self.history_window * 2:
            self.history = self.history[-self.history_window * 2:]
        
        if len(self.history) < 10:
            return
        
        # Estimate gradient for each dimension
        for dim in range(self.n_dims):
            grad_estimates = []
            
            for px, pf in self.history[-self.history_window:]:
                diff = x - px
                abs_diff = np.abs(diff)
                
                # Check if point differs significantly in this dimension
                if abs_diff[dim] < self.min_diff:
                    continue
                
                # Compute alignment: how much of the movement is in dim i
                alignment = abs_diff[dim] / (abs_diff.sum() + 1e-10)
                
                if alignment >= self.alignment_threshold:
                    # Good point pair for estimating partial derivative
                    grad_est = abs(f - pf) / abs_diff[dim]
                    grad_estimates.append(grad_est)
            
            if grad_estimates:
                # Use median for robustness to outliers
                median_grad = np.median(grad_estimates)
                
                # EMA update
                self.dim_sensitivity[dim] = (
                    (1 - self.ema_alpha) * self.dim_sensitivity[dim] +
                    self.ema_alpha * median_grad
                )
    
    def modulate_drift(self, drift: np.ndarray) -> np.ndarray:
        """
        Modulate gravitational drift based on learned geometry.
        
        High sensitivity dimensions get smaller drift steps to avoid
        overshooting in steep directions.
        
        Args:
            drift: Raw drift vector from gravity
            
        Returns:
            Modulated drift vector
        """
        avg_sensitivity = self.dim_sensitivity.mean()
        
        if avg_sensitivity < 1e-10:
            return drift
        
        # Relative sensitivity (normalized to average)
        relative = self.dim_sensitivity / avg_sensitivity
        
        # Scale factor: inverse sqrt of relative sensitivity
        # High sensitivity -> smaller scale -> smaller drift
        scale = 1.0 / np.sqrt(relative + 0.1)
        scale = np.clip(scale, 0.3, 2.0)
        
        return drift * scale
    
    def get_scale(self) -> np.ndarray:
        """
        Get per-dimension scale factors.
        
        Returns:
            Scale factors where high sensitivity dims have lower scale.
        """
        avg = self.dim_sensitivity.mean()
        if avg > 1e-10:
            relative = self.dim_sensitivity / avg
            scale = 1.0 / np.sqrt(relative + 0.1)
            return np.clip(scale, 0.3, 2.0)
        return np.ones(self.n_dims)
    
    def get_stats(self) -> dict:
        """Get statistics about geometry estimation."""
        return {
            'n_updates': self.n_updates,
            'history_size': len(self.history),
            'sensitivity_min': float(self.dim_sensitivity.min()),
            'sensitivity_max': float(self.dim_sensitivity.max()),
            'sensitivity_ratio': float(self.dim_sensitivity.max() / (self.dim_sensitivity.min() + 1e-10)),
            'sensitivity_std': float(self.dim_sensitivity.std()),
        }
