"""ALBA Framework - Geometry Probing

This module implements local geometry probing for anisotropic step-size adaptation.

CORE IDEA:
Occasionally spend one evaluation to probe the local geometry around the 
current best point by evaluating a perturbed point along a single dimension.

The magnitude |Δ_i| = |f(x_probe) - f(x*)| estimates how "rigid" or "flat" 
that dimension is locally.

This information is used to bias local search:
- Smaller steps on high-sensitivity dimensions
- Larger steps on low-sensitivity dimensions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np


@dataclass
class GeometryProber:
    """
    Manages geometry probing state and logic.
    
    Probing is triggered occasionally (controlled by internal heuristics)
    and cycles through dimensions round-robin.
    
    Parameters
    ----------
    dim : int
        Number of dimensions in the search space.
    continuous_mask : Optional[np.ndarray]
        Boolean mask indicating which dimensions are continuous.
        If None, all dimensions are treated as continuous.
    probe_epsilon : float
        Relative perturbation size for probing (as fraction of width).
    ema_alpha : float
        EMA smoothing factor for sensitivity updates.
    probe_interval : int
        Minimum iterations between probes.
    """
    
    dim: int
    continuous_mask: Optional[np.ndarray] = None
    probe_epsilon: float = 0.05
    ema_alpha: float = 0.4
    probe_interval: int = 12  # Balanced: good coverage without too much cost
    
    # Internal state (initialized in __post_init__)
    _dim_sensitivity: np.ndarray = field(default_factory=lambda: np.array([]))
    _probe_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    _last_probe_iter: int = field(default=-100)
    _next_probe_dim: int = field(default=0)
    _pending_probe: Optional[dict] = field(default=None)
    _probe_baseline_y: Optional[float] = field(default=None)
    _total_probes: int = field(default=0)
    
    def __post_init__(self):
        """Initialize arrays after dataclass init."""
        self._dim_sensitivity = np.ones(self.dim) * 0.5  # Start with neutral sensitivity
        self._probe_counts = np.zeros(self.dim, dtype=int)
        
        # If no continuous mask provided, assume all continuous
        if self.continuous_mask is None:
            self.continuous_mask = np.ones(self.dim, dtype=bool)
        else:
            self.continuous_mask = np.asarray(self.continuous_mask, dtype=bool)
        
        # Count continuous dimensions for round-robin
        self._continuous_dims = np.where(self.continuous_mask)[0]
        if len(self._continuous_dims) == 0:
            # No continuous dims -> probing disabled
            self._continuous_dims = np.array([])
    
    @property
    def sensitivity(self) -> np.ndarray:
        """Get current per-dimension sensitivity estimates."""
        return self._dim_sensitivity.copy()
    
    @property
    def has_pending_probe(self) -> bool:
        """Check if there's a pending probe awaiting result."""
        return self._pending_probe is not None
    
    def should_probe(
        self,
        iteration: int,
        stagnation: int,
        phase: str,
    ) -> bool:
        """
        Decide whether to perform a probe this iteration.
        
        Probing is triggered:
        - Only if we have continuous dimensions
        - Not if there's already a pending probe
        - During local search phase OR when stagnating
        - Respecting minimum interval between probes
        - NOT if we've detected isotropic landscape (early stopping)
        
        Parameters
        ----------
        iteration : int
            Current optimization iteration.
        stagnation : int
            Number of iterations without improvement.
        phase : str
            Current phase ('exploration' or 'local_search').
            
        Returns
        -------
        bool
            True if probing should be performed.
        """
        # Skip if no continuous dimensions or pending probe
        if len(self._continuous_dims) == 0 or self._pending_probe is not None:
            return False
        
        # EARLY STOPPING: If we've done enough probes and detected isotropic landscape,
        # stop wasting budget on probing (check this FIRST before other conditions)
        # We need at least dim/2 probes to get a reasonable estimate
        min_probes = max(len(self._continuous_dims) // 2, 3)  # At least 3, or dim/2
        if self._total_probes >= min_probes:
            log_sens = np.log(self._dim_sensitivity + 1e-8)
            log_range = np.max(log_sens) - np.min(log_sens)
            if log_range < 2.0:
                # Isotropic detected - stop probing permanently
                return False
        
        # Check interval
        if iteration - self._last_probe_iter < self.probe_interval:
            return False
        
        # Early phase: aggressive probing to cover all dimensions at least once
        # We want at least dim probes in first ~2*dim*interval iterations
        if self._total_probes < len(self._continuous_dims):
            return True
        
        # Probe more aggressively during:
        # 1. Local search phase
        # 2. Stagnation periods
        if phase == 'local_search':
            return True
        
        if stagnation > 10:
            # Stagnating -> probe to gather geometry info
            return True
        
        return False
    
    def generate_probe_point(
        self,
        best_x: np.ndarray,
        bounds: List[Tuple[float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate a probe point by perturbing best_x along one dimension.
        
        Parameters
        ----------
        best_x : np.ndarray
            Current best point.
        bounds : List[Tuple[float, float]]
            Search space bounds.
        rng : np.random.Generator
            Random number generator.
            
        Returns
        -------
        np.ndarray
            Probe point.
        """
        if len(self._continuous_dims) == 0:
            return best_x.copy()
        
        # Select dimension (round-robin over continuous dims)
        dim_idx = self._continuous_dims[self._next_probe_dim % len(self._continuous_dims)]
        self._next_probe_dim = (self._next_probe_dim + 1) % len(self._continuous_dims)
        
        # Compute perturbation
        lo, hi = bounds[dim_idx]
        width = hi - lo
        
        # Perturbation direction: random sign for unbiased estimation
        # Note: This consumes RNG state, which changes the optimization trajectory
        # when probing is enabled. This is an unavoidable trade-off.
        direction = rng.choice([-1, 1])
        delta = direction * self.probe_epsilon * width
        
        # Create probe point
        x_probe = best_x.copy()
        new_val = best_x[dim_idx] + delta
        
        # Clip to bounds
        x_probe[dim_idx] = np.clip(new_val, lo, hi)
        
        # Store pending probe info
        self._pending_probe = {
            'dim_idx': dim_idx,
            'x_probe': x_probe.copy(),
            'x_base': best_x.copy(),
            'delta_x': x_probe[dim_idx] - best_x[dim_idx],
        }
        
        return x_probe
    
    def record_baseline(self, y_baseline: float, iteration: int) -> None:
        """
        Record the baseline objective value (at best_x) before probing.
        
        Parameters
        ----------
        y_baseline : float
            Objective value at current best point.
        iteration : int
            Current iteration.
        """
        self._probe_baseline_y = y_baseline
        self._last_probe_iter = iteration
    
    def update_from_probe(self, y_probe: float) -> int:
        """
        Update sensitivity estimates from completed probe.
        
        Parameters
        ----------
        y_probe : float
            Objective value at probe point (internal scale: higher is better).
            
        Returns
        -------
        int
            Dimension that was probed.
        """
        if self._pending_probe is None or self._probe_baseline_y is None:
            return -1
        
        dim_idx = self._pending_probe['dim_idx']
        delta_x = abs(self._pending_probe['delta_x'])
        x_base_val = abs(self._pending_probe['x_base'][dim_idx])
        
        # Compute objective difference (absolute value)
        delta_y = abs(y_probe - self._probe_baseline_y)
        
        # Normalize by perturbation size to get sensitivity estimate
        # Key insight: For f(x) = sum(c_i * x_i^2), df/dx_i = 2*c_i*x_i
        # We want to estimate c_i (intrinsic curvature), not 2*c_i*x_i
        # So we normalize by |x_i| as well to remove position dependence
        if delta_x > 1e-12:
            raw_sensitivity = delta_y / delta_x
            # Normalize by position to get curvature-like estimate
            # Add small epsilon to avoid division by zero near origin
            position_factor = max(x_base_val, 0.1)
            sensitivity = raw_sensitivity / position_factor
        else:
            sensitivity = 0.0
        
        # Update EMA
        old_sens = self._dim_sensitivity[dim_idx]
        self._dim_sensitivity[dim_idx] = (
            self.ema_alpha * sensitivity + (1 - self.ema_alpha) * old_sens
        )
        
        # Update counts
        self._probe_counts[dim_idx] += 1
        self._total_probes += 1
        
        # Clear pending state
        self._pending_probe = None
        self._probe_baseline_y = None
        
        return dim_idx
    
    def get_statistics(self) -> dict:
        """Get probing statistics for diagnostics."""
        return {
            'total_probes': self._total_probes,
            'probe_counts': self._probe_counts.tolist(),
            'sensitivity': self._dim_sensitivity.tolist(),
            'continuous_dims': self._continuous_dims.tolist(),
        }
