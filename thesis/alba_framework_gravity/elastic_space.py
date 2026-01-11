"""
Elastic Space Module for ALBA

Implements physics-inspired adaptive sampling based on local "stiffness".
The search space is treated as an elastic medium where:
- Stiff regions (high gradient variance) → smaller steps
- Soft regions (smooth landscape) → larger steps

Key concepts:
1. Per-cube stiffness: Updated via EMA of observed |Δf|
2. Diffusion: Stiffness propagates to neighboring cubes
3. Preconditioned sampling: step_scale ∝ 1/stiffness

This is "free" - no extra evaluations needed, uses existing history.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class ElasticSpace:
    """
    Manages per-cube stiffness for physics-based adaptive sampling.
    
    The space "learns" where it's rigid (high sensitivity) vs soft (smooth),
    and this information diffuses to neighboring regions.
    """
    
    def __init__(
        self,
        n_dims: int,
        n_bins: int = 10,
        ema_alpha: float = 0.3,
        diffusion_rate: float = 0.1,
        diffusion_every: int = 20,
        min_stiffness: float = 0.1,
        max_stiffness: float = 10.0,
        initial_stiffness: float = 1.0,
    ):
        """
        Args:
            n_dims: Number of continuous dimensions
            n_bins: Number of bins per dimension for discretization
            ema_alpha: EMA decay for stiffness updates (higher = faster adaptation)
            diffusion_rate: How much stiffness diffuses to neighbors (0-1)
            diffusion_every: Diffuse every N evaluations
            min_stiffness: Minimum allowed stiffness (prevents infinite steps)
            max_stiffness: Maximum allowed stiffness (prevents zero steps)
            initial_stiffness: Starting stiffness (1.0 = neutral)
        """
        self.n_dims = n_dims
        self.n_bins = n_bins
        self.ema_alpha = ema_alpha
        self.diffusion_rate = diffusion_rate
        self.diffusion_every = diffusion_every
        self.min_stiffness = min_stiffness
        self.max_stiffness = max_stiffness
        self.initial_stiffness = initial_stiffness
        
        # Per-cube stiffness (cube_id -> stiffness value)
        self.stiffness: Dict[Tuple[int, ...], float] = defaultdict(
            lambda: initial_stiffness
        )
        
        # Track observations per cube for confidence
        self.cube_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        
        # History for computing deltas
        self.last_values: Dict[Tuple[int, ...], float] = {}
        
        # Evaluation counter for diffusion trigger
        self.eval_count = 0
        
        # Per-dimension stiffness (for anisotropic scaling)
        self.dim_stiffness = np.ones(n_dims) * initial_stiffness
        self.dim_grad_ema = np.ones(n_dims)  # EMA of per-dim gradient
        
        # Statistics
        self.stats = {
            'updates': 0,
            'diffusions': 0,
            'avg_stiffness': initial_stiffness,
        }
    
    def _point_to_cube(self, x: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous point to cube index."""
        # Clip to [0, 1] and discretize
        x_clipped = np.clip(x, 0.0, 0.9999)
        indices = (x_clipped * self.n_bins).astype(int)
        return tuple(indices)
    
    def _get_neighbors(self, cube: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get all neighboring cubes (26-connected in nD, but only existing ones)."""
        neighbors = []
        for dim in range(len(cube)):
            for delta in [-1, 1]:
                neighbor = list(cube)
                neighbor[dim] = cube[dim] + delta
                if 0 <= neighbor[dim] < self.n_bins:
                    neighbor_tuple = tuple(neighbor)
                    # Only include if we have data there
                    if neighbor_tuple in self.stiffness:
                        neighbors.append(neighbor_tuple)
        return neighbors
    
    def update(self, x: np.ndarray, f_value: float) -> None:
        """
        Update stiffness based on new observation.
        
        Uses gradient estimation from nearby points (same cube or neighbors)
        to estimate local sensitivity. This works even with sparse sampling.
        """
        cube = self._point_to_cube(x)
        self.cube_counts[cube] += 1
        self.eval_count += 1
        
        # Store point and value for gradient computation
        if not hasattr(self, '_point_history'):
            self._point_history = []
        self._point_history.append((x.copy(), f_value, cube))
        
        # Keep only recent history (memory efficient)
        if len(self._point_history) > 500:
            self._point_history = self._point_history[-500:]
        
        # Find nearby points and compute local gradient estimate
        # This is key: we use spatial neighbors, not just same-cube revisits
        nearby_deltas = []
        for px, pf, pc in self._point_history[:-1]:  # Exclude current point
            dist = np.linalg.norm(x - px)
            if 0.001 < dist < 0.2:  # Nearby but not identical
                # Gradient magnitude estimate: |Δf| / |Δx|
                grad_est = abs(f_value - pf) / dist
                nearby_deltas.append(grad_est)
        
        if nearby_deltas:
            # Use median gradient as stiffness signal (robust to outliers)
            grad_signal = np.median(nearby_deltas)
            
            # Convert to stiffness: higher gradient = stiffer
            # Normalize by typical gradient scale
            if not hasattr(self, '_grad_ema'):
                self._grad_ema = grad_signal
            else:
                self._grad_ema = 0.9 * self._grad_ema + 0.1 * grad_signal
            
            # Relative stiffness: how much stiffer than average?
            if self._grad_ema > 0:
                relative_stiff = grad_signal / self._grad_ema
                new_signal = np.clip(relative_stiff, 0.2, 5.0)
            else:
                new_signal = 1.0
            
            # EMA update
            old_stiff = self.stiffness[cube]
            new_stiff = (1 - self.ema_alpha) * old_stiff + self.ema_alpha * new_signal
            self.stiffness[cube] = np.clip(new_stiff, self.min_stiffness, self.max_stiffness)
            
            self.stats['updates'] += 1
        
        self.last_values[cube] = f_value
        
        # Also update per-dimension stiffness
        self._update_dim_stiffness(x, f_value)
        
        # Trigger diffusion periodically
        if self.eval_count % self.diffusion_every == 0:
            self._diffuse()
    
    def _update_dim_stiffness(self, x: np.ndarray, f_value: float) -> None:
        """
        Update per-dimension stiffness based on approximate partial derivatives.
        
        Uses pairs of points to estimate |∂f/∂x_i| for each dimension.
        More robust approach: accumulate gradient estimates over time.
        """
        if not hasattr(self, '_point_history') or len(self._point_history) < 5:
            return
        
        # Estimate gradient using finite differences with nearby points
        for dim in range(self.n_dims):
            grad_estimates = []
            
            for px, pf, _ in self._point_history[-50:]:  # Recent history only
                diff = x - px
                
                # Point is useful for dim i if:
                # 1. The difference in dim i is significant
                # 2. Other dimensions don't change too much (isolates effect)
                abs_diff = np.abs(diff)
                if abs_diff[dim] < 0.02:  # Need meaningful difference
                    continue
                    
                # Compute how "aligned" this point pair is with axis i
                # Perfect alignment: only dim i differs
                alignment = abs_diff[dim] / (abs_diff.sum() + 1e-10)
                
                if alignment > 0.3:  # At least 30% of movement is in dim i
                    # Partial derivative estimate
                    partial = abs(f_value - pf) / abs_diff[dim]
                    # Weight by alignment quality
                    grad_estimates.append((partial, alignment))
            
            if grad_estimates:
                # Weighted average by alignment quality
                total_weight = sum(w for _, w in grad_estimates)
                weighted_grad = sum(g * w for g, w in grad_estimates) / total_weight
                
                # EMA update
                alpha = 0.2
                self.dim_grad_ema[dim] = (1 - alpha) * self.dim_grad_ema[dim] + alpha * weighted_grad
        
        # Convert gradients to relative stiffness
        avg_grad = self.dim_grad_ema.mean()
        if avg_grad > 1e-10:
            for dim in range(self.n_dims):
                relative = self.dim_grad_ema[dim] / avg_grad
                # Smooth update to stiffness
                self.dim_stiffness[dim] = 0.9 * self.dim_stiffness[dim] + 0.1 * np.clip(relative, 0.2, 5.0)
        
        # Trigger diffusion periodically
        if self.eval_count % self.diffusion_every == 0:
            self._diffuse()
    
    def _diffuse(self) -> None:
        """
        Diffuse stiffness to neighboring cubes.
        
        This is a single step of the heat equation:
        stiff_new = (1-η) * stiff + η * avg(neighbors)
        """
        if len(self.stiffness) < 2:
            return
        
        # Compute new values (don't modify during iteration)
        new_stiffness = {}
        
        for cube, stiff in self.stiffness.items():
            neighbors = self._get_neighbors(cube)
            if neighbors:
                neighbor_avg = np.mean([self.stiffness[n] for n in neighbors])
                new_stiff = (1 - self.diffusion_rate) * stiff + self.diffusion_rate * neighbor_avg
                new_stiffness[cube] = np.clip(new_stiff, self.min_stiffness, self.max_stiffness)
            else:
                new_stiffness[cube] = stiff
        
        # Apply updates
        for cube, stiff in new_stiffness.items():
            self.stiffness[cube] = stiff
        
        self.stats['diffusions'] += 1
        self.stats['avg_stiffness'] = np.mean(list(self.stiffness.values()))
    
    def get_step_scale(self, x: np.ndarray) -> float:
        """
        Get the step scale factor for sampling at point x.
        
        Returns scale ∝ 1/stiffness (softer = bigger steps).
        Normalized so scale=1.0 at initial_stiffness.
        """
        cube = self._point_to_cube(x)
        stiff = self.stiffness[cube]
        
        # Scale inversely with stiffness
        # At initial_stiffness (1.0), scale = 1.0
        # At high stiffness (10.0), scale = 0.1
        # At low stiffness (0.1), scale = 10.0 → clip to reasonable range
        scale = self.initial_stiffness / stiff
        
        # Clip to reasonable range [0.2, 2.0]
        return np.clip(scale, 0.2, 2.0)
    
    def get_per_dim_scale(self, x: np.ndarray) -> np.ndarray:
        """
        Get per-dimension scale factors.
        
        Uses learned per-dimension stiffness for anisotropic scaling.
        Scale ∝ 1/stiffness (softer dims get larger steps).
        """
        # Use per-dimension stiffness (learned from gradients)
        scales = self.initial_stiffness / self.dim_stiffness
        
        # Also blend with local cube stiffness
        cube = self._point_to_cube(x)
        local_scale = self.initial_stiffness / self.stiffness[cube]
        
        # Combine: per-dim anisotropy * local scale
        combined = scales * local_scale
        
        # Clip to reasonable range
        return np.clip(combined, 0.2, 2.0)
    
    def record_improvement(self, x: np.ndarray, improvement: float) -> None:
        """
        Record that an improvement was found at x.
        
        This implements "plasticity" - regions that give improvements
        become permanently softer (easier to traverse).
        """
        if improvement <= 0:
            return
        
        cube = self._point_to_cube(x)
        
        # Reduce stiffness (make softer) proportional to improvement
        # Larger improvements → bigger softening
        softening = 1.0 / (1.0 + improvement * 10)  # 0.5 at improvement=0.1
        
        old_stiff = self.stiffness[cube]
        new_stiff = old_stiff * softening
        self.stiffness[cube] = max(new_stiff, self.min_stiffness)
    
    def record_trap(self, x: np.ndarray) -> None:
        """
        Record that x led to a trap (worse than expected).
        
        This implements "plasticity" - trap regions become permanently
        stiffer (harder to traverse).
        """
        cube = self._point_to_cube(x)
        
        # Increase stiffness (make stiffer)
        old_stiff = self.stiffness[cube]
        new_stiff = old_stiff * 1.5
        self.stiffness[cube] = min(new_stiff, self.max_stiffness)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the elastic space."""
        if self.stiffness:
            stiff_values = list(self.stiffness.values())
            return {
                'n_cubes': len(self.stiffness),
                'updates': self.stats['updates'],
                'diffusions': self.stats['diffusions'],
                'avg_stiffness': np.mean(stiff_values),
                'min_stiffness': min(stiff_values),
                'max_stiffness': max(stiff_values),
                'stiffness_range': max(stiff_values) / max(min(stiff_values), 0.01),
            }
        return {'n_cubes': 0}
    
    def visualize(self, dim1: int = 0, dim2: int = 1) -> np.ndarray:
        """
        Create 2D visualization of stiffness field.
        
        Returns n_bins x n_bins array showing stiffness for dims dim1, dim2.
        """
        grid = np.full((self.n_bins, self.n_bins), self.initial_stiffness)
        
        for cube, stiff in self.stiffness.items():
            if len(cube) > max(dim1, dim2):
                i, j = cube[dim1], cube[dim2]
                # Take max stiffness seen at this (dim1, dim2) location
                grid[i, j] = max(grid[i, j], stiff)
        
        return grid


class ElasticLocalSearch:
    """
    Local search that uses ElasticSpace for adaptive step sizes.
    
    Replaces the AnisotropicLocalSearchSampler with a physics-based approach
    that learns from the optimization history.
    """
    
    def __init__(
        self,
        elastic_space: ElasticSpace,
        base_std: float = 0.1,
        use_per_dim: bool = True,
    ):
        """
        Args:
            elastic_space: The ElasticSpace instance to use
            base_std: Base standard deviation for Gaussian perturbation
            use_per_dim: If True, use per-dimension scaling; else scalar
        """
        self.elastic_space = elastic_space
        self.base_std = base_std
        self.use_per_dim = use_per_dim
    
    def sample(
        self,
        center: np.ndarray,
        rng: np.random.Generator,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Sample a point near center using elastic-scaled step.
        
        Args:
            center: Center point for local search
            rng: Random number generator
            bounds: Optional (lower, upper) bounds
            
        Returns:
            New point sampled with elastic scaling
        """
        n_dims = len(center)
        
        if self.use_per_dim:
            # Per-dimension scaling
            scales = self.elastic_space.get_per_dim_scale(center)
            perturbation = rng.normal(0, self.base_std * scales)
        else:
            # Scalar scaling
            scale = self.elastic_space.get_step_scale(center)
            perturbation = rng.normal(0, self.base_std * scale, size=n_dims)
        
        new_point = center + perturbation
        
        # Apply bounds if provided
        if bounds is not None:
            lower, upper = bounds
            new_point = np.clip(new_point, lower, upper)
        
        return new_point
    
    def sample_multiple(
        self,
        center: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Sample multiple points near center."""
        samples = np.array([
            self.sample(center, rng, bounds) for _ in range(n_samples)
        ])
        return samples


# Convenience function for integration with ALBA
def create_elastic_space_for_alba(
    n_continuous_dims: int,
    n_bins: int = 10,
    **kwargs
) -> ElasticSpace:
    """
    Factory function to create ElasticSpace configured for ALBA.
    
    Args:
        n_continuous_dims: Number of continuous dimensions in search space
        n_bins: Number of bins (should match ALBA's heatmap resolution)
        **kwargs: Additional arguments passed to ElasticSpace
        
    Returns:
        Configured ElasticSpace instance
    """
    return ElasticSpace(
        n_dims=n_continuous_dims,
        n_bins=n_bins,
        **kwargs
    )
