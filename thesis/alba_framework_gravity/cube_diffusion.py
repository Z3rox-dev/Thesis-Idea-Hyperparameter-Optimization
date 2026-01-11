"""
Cube Diffusion Module for ALBA

Implements physics-inspired information diffusion between neighboring cubes.
This is more natural for ALBA's hierarchical cube structure than a global
elastic space.

Key concepts:
1. Each cube has a "field" value (e.g., average quality, exploration score)
2. Information diffuses between adjacent cubes via heat equation
3. This creates smooth transitions and helps exploration avoid local minima

This integrates directly with ALBA's cube tree structure.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cube import Cube


def cubes_are_adjacent(cube1: 'Cube', cube2: 'Cube') -> bool:
    """
    Check if two cubes are adjacent (share a face).
    
    Two cubes are adjacent if they touch along exactly one dimension
    and overlap in all other dimensions.
    """
    n_dims = len(cube1.bounds)
    
    touching_dims = 0
    overlapping_dims = 0
    
    for i in range(n_dims):
        lo1, hi1 = cube1.bounds[i]
        lo2, hi2 = cube2.bounds[i]
        
        # Check if they touch (one starts where other ends)
        if abs(hi1 - lo2) < 1e-10 or abs(hi2 - lo1) < 1e-10:
            touching_dims += 1
        # Check if they overlap
        elif lo1 < hi2 and lo2 < hi1:
            overlapping_dims += 1
    
    # Adjacent: touch in exactly 1 dim, overlap in all others
    return touching_dims == 1 and overlapping_dims == n_dims - 1


def find_adjacent_cubes(cube: 'Cube', all_leaves: List['Cube']) -> List['Cube']:
    """Find all leaf cubes adjacent to the given cube."""
    return [other for other in all_leaves if other is not cube and cubes_are_adjacent(cube, other)]


class CubeDiffusion:
    """
    Manages diffusion of information between ALBA cubes.
    
    Can diffuse various "fields" like:
    - Quality score (average y in cube)
    - Exploration bonus
    - Stiffness / sensitivity
    """
    
    def __init__(
        self,
        diffusion_rate: float = 0.1,
        diffusion_every: int = 10,
    ):
        """
        Args:
            diffusion_rate: How much information diffuses per step (0-1)
            diffusion_every: Diffuse every N evaluations
        """
        self.diffusion_rate = diffusion_rate
        self.diffusion_every = diffusion_every
        
        # Per-cube fields (cube_id -> value)
        self.quality_field: Dict[int, float] = {}  # Average quality
        self.exploration_field: Dict[int, float] = {}  # Exploration score
        
        self.eval_count = 0
        self.diffusion_count = 0
    
    def update_cube(self, cube: 'Cube', y_value: float) -> None:
        """Update fields for a cube after an observation."""
        cube_id = id(cube)
        
        # Update quality field (EMA of y values)
        alpha = 0.3
        old_quality = self.quality_field.get(cube_id, y_value)
        self.quality_field[cube_id] = (1 - alpha) * old_quality + alpha * y_value
        
        # Initialize exploration field if needed
        if cube_id not in self.exploration_field:
            self.exploration_field[cube_id] = 1.0  # Start neutral
        
        self.eval_count += 1
    
    def should_diffuse(self) -> bool:
        """Check if it's time to diffuse."""
        return self.eval_count > 0 and self.eval_count % self.diffusion_every == 0
    
    def diffuse(self, leaves: List['Cube']) -> None:
        """
        Perform one diffusion step across all leaf cubes.
        
        Uses heat equation: new = (1-η)*old + η*avg(neighbors)
        """
        if len(leaves) < 2:
            return
        
        # Build adjacency for current leaves
        adjacency: Dict[int, List[int]] = {}
        for cube in leaves:
            cube_id = id(cube)
            neighbors = find_adjacent_cubes(cube, leaves)
            adjacency[cube_id] = [id(n) for n in neighbors]
        
        # Diffuse quality field
        new_quality = {}
        for cube in leaves:
            cube_id = id(cube)
            if cube_id not in self.quality_field:
                continue
            
            old_val = self.quality_field[cube_id]
            neighbor_ids = adjacency.get(cube_id, [])
            
            if neighbor_ids:
                neighbor_vals = [
                    self.quality_field.get(nid, old_val) 
                    for nid in neighbor_ids
                ]
                neighbor_avg = np.mean(neighbor_vals)
                new_val = (1 - self.diffusion_rate) * old_val + self.diffusion_rate * neighbor_avg
            else:
                new_val = old_val
            
            new_quality[cube_id] = new_val
        
        # Apply updates
        self.quality_field.update(new_quality)
        
        # Diffuse exploration field
        new_exploration = {}
        for cube in leaves:
            cube_id = id(cube)
            if cube_id not in self.exploration_field:
                continue
            
            old_val = self.exploration_field[cube_id]
            neighbor_ids = adjacency.get(cube_id, [])
            
            if neighbor_ids:
                neighbor_vals = [
                    self.exploration_field.get(nid, old_val) 
                    for nid in neighbor_ids
                ]
                neighbor_avg = np.mean(neighbor_vals)
                new_val = (1 - self.diffusion_rate) * old_val + self.diffusion_rate * neighbor_avg
            else:
                new_val = old_val
            
            new_exploration[cube_id] = new_val
        
        self.exploration_field.update(new_exploration)
        self.diffusion_count += 1
    
    def get_quality_bonus(self, cube: 'Cube') -> float:
        """
        Get quality-based exploration bonus for a cube.
        
        Cubes near high-quality regions get a bonus from diffusion.
        """
        cube_id = id(cube)
        if cube_id not in self.quality_field:
            return 0.0
        
        # Normalize by range of quality values
        all_qualities = list(self.quality_field.values())
        if len(all_qualities) < 2:
            return 0.0
        
        q_min, q_max = min(all_qualities), max(all_qualities)
        if q_max - q_min < 1e-10:
            return 0.0
        
        # Higher quality = higher bonus
        normalized = (self.quality_field[cube_id] - q_min) / (q_max - q_min)
        return normalized * 0.5  # Scale bonus
    
    def get_exploration_bonus(self, cube: 'Cube') -> float:
        """
        Get exploration bonus based on diffused field.
        
        Cubes far from explored regions get higher bonus.
        """
        cube_id = id(cube)
        if cube_id not in self.exploration_field:
            return 0.5  # Unknown cubes get moderate bonus
        
        return self.exploration_field[cube_id] * 0.3
    
    def mark_explored(self, cube: 'Cube') -> None:
        """Mark a cube as explored (reduces its exploration field)."""
        cube_id = id(cube)
        old_val = self.exploration_field.get(cube_id, 1.0)
        self.exploration_field[cube_id] = old_val * 0.8  # Decay
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about diffusion state."""
        return {
            'n_cubes_tracked': len(self.quality_field),
            'diffusion_count': self.diffusion_count,
            'quality_range': max(self.quality_field.values()) - min(self.quality_field.values()) if self.quality_field else 0,
        }


def visualize_cube_field(
    leaves: List['Cube'],
    field: Dict[int, float],
    dim1: int = 0,
    dim2: int = 1,
) -> np.ndarray:
    """
    Create 2D visualization of a field across cubes.
    
    Returns a grid showing field values at cube centers.
    """
    resolution = 50
    grid = np.zeros((resolution, resolution))
    counts = np.zeros((resolution, resolution))
    
    for cube in leaves:
        cube_id = id(cube)
        if cube_id not in field:
            continue
        
        # Get cube center in dims dim1, dim2
        center1 = (cube.bounds[dim1][0] + cube.bounds[dim1][1]) / 2
        center2 = (cube.bounds[dim2][0] + cube.bounds[dim2][1]) / 2
        
        # Map to grid
        i = int(center1 * (resolution - 1))
        j = int(center2 * (resolution - 1))
        
        grid[i, j] += field[cube_id]
        counts[i, j] += 1
    
    # Average where multiple cubes map to same cell
    mask = counts > 0
    grid[mask] /= counts[mask]
    
    return grid
