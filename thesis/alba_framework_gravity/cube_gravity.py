"""
Cube Gravity Module for ALBA

Implements physics-inspired gravitational field at the CUBE level,
as suggested by ChatGPT. This is more integrated with ALBA's architecture.

Key differences from point-based gravity:
1. Potential Φ_c = EMA(f) per cube (not per point)
2. Force between cubes based on potential difference
3. Integrates into LEAF SELECTION (not just local search)
4. Penalizes over-visited cubes (repulsion)

Formula for cube selection:
    score(c) = -Φ_c + λ * Σ attraction(c, c') - μ * visits(c)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cube import Cube


class CubeGravity:
    """
    Manages gravitational field at the cube level.
    
    Each cube has:
    - potential Φ = EMA of observed loss values
    - mass = confidence (based on visit count)
    - position = center of cube
    """
    
    def __init__(
        self,
        attraction_weight: float = 0.3,
        repulsion_weight: float = 0.1,
        ema_alpha: float = 0.3,
        visit_decay: float = 0.02,
    ):
        """
        Args:
            attraction_weight: λ - weight for attraction from better cubes
            repulsion_weight: μ - weight for visit-based repulsion
            ema_alpha: Smoothing for potential updates
            visit_decay: How much visits penalize score
        """
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.ema_alpha = ema_alpha
        self.visit_decay = visit_decay
        
        # Per-cube data (keyed by cube id)
        self.potential: Dict[int, float] = {}  # Φ_c = EMA of loss
        self.visits: Dict[int, int] = {}       # Visit count
        self.centers: Dict[int, np.ndarray] = {}  # Cube centers
        
        # Global stats for normalization
        self.global_potential_min = float('inf')
        self.global_potential_max = float('-inf')
        self.total_visits = 0
    
    def update_cube(self, cube: 'Cube', y_raw: float) -> None:
        """
        Update potential for a cube after observation.
        
        Args:
            cube: The cube that was sampled
            y_raw: Raw objective value (lower is better for minimization)
        """
        cube_id = id(cube)
        
        # Update visit count
        self.visits[cube_id] = self.visits.get(cube_id, 0) + 1
        self.total_visits += 1
        
        # Store cube center
        self.centers[cube_id] = cube.center()
        
        # Update potential Φ_c = EMA(f)
        if cube_id in self.potential:
            old_phi = self.potential[cube_id]
            new_phi = (1 - self.ema_alpha) * old_phi + self.ema_alpha * y_raw
        else:
            new_phi = y_raw
        
        self.potential[cube_id] = new_phi
        
        # Update global stats
        self.global_potential_min = min(self.global_potential_min, new_phi)
        self.global_potential_max = max(self.global_potential_max, new_phi)
    
    def compute_attraction(self, cube: 'Cube', all_cubes: List['Cube']) -> float:
        """
        Compute total attraction force on a cube from all other cubes.
        
        Attraction from cube c' to c:
            F = (Φ_c - Φ_c') / dist(c, c')
        
        Positive F means c' is better (lower potential) → attracts
        """
        cube_id = id(cube)
        if cube_id not in self.potential:
            return 0.0
        
        phi_c = self.potential[cube_id]
        center_c = self.centers[cube_id]
        
        total_attraction = 0.0
        
        for other in all_cubes:
            other_id = id(other)
            if other_id == cube_id or other_id not in self.potential:
                continue
            
            phi_other = self.potential[other_id]
            center_other = self.centers[other_id]
            
            # Distance between cube centers
            dist = np.linalg.norm(center_c - center_other) + 0.01
            
            # Attraction: positive if other has lower potential (better)
            # Using potential difference / distance (like gravitational force)
            attraction = (phi_c - phi_other) / dist
            
            total_attraction += attraction
        
        return total_attraction
    
    def get_gravity_score(self, cube: 'Cube', all_cubes: List['Cube']) -> float:
        """
        Compute gravity-based score for cube selection.
        
        score(c) = -Φ_c (lower potential is better)
                   + λ * attraction (pulled by better cubes)
                   - μ * visits (penalize over-exploration)
        """
        cube_id = id(cube)
        
        # Base score: negative potential (lower loss = higher score)
        if cube_id in self.potential:
            # Normalize potential to [0, 1]
            phi = self.potential[cube_id]
            if self.global_potential_max > self.global_potential_min:
                phi_norm = (phi - self.global_potential_min) / \
                          (self.global_potential_max - self.global_potential_min)
            else:
                phi_norm = 0.5
            base_score = -phi_norm  # Lower potential = higher score
        else:
            base_score = 0.0  # Unknown cubes get neutral score
        
        # Attraction term
        attraction = self.compute_attraction(cube, all_cubes)
        # Normalize attraction
        if len(all_cubes) > 1:
            attraction = attraction / len(all_cubes)
        
        # Repulsion from over-visiting
        visits = self.visits.get(cube_id, 0)
        avg_visits = self.total_visits / max(len(self.visits), 1)
        visit_penalty = max(0, visits - avg_visits) * self.visit_decay
        
        # Combined score
        score = base_score + self.attraction_weight * attraction - self.repulsion_weight * visit_penalty
        
        return score
    
    def get_drift_vector(self, x: np.ndarray, all_cubes: List['Cube']) -> np.ndarray:
        """
        Compute gravitational drift vector for local search.
        
        This is the direction the particle should drift based on
        the gravitational field from all cubes.
        
        F(x) = Σ_c  (Φ_max - Φ_c) * (center_c - x) / dist²
        
        Better cubes (lower Φ) attract more strongly.
        """
        n_dims = len(x)
        drift = np.zeros(n_dims)
        
        if not self.potential:
            return drift
        
        # Use inverted potential as "mass" (better = heavier = more attraction)
        phi_max = self.global_potential_max
        
        for cube_id, phi in self.potential.items():
            if cube_id not in self.centers:
                continue
            
            center = self.centers[cube_id]
            diff = center - x
            dist = np.linalg.norm(diff) + 0.01
            
            # Mass = how much better than worst
            mass = max(0, phi_max - phi) + 0.1  # +0.1 to avoid zero mass
            
            # Gravitational attraction: mass / dist²
            force_mag = mass / (dist ** 2)
            
            # Direction toward cube center
            drift += force_mag * diff / dist
        
        # Normalize to unit vector
        drift_mag = np.linalg.norm(drift)
        if drift_mag > 0:
            drift = drift / drift_mag
        
        return drift
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about the gravity field."""
        if not self.potential:
            return {'n_cubes': 0}
        
        potentials = list(self.potential.values())
        visits_list = list(self.visits.values())
        
        return {
            'n_cubes': len(self.potential),
            'potential_min': min(potentials),
            'potential_max': max(potentials),
            'potential_range': max(potentials) - min(potentials),
            'avg_visits': np.mean(visits_list),
            'max_visits': max(visits_list),
            'total_visits': self.total_visits,
        }
