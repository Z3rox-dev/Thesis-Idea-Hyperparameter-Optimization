"""
Cube Gravity Module for ALBA

Implements physics-inspired gravitational field at the CUBE level.

Two drift modes are supported (used in local search):
1) "potential" (default): attract towards historically good cubes using a
   potential Φ_c = EMA(f) per cube.
2) "surrogate_gradient": use the local surrogate slope (LGS gradient) as a
   "gravity-like" acceleration, i.e. stronger where the estimated slope is
   larger.

Cube selection score (used by GravityLeafSelector, if enabled):
    score(c) = -Φ_c + λ * Σ attraction(c, c') - μ * visits(c)
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

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
        drift_mode: str = "potential",
        gradient_neighbor_k: int = 3,
        gradient_distance_power: float = 2.0,
    ):
        """
        Args:
            attraction_weight: λ - weight for attraction from better cubes
            repulsion_weight: μ - weight for visit-based repulsion
            ema_alpha: Smoothing for potential updates
            visit_decay: How much visits penalize score
            drift_mode: "potential" or "surrogate_gradient"
            gradient_neighbor_k: If using surrogate drift, number of nearest cubes
                (with gradients) to blend when the local cube has no gradient.
            gradient_distance_power: Distance power for neighbor weighting.
        """
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.ema_alpha = ema_alpha
        self.visit_decay = visit_decay

        self.drift_mode = str(drift_mode)
        self.gradient_neighbor_k = int(gradient_neighbor_k)
        self.gradient_distance_power = float(gradient_distance_power)
        
        # Per-cube data (keyed by cube id)
        self.potential: Dict[int, float] = {}  # Φ_c = EMA of loss
        self.visits: Dict[int, int] = {}       # Visit count
        self.centers: Dict[int, np.ndarray] = {}  # Cube centers
        
        # Global stats for normalization
        self.global_potential_min = float('inf')
        self.global_potential_max = float('-inf')
        self.total_visits = 0

    def _containing_or_closest_cube(self, x: np.ndarray, cubes: List["Cube"]) -> Optional["Cube"]:
        """Return the cube containing x, or the closest-by-center cube as fallback."""
        for c in cubes:
            try:
                if c.contains(x):
                    return c
            except Exception:
                continue
        if not cubes:
            return None
        try:
            return min(cubes, key=lambda c: float(np.linalg.norm(c.center() - x)))
        except Exception:
            return cubes[0]

    def _cube_lgs_gradient_x(self, cube: "Cube", *, dim: int) -> Optional[np.ndarray]:
        """
        Return an estimated gradient in x-space from the cube LGS model.

        LGS models y ≈ y_mean + ((x-center)/widths) · grad, so:
            ∂y/∂x ≈ grad / widths
        """
        model = getattr(cube, "lgs_model", None)
        if model is None:
            return None
        grad = model.get("grad")
        if grad is None:
            return None
        grad = np.asarray(grad, dtype=float)
        if getattr(grad, "shape", None) != (dim,):
            return None

        widths = model.get("widths")
        if widths is None:
            try:
                widths = cube.widths()
            except Exception:
                return None
        widths = np.asarray(widths, dtype=float)
        if getattr(widths, "shape", None) != (dim,):
            return None
        widths = np.maximum(widths, 1e-12)

        g = grad / widths

        # Categorical dims: don't drift; categorical sampler handles them.
        cat_dims = getattr(cube, "categorical_dims", tuple())
        for i in cat_dims:
            ii = int(i)
            if 0 <= ii < dim:
                g[ii] = 0.0

        if not np.all(np.isfinite(g)):
            return None
        return g

    def _surrogate_gradient_drift(
        self,
        x: np.ndarray,
        cubes: List["Cube"],
        *,
        normalize: bool,
    ) -> np.ndarray:
        """Slope-based drift: use LGS gradient (stronger where slope is larger)."""
        dim = int(len(x))
        drift = np.zeros(dim, dtype=float)
        if not cubes:
            return drift

        # Prefer the local cube (containing x) if it has a gradient.
        local = self._containing_or_closest_cube(np.asarray(x, dtype=float), cubes)
        if local is not None:
            g_local = self._cube_lgs_gradient_x(local, dim=dim)
            if g_local is not None:
                drift = g_local
                if normalize:
                    n = float(np.linalg.norm(drift))
                    if np.isfinite(n) and n > 1e-12:
                        drift = drift / n
                return drift

        # Fallback: blend nearest cubes that have gradients (smooth across boundaries).
        grads: List[np.ndarray] = []
        dists: List[float] = []
        x0 = np.asarray(x, dtype=float)

        for c in cubes:
            g = self._cube_lgs_gradient_x(c, dim=dim)
            if g is None:
                continue
            try:
                d = float(np.linalg.norm(c.center() - x0))
            except Exception:
                d = float("inf")
            grads.append(g)
            dists.append(d)

        if not grads:
            return drift

        k = max(1, min(int(self.gradient_neighbor_k), len(grads)))
        idx_sorted = np.argsort(np.asarray(dists, dtype=float))[:k]

        p = float(self.gradient_distance_power)
        weights = []
        for j in idx_sorted:
            d = float(dists[int(j)])
            weights.append(1.0 / ((d + 1e-3) ** p))
        w = np.asarray(weights, dtype=float)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            return drift
        w = w / w_sum

        for wj, j in zip(w, idx_sorted):
            drift = drift + float(wj) * np.asarray(grads[int(j)], dtype=float)

        if normalize:
            n = float(np.linalg.norm(drift))
            if np.isfinite(n) and n > 1e-12:
                drift = drift / n
        return drift
    
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
        
        # Store cube center (geometric position for this cube).
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
    
    def get_drift_vector(
        self,
        x: np.ndarray,
        all_cubes: List["Cube"],
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute gravitational drift vector for local search.
        
        This is the direction the particle should drift based on
        the gravitational field from all cubes.
        
        Drift modes:
        - "potential": F(x) = Σ_c (Φ_max - Φ_c) * (center_c - x) / dist²
        - "surrogate_gradient": use the local surrogate slope (LGS gradient)
        """
        x = np.asarray(x, dtype=float)
        n_dims = int(len(x))
        drift = np.zeros(n_dims, dtype=float)

        mode = str(self.drift_mode).strip().lower()
        if mode in {"surrogate_gradient", "gradient", "slope"}:
            return self._surrogate_gradient_drift(x, all_cubes, normalize=normalize)
        if mode not in {"potential", ""}:
            raise ValueError(f"Unknown drift_mode={self.drift_mode!r}; expected 'potential' or 'surrogate_gradient'")
        
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
        
        if normalize:
            # Normalize to unit vector (direction-only).
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
