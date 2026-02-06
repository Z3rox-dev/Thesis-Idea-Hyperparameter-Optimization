"""
ALBA Framework - Geometric Coherence Module

This module implements geometric coherence scoring for leaf-wise gradient fields.

The key insight: if each leaf has a local gradient estimate g_l, a globally
consistent surrogate should satisfy:
    f(c_m) - f(c_l) ≈ g_l^T (c_m - c_l)  for neighboring leaves l, m

We project onto a global potential via least-squares and measure residuals
as a coherence score. High coherence → trust gradient (exploit).
Low coherence → gradient unreliable (explore).

This provides a principled, data-driven gating mechanism without magic thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

if TYPE_CHECKING:
    from .cube import Cube


@dataclass
class CoherenceCache:
    """Cache for coherence computations to avoid recomputing on every sample."""
    
    # Per-leaf coherence scores (higher = more coherent = trust gradient)
    scores: Dict[int, float] = field(default_factory=dict)

    # Per-leaf global potential values (lower = better/deeper minimum)
    potentials: Dict[int, float] = field(default_factory=dict)
    
    # Thresholds for exploit/explore gating (data-driven percentiles)
    q60_threshold: float = 0.0
    q80_threshold: float = 0.0
    
    # Global coherence metric
    global_coherence: float = 0.5
    
    # Iteration when cache was last updated
    last_update_iter: int = -1
    
    # Number of leaves when cache was built
    n_leaves_cached: int = 0


def _build_knn_graph(
    leaves: List["Cube"],
    k: int = 6,
) -> List[Tuple[int, int]]:
    """
    Build k-nearest neighbor graph on leaf centers.
    
    Uses normalized distances (each dimension scaled by its range) so that
    all dimensions contribute equally regardless of their original scale.
    
    Parameters
    ----------
    leaves : List[Cube]
        List of leaf cubes.
    k : int
        Number of nearest neighbors per leaf.
        
    Returns
    -------
    List[Tuple[int, int]]
        List of directed edges (i, j) where j is a neighbor of i.
    """
    n = len(leaves)
    if n < 2:
        return []
    
    # Compute centers and global widths for normalization
    centers = np.array([leaf.center() for leaf in leaves])
    
    # Get global bounds from first leaf (assumes all cubes share same original space)
    # Use the root bounds if available, otherwise estimate from leaves
    if leaves:
        # Estimate global widths from the range of centers
        global_widths = np.ptp(centers, axis=0)
        global_widths = np.where(global_widths < 1e-9, 1.0, global_widths)  # Avoid div by zero
    else:
        global_widths = np.ones(centers.shape[1])
    
    # Normalize centers
    centers_normalized = centers / global_widths
    
    edges = []
    k_actual = min(k, n - 1)
    
    for i in range(n):
        # Compute normalized squared distances
        dists = np.sum((centers_normalized - centers_normalized[i]) ** 2, axis=1)
        dists[i] = np.inf  # Exclude self
        
        # Get k nearest neighbors
        neighbor_idxs = np.argpartition(dists, k_actual)[:k_actual]
        
        for j in neighbor_idxs:
            edges.append((i, j))
    
    return edges


def _compute_predicted_drops(
    leaves: List["Cube"],
    edges: List[Tuple[int, int]],
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Compute predicted drops d_lm = g_l^T (c_m - c_l) for each edge.
    
    Uses normalized gradients and directions for scale-invariance.
    
    Parameters
    ----------
    leaves : List[Cube]
        List of leaf cubes.
    edges : List[Tuple[int, int]]
        List of directed edges.
    categorical_dims : Optional[List[Tuple[int, int]]]
        List of (dim_idx, n_choices) for categorical dimensions.
        Gradients on these dimensions are zeroed out.
        
    Returns
    -------
    d_lm : np.ndarray
        Predicted drops (normalized) for valid edges.
    direction_alignments : np.ndarray
        Cosine similarities between gradient directions for valid edges.
    valid_edges : List[Tuple[int, int]]
        Edges where both leaves have valid gradients.
    """
    cat_dims_set = set(d for d, _ in (categorical_dims or []))
    
    d_lm_list = []
    alignment_list = []
    valid_edges = []
    
    for i, j in edges:
        leaf_i = leaves[i]
        leaf_j = leaves[j]
        
        # Both leaves must have LGS models with valid gradients
        if leaf_i.lgs_model is None or leaf_j.lgs_model is None:
            continue
        
        g_i = leaf_i.lgs_model.get("grad")
        g_j = leaf_j.lgs_model.get("grad")
        if g_i is None or g_j is None:
            continue
        
        # Get centers
        c_i = leaf_i.center()
        c_j = leaf_j.center()
        
        # Compute direction vector
        delta = c_j - c_i
        
        # Zero out categorical dimensions
        g_i_masked = g_i.copy()
        g_j_masked = g_j.copy()
        delta_masked = delta.copy()
        for dim_idx in cat_dims_set:
            if dim_idx < len(g_i_masked):
                g_i_masked[dim_idx] = 0.0
                g_j_masked[dim_idx] = 0.0
                delta_masked[dim_idx] = 0.0
        
        # Compute distance
        dist = np.linalg.norm(delta_masked)
        if dist < 1e-9:
            continue
        
        # Normalize gradients for scale-invariance
        g_i_norm = np.linalg.norm(g_i_masked)
        g_j_norm = np.linalg.norm(g_j_masked)
        if g_i_norm < 1e-9 or g_j_norm < 1e-9:
            continue
        
        g_i_unit = g_i_masked / g_i_norm
        g_j_unit = g_j_masked / g_j_norm
        delta_unit = delta_masked / dist
        
        # Compute predicted drop using normalized gradient  
        # This is the cosine of angle between gradient and direction
        d_ij = np.dot(g_i_unit, delta_unit)
        
        # Compute gradient alignment between neighboring leaves
        alignment = np.dot(g_i_unit, g_j_unit)
        
        d_lm_list.append(d_ij)
        alignment_list.append(alignment)
        valid_edges.append((i, j))
    
    if not valid_edges:
        return np.array([]), np.array([]), []
    
    return np.array(d_lm_list), np.array(alignment_list), valid_edges


def _solve_potential_least_squares(
    n_leaves: int,
    edges: List[Tuple[int, int]],
    d_lm: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve weighted least squares for potential values u_l.
    
    Minimize: sum_e w_e * (u_m - u_l - d_lm)^2
    Subject to: u_0 = 0 (fix gauge)
    
    Parameters
    ----------
    n_leaves : int
        Number of leaves.
    edges : List[Tuple[int, int]]
        List of valid edges.
    d_lm : np.ndarray
        Predicted drops for each edge.
    weights : Optional[np.ndarray]
        Weights for each edge. Default: uniform.
        
    Returns
    -------
    u : np.ndarray
        Potential values for each leaf.
    """
    n_edges = len(edges)
    if n_edges == 0 or n_leaves < 2:
        return np.zeros(n_leaves)
    
    if weights is None:
        weights = np.ones(n_edges)
    
    # Build sparse matrix A where A[e, :] = (0, ..., -1, ..., +1, ..., 0)
    # for edge e = (i, j): coefficient -1 at i, +1 at j
    # We fix u[0] = 0 by removing column 0 and solving for u[1:]
    
    row_idx = []
    col_idx = []
    data = []
    
    for e, (i, j) in enumerate(edges):
        sqrt_w = np.sqrt(weights[e])
        
        if i > 0:  # Column 0 is removed (u[0] = 0)
            row_idx.append(e)
            col_idx.append(i - 1)
            data.append(-sqrt_w)
        
        if j > 0:
            row_idx.append(e)
            col_idx.append(j - 1)
            data.append(sqrt_w)
    
    if n_leaves <= 1:
        return np.zeros(n_leaves)
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_edges, n_leaves - 1))
    b = np.sqrt(weights) * d_lm
    
    # Solve with LSQR (handles sparse matrices efficiently)
    result = lsqr(A, b, damp=0.01)
    u_reduced = result[0]
    
    # Reconstruct full u with u[0] = 0
    u = np.zeros(n_leaves)
    u[1:] = u_reduced
    
    return u


def compute_coherence_scores(
    leaves: List["Cube"],
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
    k_neighbors: int = 6,
) -> Tuple[Dict[int, float], Dict[int, float], float, float, float]:
    """
    Compute geometric coherence scores for all leaves.
    
    MATHEMATICAL FOUNDATION:
    ========================
    For a scalar function f, the gradient ∇f points in the direction of
    steepest INCREASE. For minimization, we want to move AGAINST the gradient.
    
    The potential field u is computed by integrating gradient predictions:
        u_j - u_i ≈ g_i · (c_j - c_i)
    
    where g_i is the local gradient estimate at leaf i.
    
    For a convex function with minimum at x*:
    - ∇f(x) points AWAY from x* (toward higher values)
    - Moving from x=4 to x=2 (toward minimum): g·Δ < 0, so u decreases
    - Therefore: u is LOWER near the minimum, HIGHER far from it
    
    COHERENCE INTERPRETATION:
    - High coherence = gradients align well = field is conservative
    - Low coherence = gradients conflict = field has "curl" (inconsistent)
    
    Parameters
    ----------
    leaves : List[Cube]
        List of leaf cubes.
    categorical_dims : Optional[List[Tuple[int, int]]]
        Categorical dimensions to mask.
    k_neighbors : int
        Number of nearest neighbors for graph construction.
        
    Returns
    -------
    scores : Dict[int, float]
        Coherence score for each leaf in [0, 1]. Higher = more coherent.
    potentials : Dict[int, float]
        Potential value for each leaf in [0, 1]. LOWER = better (closer to minimum).
    global_coherence : float
        Global coherence metric in [0, 1].
    q60, q80 : float
        Percentile thresholds for exploit/explore gating.
    """
    n = len(leaves)
    
    if n < 3:
        return {i: 0.5 for i in range(n)}, {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Step 1: Build kNN graph
    edges = _build_knn_graph(leaves, k=k_neighbors)
    
    if not edges:
        return {i: 0.5 for i in range(n)}, {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Step 2: Compute gradient-based predictions
    d_lm, alignments, valid_edges = _compute_predicted_drops(
        leaves, edges, categorical_dims
    )
    
    if len(valid_edges) < 2:
        return {i: 0.5 for i in range(n)}, {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Step 3: Compute per-leaf coherence from gradient alignments
    leaf_alignments: Dict[int, List[float]] = {i: [] for i in range(n)}
    
    for e, (i, j) in enumerate(valid_edges):
        leaf_alignments[i].append(alignments[e])
        leaf_alignments[j].append(alignments[e])
    
    scores: Dict[int, float] = {}
    all_coherences = []
    
    for i in range(n):
        if leaf_alignments[i]:
            mean_align = float(np.mean(leaf_alignments[i]))
            coherence = (mean_align + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        else:
            coherence = 0.5
        scores[i] = coherence
        all_coherences.append(coherence)
    
    # Step 4: Solve global potential field
    # Weight edges by alignment quality (higher alignment = more trustworthy)
    weights = np.clip(alignments + 1.0, 0.1, 2.0)
    u = _solve_potential_least_squares(n, valid_edges, d_lm, weights)
    
    # Step 5: Compute empirical quality signal (density-based)
    # This provides robustness when gradient field is noisy
    leaf_densities = np.zeros(n)
    for i in range(n):
        vol = leaves[i].volume()
        if vol < 1e-12:
            vol = 1e-12
        leaf_densities[i] = leaves[i].n_good / vol
    
    # Sanitize and normalize densities
    valid_mask = np.isfinite(leaf_densities)
    if valid_mask.any() and np.std(leaf_densities[valid_mask]) > 1e-9:
        d_min = np.min(leaf_densities[valid_mask])
        d_max = np.max(leaf_densities[valid_mask])
        if d_max > d_min:
            density_norm = (leaf_densities - d_min) / (d_max - d_min)
        else:
            density_norm = np.full(n, 0.5)
        median_val = float(np.median(density_norm[valid_mask]))
        density_norm = np.where(valid_mask, density_norm, median_val)
    else:
        density_norm = np.full(n, 0.5)
    density_norm = np.clip(density_norm, 0.0, 1.0)
    
    # Step 6: Combine potential field with empirical signal
    # 
    # KEY INSIGHT: LGS gradients are INVERTED relative to ∇f!
    # 
    # This happens because ALBA stores y_internal = -f(x) for minimization.
    # LGS fits on y_internal, so LGS gradient = ∇y_internal = -∇f.
    # This means LGS gradient points TOWARD the minimum (not away).
    # 
    # When we integrate this field:
    # - Walking toward minimum: g·Δ > 0 (LGS gradient aligned with direction)
    # - So u INCREASES as we approach the minimum
    # 
    # We want: low potential = good = near minimum
    # So we MUST INVERT u: u_inverted = -u
    # 
    # After inversion:
    # - u_inverted is LOW near minimum (good)
    # - u_inverted is HIGH far from minimum (bad)
    u_inverted = -u
    
    # Density bonus: high density = promising region = lower potential
    # We subtract density (scaled) to boost promising regions
    density_bonus = density_norm * 1.5  # Moderate weight
    u_combined = u_inverted - density_bonus
    
    # Step 7: Anchor potential field
    # Find the leaf with LOWEST u_combined (best) and set it to 0
    best_leaf_idx = int(np.argmin(u_combined))
    u_anchored = u_combined - u_combined[best_leaf_idx]
    
    # Step 8: Normalize to [0, 1]
    # After anchoring, minimum is 0. Normalize so maximum is 1.
    u_var = np.var(u_anchored) if u_anchored.size > 0 else 0.0
    
    if u_var < 0.001:
        # Potential is uninformative, fall back to density
        # High density = low potential (good)
        if np.std(density_norm) > 0.01:
            u_norm = 1.0 - density_norm
        else:
            u_norm = np.full(n, 0.5)
    else:
        u_max = np.max(u_anchored)
        if u_max > 1e-9:
            u_norm = u_anchored / u_max
        else:
            u_norm = np.full(n, 0.5)
    
    # Clip to valid range
    u_norm = np.clip(u_norm, 0.0, 1.0)
    
    potentials = {i: float(u_norm[i]) for i in range(n)}

    # Step 9: Compute global metrics
    global_coherence = float(np.median(all_coherences)) if all_coherences else 0.5
    
    score_values = list(scores.values())
    if score_values:
        q60 = float(np.percentile(score_values, 60))
        q80 = float(np.percentile(score_values, 80))
    else:
        q60, q80 = 0.5, 0.5
    
    return scores, potentials, global_coherence, q60, q80


class CoherenceTracker:
    """
    Tracks and caches coherence scores for the optimizer.
    
    Provides the exploit/explore gating logic based on geometric coherence.
    """
    
    def __init__(
        self,
        categorical_dims: Optional[List[Tuple[int, int]]] = None,
        k_neighbors: int = 6,
        update_interval: int = 5,
        min_leaves_for_coherence: int = 5,
    ):
        """
        Parameters
        ----------
        categorical_dims : Optional[List[Tuple[int, int]]]
            Categorical dimensions to mask in gradient computations.
        k_neighbors : int
            Number of neighbors for kNN graph.
        update_interval : int
            Iterations between cache updates.
        min_leaves_for_coherence : int
            Minimum number of leaves before computing coherence.
        """
        self.categorical_dims = categorical_dims or []
        self.k_neighbors = k_neighbors
        self.update_interval = update_interval
        self.min_leaves_for_coherence = min_leaves_for_coherence
        
        self._cache = CoherenceCache()
        self._leaf_id_map: Dict[int, int] = {}  # id(cube) -> index
    
    def update(
        self,
        leaves: List["Cube"],
        iteration: int,
        force: bool = False,
    ) -> None:
        """
        Update coherence cache if needed.
        
        Parameters
        ----------
        leaves : List[Cube]
            Current list of leaves.
        iteration : int
            Current iteration.
        force : bool
            Force update even if not due.
        """
        # Check if update is needed
        n_leaves = len(leaves)
        iter_due = (iteration - self._cache.last_update_iter) >= self.update_interval
        structure_changed = n_leaves != self._cache.n_leaves_cached
        
        if not (force or iter_due or structure_changed):
            return
        
        # Skip if too few leaves
        if n_leaves < self.min_leaves_for_coherence:
            self._cache.scores = {i: 0.5 for i in range(n_leaves)}
            self._cache.potentials = {i: 0.5 for i in range(n_leaves)}
            self._cache.global_coherence = 0.5
            self._cache.q60_threshold = 0.5
            self._cache.q80_threshold = 0.5
            self._cache.last_update_iter = iteration
            self._cache.n_leaves_cached = n_leaves
            self._leaf_id_map = {id(leaf): i for i, leaf in enumerate(leaves)}
            return
        
        # Compute coherence scores
        scores, potentials, global_coh, q60, q80 = compute_coherence_scores(
            leaves,
            categorical_dims=self.categorical_dims,
            k_neighbors=self.k_neighbors,
        )
        
        # Update cache
        self._cache.scores = scores
        self._cache.potentials = potentials
        self._cache.global_coherence = global_coh
        self._cache.q60_threshold = q60
        self._cache.q80_threshold = q80
        self._cache.last_update_iter = iteration
        self._cache.n_leaves_cached = n_leaves
        self._leaf_id_map = {id(leaf): i for i, leaf in enumerate(leaves)}
    
    def get_coherence(self, cube: "Cube", leaves: List["Cube"]) -> float:
        """
        Get coherence score for a specific cube.
        
        Parameters
        ----------
        cube : Cube
            The cube to query.
        leaves : List[Cube]
            Current list of leaves (for index lookup).
            
        Returns
        -------
        float
            Coherence score in (0, 1]. Higher = more coherent.
        """
        cube_id = id(cube)
        
        if cube_id in self._leaf_id_map:
            idx = self._leaf_id_map[cube_id]
            return self._cache.scores.get(idx, 0.5)
        
        # Cube not in cache (possibly new) - return neutral
        return 0.5
    
    def get_potential(self, cube: "Cube", leaves: List["Cube"]) -> float:
        """
        Get global potential value for a specific cube.
        
        Parameters
        ----------
        cube : Cube
            The cube to query.
        leaves : List[Cube]
            Current list of leaves.
            
        Returns
        -------
        float
            Potential value in [0, 1]. Lower is better (deeper minimum).
        """
        cube_id = id(cube)
        
        if cube_id in self._leaf_id_map:
            idx = self._leaf_id_map[cube_id]
            return self._cache.potentials.get(idx, 0.5)
        
        return 0.5

    def should_exploit(self, cube: "Cube", leaves: List["Cube"]) -> bool:
        """
        Determine if we should exploit (use gradient) or explore (random).
        
        Uses data-driven threshold: exploit if coherence > Q60.
        
        Parameters
        ----------
        cube : Cube
            The cube to decide for.
        leaves : List[Cube]
            Current list of leaves.
            
        Returns
        -------
        bool
            True if should exploit (trust gradient), False if should explore.
        """
        coherence = self.get_coherence(cube, leaves)
        return coherence >= self._cache.q60_threshold
    
    @property
    def global_coherence(self) -> float:
        """Global coherence metric."""
        return self._cache.global_coherence
    
    @property
    def q60_threshold(self) -> float:
        """60th percentile coherence threshold."""
        return self._cache.q60_threshold
    
    def get_statistics(self) -> Dict[str, float]:
        """Get coherence statistics for diagnostics."""
        scores = list(self._cache.scores.values())
        return {
            "global_coherence": self._cache.global_coherence,
            "q60_threshold": self._cache.q60_threshold,
            "q80_threshold": self._cache.q80_threshold,
            "n_leaves_cached": self._cache.n_leaves_cached,
            "min_coherence": min(scores) if scores else 0.5,
            "max_coherence": max(scores) if scores else 0.5,
            "mean_coherence": float(np.mean(scores)) if scores else 0.5,
        }
