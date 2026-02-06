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
    
    # Compute centers
    centers = np.array([leaf.center() for leaf in leaves])
    
    # Compute pairwise distances
    # For efficiency, we compute squared distances
    edges = []
    k_actual = min(k, n - 1)
    
    for i in range(n):
        dists = np.sum((centers - centers[i]) ** 2, axis=1)
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
    distances : np.ndarray
        Euclidean distances for valid edges.
    direction_alignments : np.ndarray
        Cosine similarities between gradient directions for valid edges.
    valid_edges : List[Tuple[int, int]]
        Edges where both leaves have valid gradients.
    """
    cat_dims_set = set(d for d, _ in (categorical_dims or []))
    
    d_lm_list = []
    dist_list = []
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
        dist_list.append(dist)
        alignment_list.append(alignment)
        valid_edges.append((i, j))
    
    if not valid_edges:
        return np.array([]), np.array([]), np.array([]), []
    
    return np.array(d_lm_list), np.array(dist_list), np.array(alignment_list), valid_edges


def _solve_potential_least_squares(
    n_leaves: int,
    edges: List[Tuple[int, int]],
    d_lm: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    residuals : np.ndarray
        Residuals (u_m - u_l) - d_lm for each edge.
    """
    n_edges = len(edges)
    if n_edges == 0 or n_leaves < 2:
        return np.zeros(n_leaves), np.array([])
    
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
        return np.zeros(n_leaves), d_lm
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_edges, n_leaves - 1))
    b = np.sqrt(weights) * d_lm
    
    # Solve with LSQR (handles sparse matrices efficiently)
    result = lsqr(A, b, damp=0.01)
    u_reduced = result[0]
    
    # Reconstruct full u with u[0] = 0
    u = np.zeros(n_leaves)
    u[1:] = u_reduced
    
    # Compute residuals
    residuals = np.zeros(n_edges)
    for e, (i, j) in enumerate(edges):
        residuals[e] = (u[j] - u[i]) - d_lm[e]
    
    return u, residuals


def compute_coherence_scores(
    leaves: List["Cube"],
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
    k_neighbors: int = 6,
) -> Tuple[Dict[int, float], float, float, float]:
    """
    Compute geometric coherence scores for all leaves.
    
    Uses gradient direction alignment between neighboring leaves as the
    coherence metric. This is scale-invariant and robust to large gradients.
    
    Coherence is computed as:
    - For each leaf, compute cosine similarity between its gradient and
      gradients of neighboring leaves
    - High similarity → gradients point in similar directions → coherent
    - Low/negative similarity → gradients conflict → incoherent
    
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
        Coherence score for each leaf (keyed by leaf index).
        Higher = more coherent (trust gradient), in [0, 1].
    global_coherence : float
        Global coherence metric in [0, 1].
    q60 : float
        60th percentile threshold.
    q80 : float  
        80th percentile threshold.
    """
    n = len(leaves)
    
    if n < 3:
        # Not enough leaves for meaningful coherence
        return {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Step 1: Build kNN graph
    edges = _build_knn_graph(leaves, k=k_neighbors)
    
    if not edges:
        return {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Step 2: Compute gradient alignments
    d_lm, distances, alignments, valid_edges = _compute_predicted_drops(
        leaves, edges, categorical_dims
    )
    
    if len(valid_edges) < 2:
        return {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Step 3: Compute per-leaf coherence from gradient alignments
    # Aggregate alignments for each leaf
    leaf_alignments: Dict[int, List[float]] = {i: [] for i in range(n)}
    
    for e, (i, j) in enumerate(valid_edges):
        # Alignment is cosine similarity in [-1, 1]
        leaf_alignments[i].append(alignments[e])
        leaf_alignments[j].append(alignments[e])
    
    # Step 4: Convert to coherence scores
    # Use mean alignment, mapped from [-1, 1] to [0, 1]
    scores: Dict[int, float] = {}
    all_coherences = []
    
    for i in range(n):
        if leaf_alignments[i]:
            mean_align = float(np.mean(leaf_alignments[i]))
            # Map from [-1, 1] to [0, 1]
            coherence = (mean_align + 1.0) / 2.0
        else:
            coherence = 0.5  # No neighbors → neutral
        
        scores[i] = coherence
        all_coherences.append(coherence)
    
    # Step 5: Compute global metrics
    if all_coherences:
        global_coherence = float(np.median(all_coherences))
    else:
        global_coherence = 0.5
    
    # Compute percentile thresholds on coherence scores
    score_values = list(scores.values())
    if score_values:
        q60 = float(np.percentile(score_values, 60))
        q80 = float(np.percentile(score_values, 80))
    else:
        q60, q80 = 0.5, 0.5
    
    return scores, global_coherence, q60, q80


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
            self._cache.global_coherence = 0.5
            self._cache.q60_threshold = 0.5
            self._cache.q80_threshold = 0.5
            self._cache.last_update_iter = iteration
            self._cache.n_leaves_cached = n_leaves
            self._leaf_id_map = {id(leaf): i for i, leaf in enumerate(leaves)}
            return
        
        # Compute coherence scores
        scores, global_coh, q60, q80 = compute_coherence_scores(
            leaves,
            categorical_dims=self.categorical_dims,
            k_neighbors=self.k_neighbors,
        )
        
        # Update cache
        self._cache.scores = scores
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
