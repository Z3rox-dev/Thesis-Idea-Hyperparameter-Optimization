#!/usr/bin/env python3
"""
Test A/B: Verifica se rimuovere l'inversione del segno nel potential field migliora le performance.

Bug identificato:
- Riga 410-411 di coherence.py dice: "u_inverted = -u"
- Commento spiega: "For minimization: gradients point AWAY from minimum"
- MA questo è SBAGLIATO! I gradienti puntano verso dove f AUMENTA (massima crescita)
- Quindi gradienti puntano LONTANO dal minimo (corretto), ma il potential già riflette questo
- L'inversione del segno ROVINA il campo potenziale

Test: confronta ALBA con inversione (attuale) vs senza inversione (fix proposto)
"""

import sys
sys.path.insert(0, '/mnt/workspace')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors

# Import necessari
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube


def compute_coherence_scores_fixed(
    leaves: List["Cube"],
    categorical_dims: Optional[List[Tuple[int, int]]] = None,
    k_neighbors: int = 6,
) -> Tuple[Dict[int, float], Dict[int, float], float, float, float]:
    """
    FIXED VERSION: Same as original but WITHOUT sign inversion.
    
    The original code has:
        u_inverted = -u
        u_combined = u_inverted - empirical_bonus
    
    The fix is:
        u_combined = u - empirical_bonus  # No inversion!
    
    Reasoning:
    - For minimization, gradients point TOWARD higher values (away from minimum)
    - Potential field integral: higher potential = worse region = farther from minimum
    - This is already what we want! No inversion needed.
    - The empirical_bonus subtracts from potential (good leaves get bonus → lower potential)
    """
    from alba_framework_potential.coherence import (
        _build_knn_graph, 
        _compute_predicted_drops,
        _solve_potential_least_squares,
    )
    
    n = len(leaves)
    
    if n < 3:
        scores = {i: 0.5 for i in range(n)}
        potentials = {i: 0.5 for i in range(n)}
        return scores, potentials, 0.5, 0.5, 0.5
    
    # Step 1: Build kNN graph
    edges = _build_knn_graph(leaves, k_neighbors)
    
    if not edges:
        scores = {i: 0.5 for i in range(n)}
        potentials = {i: 0.5 for i in range(n)}
        return scores, potentials, 0.5, 0.5, 0.5
    
    # Step 2: Compute potential drops and alignment
    d_lm, alignment, valid_edges = _compute_predicted_drops(
        leaves, edges, categorical_dims
    )
    
    if len(valid_edges) == 0:
        scores = {i: 0.5 for i in range(n)}
        potentials = {i: 0.5 for i in range(n)}
        return scores, potentials, 0.5, 0.5, 0.5
    
    # Step 3: Convert alignment to coherence scores
    scores = {}
    all_coherences = []
    
    leaf_alignments = {i: [] for i in range(n)}
    for e, (i, j) in enumerate(valid_edges):
        leaf_alignments[i].append(alignment[e])
        leaf_alignments[j].append(alignment[e])
    
    for i in range(n):
        if leaf_alignments[i]:
            mean_align = float(np.mean(leaf_alignments[i]))
            coherence = (mean_align + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        else:
            coherence = 0.5
        scores[i] = coherence
        all_coherences.append(coherence)
    
    # Step 4: Solve for potential field
    weights = np.clip(alignment, 0.1, 1.0)
    u = _solve_potential_least_squares(n, valid_edges, d_lm, weights)
    
    # Step 5: Combine with empirical density signal - NO INVERSION
    leaf_densities = np.zeros(n)
    for i in range(n):
        vol = leaves[i].volume()
        if vol < 1e-12:
            vol = 1e-12
        leaf_densities[i] = leaves[i].n_good / vol
    
    valid_mask = np.isfinite(leaf_densities)
    if valid_mask.any() and np.std(leaf_densities[valid_mask]) > 1e-9:
        d_min = np.min(leaf_densities[valid_mask])
        d_max = np.max(leaf_densities[valid_mask])
        if d_max > d_min:
            leaf_densities_norm = (leaf_densities - d_min) / (d_max - d_min)
        else:
            leaf_densities_norm = np.full(n, 0.5)
        median_density = float(np.median(leaf_densities_norm[valid_mask]))
        leaf_densities_norm = np.where(valid_mask, leaf_densities_norm, median_density)
    else:
        leaf_densities_norm = np.full(n, 0.5)
    
    leaf_densities_norm = np.clip(leaf_densities_norm, 0.0, 1.0)
    
    # FIX: DO NOT invert u!
    # Original: u_inverted = -u
    # Fixed: just use u directly
    # Reasoning: potential already increases away from minimum (gradient direction)
    
    empirical_bonus = leaf_densities_norm * 2.0
    u_combined = u - empirical_bonus  # NO INVERSION! Was: u_inverted - empirical_bonus
    
    # Re-anchor so best leaf (highest density) has potential 0
    best_leaf_idx = int(np.argmax(leaf_densities_norm))
    u_anchored = u_combined - u_combined[best_leaf_idx]
    
    # Normalize
    u_var = np.var(u_anchored) if u_anchored.size > 0 else 0.0
    MIN_POTENTIAL_VARIANCE = 0.001
    
    if u_var < MIN_POTENTIAL_VARIANCE:
        if np.std(leaf_densities_norm) > 0.01:
            u_norm = 1.0 - leaf_densities_norm
        else:
            u_norm = np.full(n, 0.5)
    else:
        u_min = np.min(u_anchored)
        u_max = np.max(u_anchored)
        u_range = u_max - u_min
        
        if u_range < 1e-9:
            u_norm = np.full(n, 0.5)
        else:
            u_norm = (u_anchored - u_min) / u_range
    
    potentials = {i: float(u_norm[i]) for i in range(n)}
    
    if all_coherences:
        global_coherence = float(np.median(all_coherences))
    else:
        global_coherence = 0.5
    
    score_values = list(scores.values())
    if score_values:
        q60 = float(np.percentile(score_values, 60))
        q80 = float(np.percentile(score_values, 80))
    else:
        q60, q80 = 0.5, 0.5
    
    return scores, potentials, global_coherence, q60, q80


def run_ab_test():
    """Run A/B test comparing inversion vs no-inversion."""
    from alba_framework_potential import coherence as coh_module
    
    # Save original function
    original_compute = coh_module.compute_coherence_scores
    
    test_cases = [
        ("Sphere", 5, lambda x: float(np.sum(np.array(x)**2))),
        ("Sphere", 10, lambda x: float(np.sum(np.array(x)**2))),
        ("Rosenbrock", 5, lambda x: sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))),
        ("Ellipsoid", 10, lambda x: float(np.sum([10**(6*i/9) * x[i]**2 for i in range(10)]))),
    ]
    
    results = []
    
    for name, dim, func in test_cases:
        bounds = [(-5.0, 5.0)] * dim
        budget = 200
        n_seeds = 5
        
        # Test with inversion (original)
        coh_module.compute_coherence_scores = original_compute
        vals_inversion = []
        for seed in range(n_seeds):
            opt = ALBA(bounds=bounds, total_budget=budget, use_potential_field=True, seed=seed)
            _, val = opt.optimize(func, budget)
            vals_inversion.append(val)
        
        # Test without inversion (fixed)
        coh_module.compute_coherence_scores = compute_coherence_scores_fixed
        vals_no_inversion = []
        for seed in range(n_seeds):
            opt = ALBA(bounds=bounds, total_budget=budget, use_potential_field=True, seed=seed)
            _, val = opt.optimize(func, budget)
            vals_no_inversion.append(val)
        
        mean_inv = np.mean(vals_inversion)
        mean_no_inv = np.mean(vals_no_inversion)
        
        if mean_no_inv < mean_inv:
            winner = "NoInv"
            delta = (mean_inv - mean_no_inv) / mean_inv * 100
        else:
            winner = "Inv"
            delta = (mean_no_inv - mean_inv) / mean_no_inv * 100
        
        results.append((name, dim, mean_inv, mean_no_inv, winner, delta))
        print(f"{name:12} {dim}D: Inv={mean_inv:10.2f}  NoInv={mean_no_inv:10.2f}  Winner={winner:6} Δ={delta:5.1f}%")
    
    # Restore original
    coh_module.compute_coherence_scores = original_compute
    
    # Summary
    no_inv_wins = sum(1 for r in results if r[4] == "NoInv")
    print(f"\n{'='*60}")
    print(f"SUMMARY: NoInversion wins {no_inv_wins}/{len(results)} tests")
    
    if no_inv_wins > len(results) / 2:
        print("✅ Fix confirmed: removing inversion improves performance!")
    else:
        print("⚠️  Fix not confirmed: inversion may be correct after all")


if __name__ == "__main__":
    run_ab_test()
