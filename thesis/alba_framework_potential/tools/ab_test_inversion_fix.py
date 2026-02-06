#!/usr/bin/env python3
"""
A/B Test: Fix inversione del potenziale

BUG IDENTIFICATO:
- Linea 410-411 di coherence.py: u_inverted = -u
- Questo INVERTE la relazione tra potenziale e distanza dall'ottimo
- Il potenziale grezzo u è CORRETTO (alto lontano dal minimo)
- L'inversione lo rovina

FIX:
- Rimuovere u_inverted = -u
- Usare u direttamente

Questo test confronta le due versioni.
"""

import sys
sys.path.insert(0, '/mnt/workspace')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from copy import deepcopy

# Salva la funzione originale
import alba_framework_potential.coherence as coh_module
original_compute_coherence_scores = coh_module.compute_coherence_scores


def create_fixed_compute_coherence_scores():
    """Crea una versione corretta di compute_coherence_scores senza inversione."""
    
    from typing import Dict, List, Optional, Tuple
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    
    def compute_coherence_scores_fixed(
        leaves,
        categorical_dims = None,
        k_neighbors: int = 6,
    ):
        """FIXED: Same as original but WITHOUT sign inversion."""
        
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
        
        edges = _build_knn_graph(leaves, k_neighbors)
        
        if not edges:
            scores = {i: 0.5 for i in range(n)}
            potentials = {i: 0.5 for i in range(n)}
            return scores, potentials, 0.5, 0.5, 0.5
        
        d_lm, alignment, valid_edges = _compute_predicted_drops(leaves, edges, categorical_dims)
        
        if len(valid_edges) == 0:
            scores = {i: 0.5 for i in range(n)}
            potentials = {i: 0.5 for i in range(n)}
            return scores, potentials, 0.5, 0.5, 0.5
        
        # Coherence scores
        scores = {}
        all_coherences = []
        
        leaf_alignments = {i: [] for i in range(n)}
        for e, (i, j) in enumerate(valid_edges):
            leaf_alignments[i].append(alignment[e])
            leaf_alignments[j].append(alignment[e])
        
        for i in range(n):
            if leaf_alignments[i]:
                mean_align = float(np.mean(leaf_alignments[i]))
                coherence = (mean_align + 1.0) / 2.0
            else:
                coherence = 0.5
            scores[i] = coherence
            all_coherences.append(coherence)
        
        # Solve for potential
        weights = np.clip(alignment, 0.1, 1.0)
        u = _solve_potential_least_squares(n, valid_edges, d_lm, weights)
        
        # Density calculation
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
        # Original had: u_inverted = -u
        # We use u directly because:
        # - u is already LOWER near the minimum (correct for minimization)
        # - Inverting makes u HIGHER near the minimum (wrong!)
        
        # Higher density = lower potential (better region)
        empirical_bonus = leaf_densities_norm * 2.0
        u_combined = u - empirical_bonus  # NO INVERSION!
        
        # Re-anchor: find leaf with LOWEST u_combined (best) and set to 0
        best_leaf_idx = int(np.argmin(u_combined))  # CHANGED from argmax(density) to argmin(u)
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
    
    return compute_coherence_scores_fixed


def run_ab_test():
    """Run A/B test comparing original (with inversion) vs fixed (no inversion)."""
    from alba_framework_potential.optimizer import ALBA
    
    fixed_func = create_fixed_compute_coherence_scores()
    
    test_cases = [
        ("Sphere", 5, lambda x: float(np.sum(np.array(x)**2))),
        ("Sphere", 10, lambda x: float(np.sum(np.array(x)**2))),
        ("Rosenbrock", 5, lambda x: sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))),
        ("Rosenbrock", 10, lambda x: sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))),
        ("Rastrigin", 5, lambda x: 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)),
    ]
    
    print("="*80)
    print("A/B TEST: Potential Inversion Fix")
    print("="*80)
    print(f"{'Function':<15} {'Dim':>4} | {'Original':>12} | {'Fixed':>12} | {'Winner':>10} | {'Δ%':>7}")
    print("-"*80)
    
    results = []
    
    for name, dim, func in test_cases:
        bounds = [(-5.0, 5.0)] * dim
        budget = 200
        n_seeds = 8
        
        # Test with original (inversion)
        coh_module.compute_coherence_scores = original_compute_coherence_scores
        vals_original = []
        for seed in range(n_seeds):
            opt = ALBA(bounds=bounds, total_budget=budget, use_potential_field=True, seed=seed)
            _, val = opt.optimize(func, budget)
            vals_original.append(val)
        
        # Test with fix (no inversion)
        coh_module.compute_coherence_scores = fixed_func
        vals_fixed = []
        for seed in range(n_seeds):
            opt = ALBA(bounds=bounds, total_budget=budget, use_potential_field=True, seed=seed)
            _, val = opt.optimize(func, budget)
            vals_fixed.append(val)
        
        mean_orig = np.mean(vals_original)
        mean_fixed = np.mean(vals_fixed)
        
        if mean_fixed < mean_orig:
            winner = "Fixed"
            delta = (mean_orig - mean_fixed) / mean_orig * 100
        else:
            winner = "Original"
            delta = (mean_fixed - mean_orig) / mean_fixed * 100
        
        # Count wins
        fixed_wins = sum(1 for a, b in zip(vals_original, vals_fixed) if b < a)
        
        results.append((name, dim, mean_orig, mean_fixed, winner, delta, fixed_wins))
        print(f"{name:<15} {dim:>4} | {mean_orig:12.2f} | {mean_fixed:12.2f} | {winner:>10} | {delta:>6.1f}% ({fixed_wins}/{n_seeds})")
    
    # Restore original
    coh_module.compute_coherence_scores = original_compute_coherence_scores
    
    # Summary
    fixed_wins_total = sum(1 for r in results if r[4] == "Fixed")
    print("="*80)
    print(f"SUMMARY: Fixed version wins {fixed_wins_total}/{len(results)} test cases")
    
    if fixed_wins_total > len(results) / 2:
        print("✅ FIX VALIDATED: Removing inversion improves performance!")
    else:
        print("⚠️ FIX NOT VALIDATED: Original might be correct after all")


if __name__ == "__main__":
    run_ab_test()
