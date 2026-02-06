#!/usr/bin/env python3
"""
A/B Test: Density Anchor vs Best Score Anchor

Se best_score vince ovunque → bug confermato, va fixato
Se risultati misti → serve più indagine
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

# Import coherence module to patch
from alba_framework_potential import coherence as coherence_module
from alba_framework_potential.optimizer import ALBA


# =============================================================================
# PATCHED VERSION: Best Score Anchor
# =============================================================================

def compute_coherence_scores_bestscore(
    leaves,
    categorical_dims=None,
    k_neighbors=6,
):
    """
    Versione modificata che usa best_score invece di density per l'anchoring.
    """
    from alba_framework_potential.coherence import (
        _build_knn_graph,
        _compute_predicted_drops,
        _solve_potential_least_squares,
    )
    
    n = len(leaves)
    
    if n < 3:
        return {i: 0.5 for i in range(n)}, {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    edges = _build_knn_graph(leaves, k=k_neighbors)
    
    if not edges:
        return {i: 0.5 for i in range(n)}, {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    d_lm, alignments, valid_edges = _compute_predicted_drops(
        leaves, edges, categorical_dims
    )
    
    if len(valid_edges) < 2:
        return {i: 0.5 for i in range(n)}, {i: 0.5 for i in range(n)}, 0.5, 0.5, 0.5
    
    # Coherence scores (same as original)
    leaf_alignments = {i: [] for i in range(n)}
    for e, (i, j) in enumerate(valid_edges):
        leaf_alignments[i].append(alignments[e])
        leaf_alignments[j].append(alignments[e])
    
    scores = {}
    all_coherences = []
    for i in range(n):
        if leaf_alignments[i]:
            mean_align = float(np.mean(leaf_alignments[i]))
            coherence = (mean_align + 1.0) / 2.0
        else:
            coherence = 0.5
        scores[i] = coherence
        all_coherences.append(coherence)
    
    # Solve potential
    weights = np.array(alignments) + 1.0
    u = _solve_potential_least_squares(n, valid_edges, d_lm, weights)
    
    # =========================================================================
    # CHANGED: Use best_score instead of density
    # =========================================================================
    leaf_best_scores = np.zeros(n)
    for i in range(n):
        pairs = list(leaves[i].tested_pairs)
        if pairs:
            scores_in_leaf = [s for _, s in pairs]
            leaf_best_scores[i] = max(scores_in_leaf)  # Higher is better (internal)
        else:
            leaf_best_scores[i] = -np.inf
    
    # Normalize to [0, 1]
    valid_mask = np.isfinite(leaf_best_scores)
    if valid_mask.any() and np.std(leaf_best_scores[valid_mask]) > 1e-9:
        bs_min = np.min(leaf_best_scores[valid_mask])
        bs_max = np.max(leaf_best_scores[valid_mask])
        if bs_max > bs_min:
            leaf_scores_norm = (leaf_best_scores - bs_min) / (bs_max - bs_min)
        else:
            leaf_scores_norm = np.full(n, 0.5)
        median_score = float(np.median(leaf_scores_norm[valid_mask]))
        leaf_scores_norm = np.where(valid_mask, leaf_scores_norm, median_score)
    else:
        leaf_scores_norm = np.full(n, 0.5)
    
    leaf_scores_norm = np.clip(leaf_scores_norm, 0.0, 1.0)
    
    # Invert and combine
    u_inverted = -u
    empirical_bonus = leaf_scores_norm * 2.0
    u_combined = u_inverted - empirical_bonus
    
    # Re-anchor on best leaf (highest score = best internal value)
    best_leaf_idx = int(np.argmax(leaf_scores_norm))
    u_anchored = u_combined - u_combined[best_leaf_idx]
    
    # Normalize to [0, 1]
    u_var = np.var(u_anchored) if u_anchored.size > 0 else 0.0
    MIN_POTENTIAL_VARIANCE = 0.001
    
    if u_var < MIN_POTENTIAL_VARIANCE:
        if np.std(leaf_scores_norm) > 0.01:
            u_norm = 1.0 - leaf_scores_norm
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
    
    # Global metrics
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


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def make_sphere(dim):
    return lambda x: float(np.sum(np.array(x)**2))

def make_rosenbrock(dim):
    return lambda x: float(np.sum(100.0*(np.array(x)[1:]-np.array(x)[:-1]**2)**2 + (1-np.array(x)[:-1])**2))

def make_rastrigin(dim):
    return lambda x: float(10 * dim + np.sum(np.array(x)**2 - 10 * np.cos(2 * np.pi * np.array(x))))

def make_ellipsoid(dim):
    weights = np.array([10**(6 * i/(dim-1)) for i in range(dim)])
    return lambda x: float(np.sum(weights * np.array(x)**2))


# =============================================================================
# A/B TEST
# =============================================================================

def run_ab_test():
    """Run A/B test comparing density anchor vs best_score anchor."""
    
    # Save original function
    original_compute = coherence_module.compute_coherence_scores
    
    # Test configurations
    tests = [
        ("Sphere", 5, 150, make_sphere),
        ("Sphere", 10, 300, make_sphere),
        ("Rosenbrock", 5, 150, make_rosenbrock),
        ("Rosenbrock", 10, 300, make_rosenbrock),
        ("Rastrigin", 5, 150, make_rastrigin),
        ("Ellipsoid", 10, 300, make_ellipsoid),
    ]
    
    n_repeats = 5
    results = []
    
    print("=" * 80)
    print("A/B TEST: Density Anchor vs Best Score Anchor")
    print("=" * 80)
    print(f"{'Function':<15} {'Dim':<5} {'Density':<12} {'BestScore':<12} {'Winner':<10} {'Δ%':<8}")
    print("-" * 80)
    
    for func_name, dim, budget, func_maker in tests:
        bounds = [(-5.0, 5.0)] * dim
        func = func_maker(dim)
        
        density_results = []
        bestscore_results = []
        
        for seed in range(n_repeats):
            # --- Test A: Density Anchor (original) ---
            coherence_module.compute_coherence_scores = original_compute
            
            opt_density = ALBA(
                bounds=bounds,
                total_budget=budget,
                use_potential_field=True,
                use_coherence_gating=True,
                seed=seed
            )
            _, val_density = opt_density.optimize(func, budget)
            density_results.append(val_density)
            
            # --- Test B: Best Score Anchor (proposed) ---
            coherence_module.compute_coherence_scores = compute_coherence_scores_bestscore
            
            opt_bestscore = ALBA(
                bounds=bounds,
                total_budget=budget,
                use_potential_field=True,
                use_coherence_gating=True,
                seed=seed
            )
            _, val_bestscore = opt_bestscore.optimize(func, budget)
            bestscore_results.append(val_bestscore)
        
        # Restore original
        coherence_module.compute_coherence_scores = original_compute
        
        mean_density = np.mean(density_results)
        mean_bestscore = np.mean(bestscore_results)
        
        # Lower is better (minimization)
        if mean_bestscore < mean_density:
            winner = "BestScore"
            delta = (mean_density - mean_bestscore) / mean_density * 100
        else:
            winner = "Density"
            delta = (mean_bestscore - mean_density) / mean_bestscore * 100
        
        # Count per-seed wins
        bs_wins = sum(1 for d, b in zip(density_results, bestscore_results) if b < d)
        
        print(f"{func_name:<15} {dim:<5} {mean_density:<12.2f} {mean_bestscore:<12.2f} "
              f"{winner:<10} {delta:+.1f}% ({bs_wins}/{n_repeats})")
        
        results.append({
            "func": func_name,
            "dim": dim,
            "density": mean_density,
            "bestscore": mean_bestscore,
            "winner": winner,
            "delta": delta,
            "bs_wins": bs_wins,
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    bs_total_wins = sum(1 for r in results if r["winner"] == "BestScore")
    density_total_wins = len(results) - bs_total_wins
    
    print(f"BestScore wins: {bs_total_wins}/{len(results)}")
    print(f"Density wins:   {density_total_wins}/{len(results)}")
    
    if bs_total_wins == len(results):
        print("\n🚨 BestScore wins EVERYWHERE → This is a BUG! Fix recommended.")
    elif bs_total_wins > density_total_wins:
        print(f"\n⚠️  BestScore wins {bs_total_wins}/{len(results)} → Likely a bug, investigate further.")
    elif bs_total_wins == density_total_wins:
        print("\n🔄 Mixed results → Need deeper investigation.")
    else:
        print("\n✅ Density wins more → Current implementation may be correct.")
    
    return results


if __name__ == "__main__":
    run_ab_test()
