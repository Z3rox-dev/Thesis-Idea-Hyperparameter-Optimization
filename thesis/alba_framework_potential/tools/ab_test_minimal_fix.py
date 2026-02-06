#!/usr/bin/env python3
"""
A/B Test MINIMAL: Solo rimozione dell'inversione, nient'altro.

Questo test cambia SOLO la linea:
  u_inverted = -u  →  u_inverted = u  (senza inversione)

Nessun altro cambiamento per isolare l'effetto.
"""

import sys
sys.path.insert(0, '/mnt/workspace')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np

# Salva la funzione originale
import alba_framework_potential.coherence as coh_module
original_compute_coherence_scores = coh_module.compute_coherence_scores


def create_minimal_fix():
    """Crea una versione con fix MINIMALE: solo rimozione inversione."""
    
    def compute_coherence_scores_minimal_fix(
        leaves,
        categorical_dims = None,
        k_neighbors: int = 6,
    ):
        """MINIMAL FIX: Same as original but u_inverted = u instead of -u."""
        
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
        
        # Density calculation (UNCHANGED from original)
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
        
        # MINIMAL FIX: Use u directly instead of -u
        # Original: u_inverted = -u
        # Fixed: u_inverted = u (no inversion)
        u_inverted = u  # <-- THE ONLY CHANGE!
        
        # Rest is UNCHANGED from original
        empirical_bonus = leaf_densities_norm * 2.0
        u_combined = u_inverted - empirical_bonus
        
        best_leaf_idx = int(np.argmax(leaf_densities_norm))  # UNCHANGED
        u_anchored = u_combined - u_combined[best_leaf_idx]
        
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
    
    return compute_coherence_scores_minimal_fix


def run_ab_test():
    """Run A/B test: original vs minimal fix."""
    from alba_framework_potential.optimizer import ALBA
    
    minimal_fix = create_minimal_fix()
    
    test_cases = [
        ("Sphere", 5, lambda x: float(np.sum(np.array(x)**2))),
        ("Sphere", 10, lambda x: float(np.sum(np.array(x)**2))),
        ("Rosenbrock", 5, lambda x: sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))),
        ("Rosenbrock", 10, lambda x: sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))),
    ]
    
    print("="*80)
    print("A/B TEST: MINIMAL Fix (only remove inversion)")
    print("="*80)
    print(f"{'Function':<15} {'Dim':>4} | {'Original':>12} | {'MinFix':>12} | {'Winner':>10} | Wins")
    print("-"*80)
    
    results = []
    
    for name, dim, func in test_cases:
        bounds = [(-5.0, 5.0)] * dim
        budget = 200
        n_seeds = 10
        
        # Test with original
        coh_module.compute_coherence_scores = original_compute_coherence_scores
        vals_original = []
        for seed in range(n_seeds):
            opt = ALBA(bounds=bounds, total_budget=budget, use_potential_field=True, seed=seed)
            _, val = opt.optimize(func, budget)
            vals_original.append(val)
        
        # Test with minimal fix
        coh_module.compute_coherence_scores = minimal_fix
        vals_fixed = []
        for seed in range(n_seeds):
            opt = ALBA(bounds=bounds, total_budget=budget, use_potential_field=True, seed=seed)
            _, val = opt.optimize(func, budget)
            vals_fixed.append(val)
        
        mean_orig = np.mean(vals_original)
        mean_fixed = np.mean(vals_fixed)
        
        fixed_wins = sum(1 for a, b in zip(vals_original, vals_fixed) if b < a)
        
        if mean_fixed < mean_orig:
            winner = "MinFix"
        else:
            winner = "Original"
        
        results.append((name, dim, mean_orig, mean_fixed, winner, fixed_wins))
        print(f"{name:<15} {dim:>4} | {mean_orig:12.2f} | {mean_fixed:12.2f} | {winner:>10} | {fixed_wins}/{n_seeds}")
    
    # Restore original
    coh_module.compute_coherence_scores = original_compute_coherence_scores
    
    # Summary
    minfix_wins = sum(1 for r in results if r[4] == "MinFix")
    total_seed_wins = sum(r[5] for r in results)
    total_seeds = len(results) * 10
    
    print("="*80)
    print(f"SUMMARY: MinFix wins {minfix_wins}/{len(results)} test cases")
    print(f"         MinFix wins {total_seed_wins}/{total_seeds} individual seeds")
    
    if total_seed_wins > total_seeds / 2:
        print("✅ MINIMAL FIX VALIDATED!")
    else:
        print("⚠️ MINIMAL FIX NOT VALIDATED")


if __name__ == "__main__":
    run_ab_test()
