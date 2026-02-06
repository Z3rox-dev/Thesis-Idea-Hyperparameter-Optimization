#!/usr/bin/env python3
"""
Test dettagliato: analisi run-by-run per capire la distribuzione.
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from test_cov_fix_comparison import (
    rosenbrock_with_categorical, 
    run_benchmark,
    CovarianceLocalSearchSampler,
    CovarianceContinuousOnlySampler,
    ALBA
)

def run_detailed(func, dim, budget, n_seeds, cat_dims):
    """Ritorna risultati dettagliati per ogni seed."""
    results_all = []
    results_cont = []
    
    for seed in range(n_seeds):
        # ALL dims
        sampler = CovarianceLocalSearchSampler()
        opt = ALBA(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            maximize=True,
            seed=100 + seed,
            total_budget=budget,
            categorical_dims=cat_dims,
            use_potential_field=False,
        )
        opt._local_search_sampler = sampler
        for _ in range(budget):
            x = opt.ask()
            opt.tell(x, func(x))
        results_all.append(opt.best_y_internal)
        
        # CONT only
        sampler = CovarianceContinuousOnlySampler(categorical_dims=cat_dims)
        opt = ALBA(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            maximize=True,
            seed=100 + seed,
            total_budget=budget,
            categorical_dims=cat_dims,
            use_potential_field=False,
        )
        opt._local_search_sampler = sampler
        for _ in range(budget):
            x = opt.ask()
            opt.tell(x, func(x))
        results_cont.append(opt.best_y_internal)
    
    return results_all, results_cont


def main():
    print("=" * 70)
    print("ANALISI DETTAGLIATA: Rosen+Cat (4C+2K)")
    print("=" * 70)
    
    cat_dims = [(4, 4), (5, 3)]
    n_seeds = 20
    
    results_all, results_cont = run_detailed(
        rosenbrock_with_categorical, 6, 250, n_seeds, cat_dims
    )
    
    print(f"\n{'Seed':<6} {'ALL dims':>12} {'CONT only':>12} {'Delta':>10} {'Winner':>10}")
    print("-" * 52)
    
    for i in range(n_seeds):
        delta = results_cont[i] - results_all[i]
        winner = "CONT" if delta > 0 else ("ALL" if delta < 0 else "TIE")
        print(f"{i:<6} {results_all[i]:>12.4f} {results_cont[i]:>12.4f} {delta:>+10.4f} {winner:>10}")
    
    print("-" * 52)
    print(f"{'MEDIA':<6} {np.mean(results_all):>12.4f} {np.mean(results_cont):>12.4f}")
    print(f"{'MEDIAN':<6} {np.median(results_all):>12.4f} {np.median(results_cont):>12.4f}")
    print(f"{'STD':<6} {np.std(results_all):>12.4f} {np.std(results_cont):>12.4f}")
    
    wins_cont = sum(1 for a, b in zip(results_cont, results_all) if a > b)
    wins_all = sum(1 for a, b in zip(results_cont, results_all) if a < b)
    
    print(f"\nWins CONT: {wins_cont}/20")
    print(f"Wins ALL:  {wins_all}/20")
    
    # Calcola margine medio quando vince
    cont_margins = [results_cont[i] - results_all[i] 
                   for i in range(n_seeds) if results_cont[i] > results_all[i]]
    all_margins = [results_all[i] - results_cont[i] 
                  for i in range(n_seeds) if results_all[i] > results_cont[i]]
    
    if cont_margins:
        print(f"\nQuando CONT vince, margine medio: {np.mean(cont_margins):+.4f}")
    if all_margins:
        print(f"Quando ALL vince, margine medio:  {np.mean(all_margins):+.4f}")


if __name__ == "__main__":
    main()
