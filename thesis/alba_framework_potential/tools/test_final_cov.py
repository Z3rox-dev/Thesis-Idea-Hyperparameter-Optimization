#!/usr/bin/env python3
"""
Test finale: CovarianceLocalSearchSampler migliorato vs Gaussian.

Verifica che le modifiche applicate funzionino correttamente.
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import GaussianLocalSearchSampler, CovarianceLocalSearchSampler


def sphere(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(x**2))

def rosenbrock(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def rastrigin(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))


def run_test(func, dim, sampler, seed, budget=100):
    bounds = [(-5.0, 10.0)] * dim
    opt = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
               local_search_sampler=sampler, use_drilling=False, seed=seed)
    for _ in range(budget):
        x = opt.ask()
        opt.tell(x, func(x))
    return opt.best_y


def main():
    n_seeds = 30
    
    print("="*80)
    print("TEST FINALE: CovarianceLocalSearchSampler migliorato")
    print("="*80)
    print()
    print("Modifiche applicate:")
    print("  1. Adaptive top_k_fraction: base=0.15 + 0.02*dim (capped at 0.50)")
    print("  2. Dimension-proportional regularization: eps = 0.01 * (1 + 0.1*dim)")
    print("  3. Scale multiplier: 5.0 (was 3.0)")
    print("  4. Center on best_x (not mu_w)")
    print()
    
    for func, func_name in [(rosenbrock, "Rosenbrock"), (sphere, "Sphere"), (rastrigin, "Rastrigin")]:
        print(f"\n{'='*60}")
        print(f"Funzione: {func_name}")
        print(f"{'='*60}")
        
        for dim in [3, 5, 10]:
            gauss_results = []
            cov_results = []
            
            for seed in range(n_seeds):
                gauss_y = run_test(func, dim, GaussianLocalSearchSampler(), seed)
                cov_y = run_test(func, dim, CovarianceLocalSearchSampler(), seed)
                gauss_results.append(gauss_y)
                cov_results.append(cov_y)
            
            gauss_results = np.array(gauss_results)
            cov_results = np.array(cov_results)
            
            # Statistiche
            gauss_mean = np.mean(gauss_results)
            cov_mean = np.mean(cov_results)
            gauss_median = np.median(gauss_results)
            cov_median = np.median(cov_results)
            
            # Conteggio vittorie
            cov_wins = np.sum(cov_results < gauss_results)
            gauss_wins = np.sum(gauss_results < cov_results)
            ties = n_seeds - cov_wins - gauss_wins
            
            # Miglioramento medio
            improvement = (gauss_mean - cov_mean) / gauss_mean * 100
            
            print(f"\n--- Dim = {dim} ---")
            print(f"  Gaussian: mean={gauss_mean:10.1f}  median={gauss_median:10.1f}  std={np.std(gauss_results):8.1f}")
            print(f"  Cov:      mean={cov_mean:10.1f}  median={cov_median:10.1f}  std={np.std(cov_results):8.1f}")
            print(f"  Head-to-head: Cov wins {cov_wins}/{n_seeds}, Gauss wins {gauss_wins}/{n_seeds}, ties {ties}")
            print(f"  Improvement: {improvement:+.1f}%")
            
            if improvement > 0:
                winner = "Cov"
                emoji = "✓"
            else:
                winner = "Gauss"
                emoji = "✗"
            print(f"  Winner: {winner} {emoji}")
    
    print("\n" + "="*80)
    print("RIEPILOGO")
    print("="*80)
    print()
    print("Il CovarianceLocalSearchSampler migliorato dovrebbe:")
    print("  - Vincere su funzioni con struttura (Rosenbrock)")
    print("  - Pareggiare/perdere su funzioni isotrope (Sphere)")
    print("  - Funzionare meglio in bassa dimensione")
    print("  - Non peggiorare drasticamente in alta dimensione")


if __name__ == "__main__":
    main()
