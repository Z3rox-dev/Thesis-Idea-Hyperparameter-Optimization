#!/usr/bin/env python3
"""
Debug: ALBA_Cov vs ALBA_Gaussian su Rosenbrock 3D - Multi-seed analysis
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

np.set_printoptions(precision=4, suppress=True)

def rosenbrock(x):
    """Rosenbrock 3D: ottimo a [1,1,1], valore 0"""
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


def run_comparison(seed):
    """Esegue una run con entrambi i sampler."""
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    # Gaussian
    opt_gauss = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=GaussianLocalSearchSampler(),
        use_drilling=False,
        seed=seed
    )
    
    for _ in range(budget):
        x = opt_gauss.ask()
        y = rosenbrock(x)
        opt_gauss.tell(x, y)
    
    # Cov
    opt_cov = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=CovarianceLocalSearchSampler(),
        use_drilling=False,
        seed=seed
    )
    
    for _ in range(budget):
        x = opt_cov.ask()
        y = rosenbrock(x)
        opt_cov.tell(x, y)
    
    return opt_gauss.best_y, opt_cov.best_y


def main():
    print("="*70)
    print("MULTI-SEED COMPARISON: ALBA_Cov vs ALBA_Gaussian su Rosenbrock 3D")
    print("="*70)
    
    n_seeds = 30
    results = []
    
    for seed in range(n_seeds):
        gauss_y, cov_y = run_comparison(seed)
        winner = "Cov" if cov_y < gauss_y else ("Gauss" if gauss_y < cov_y else "Tie")
        results.append({
            'seed': seed,
            'gauss': gauss_y,
            'cov': cov_y,
            'diff': cov_y - gauss_y,
            'winner': winner
        })
        
        marker = "← " + winner if winner != "Tie" else ""
        print(f"Seed {seed:2d}: Gauss={gauss_y:10.2f}, Cov={cov_y:10.2f}, diff={cov_y-gauss_y:+10.2f} {marker}")
    
    # Statistiche
    gauss_wins = sum(1 for r in results if r['winner'] == 'Gauss')
    cov_wins = sum(1 for r in results if r['winner'] == 'Cov')
    ties = sum(1 for r in results if r['winner'] == 'Tie')
    
    gauss_mean = np.mean([r['gauss'] for r in results])
    cov_mean = np.mean([r['cov'] for r in results])
    gauss_std = np.std([r['gauss'] for r in results])
    cov_std = np.std([r['cov'] for r in results])
    
    print("\n" + "="*70)
    print("STATISTICHE (30 seed)")
    print("="*70)
    print(f"Vittorie Gaussian: {gauss_wins}")
    print(f"Vittorie Cov:      {cov_wins}")
    print(f"Pareggi:           {ties}")
    print()
    print(f"Gaussian: mean={gauss_mean:.2f} ± {gauss_std:.2f}")
    print(f"Cov:      mean={cov_mean:.2f} ± {cov_std:.2f}")
    print()
    
    if cov_mean < gauss_mean:
        print(f"✓ Cov è mediamente MIGLIORE di {gauss_mean - cov_mean:.2f}")
    else:
        print(f"⚠️ Gaussian è mediamente MIGLIORE di {cov_mean - gauss_mean:.2f}")
    
    # Analisi dei casi in cui Cov perde molto
    print("\n" + "-"*70)
    print("CASI IN CUI COV PERDE MOLTO (diff > 50):")
    print("-"*70)
    
    bad_cases = [r for r in results if r['diff'] > 50]
    for r in bad_cases:
        print(f"  Seed {r['seed']}: Gauss={r['gauss']:.2f}, Cov={r['cov']:.2f}, diff={r['diff']:+.2f}")
    
    if not bad_cases:
        print("  Nessun caso!")
    
    # Analisi dei casi in cui Cov vince molto
    print("\n" + "-"*70)
    print("CASI IN CUI COV VINCE MOLTO (diff < -50):")
    print("-"*70)
    
    good_cases = [r for r in results if r['diff'] < -50]
    for r in good_cases:
        print(f"  Seed {r['seed']}: Gauss={r['gauss']:.2f}, Cov={r['cov']:.2f}, diff={r['diff']:+.2f}")
    
    if not good_cases:
        print("  Nessun caso!")


if __name__ == "__main__":
    main()
