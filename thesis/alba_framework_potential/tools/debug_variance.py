#!/usr/bin/env python3
"""
Deep analysis: perché ALBA_Cov ha varianza così alta?

Ipotesi:
1. La covarianza degenera su alcuni seed
2. Il centering su best_x può "saltare" in modo instabile
3. La regolarizzazione non è sufficiente in certi casi
4. Il numero di top-k punti è troppo basso per stimare bene la covarianza
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler

def make_rosenbrock(dim):
    return lambda x: float(np.sum(100.0*(np.array(x)[1:]-np.array(x)[:-1]**2)**2 + (1-np.array(x)[:-1])**2))


def analyze_variance_source():
    """Run multiple seeds and analyze what causes variance"""
    print("=" * 70)
    print("VARIANCE SOURCE ANALYSIS: Rosenbrock 10D")
    print("=" * 70)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_rosenbrock(dim)
    
    all_results = []
    
    for seed in range(10):
        sampler = CovarianceLocalSearchSampler()
        
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=seed
        )
        
        _, val = opt.optimize(func, budget)
        
        all_results.append({
            'seed': seed,
            'value': val,
            'final_dist': np.linalg.norm(opt.best_x),
        })
        
        print(f"Seed {seed:2d}: val={val:8.2f}, dist={np.linalg.norm(opt.best_x):.4f}")
    
    values = [r['value'] for r in all_results]
    print(f"\nSummary: mean={np.mean(values):.2f}, std={np.std(values):.2f}, "
          f"min={np.min(values):.2f}, max={np.max(values):.2f}")
    
    # Find worst case
    worst = max(all_results, key=lambda r: r['value'])
    best = min(all_results, key=lambda r: r['value'])
    
    print(f"\nWorst case (seed {worst['seed']}): val={worst['value']:.2f}")
    print(f"Best case (seed {best['seed']}): val={best['value']:.2f}")
    print(f"Ratio worst/best: {worst['value']/best['value']:.1f}x")


def compare_with_baseline():
    """Compare ALBA_Cov variance vs ALBA base variance"""
    print("\n" + "=" * 70)
    print("VARIANCE COMPARISON: ALBA vs ALBA_Cov")
    print("=" * 70)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_rosenbrock(dim)
    
    alba_results = []
    alba_cov_results = []
    
    for seed in range(20):
        # ALBA base
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            seed=seed
        )
        _, val = opt.optimize(func, budget)
        alba_results.append(val)
        
        # ALBA Cov
        sampler = CovarianceLocalSearchSampler()
        opt_cov = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=seed
        )
        _, val_cov = opt_cov.optimize(func, budget)
        alba_cov_results.append(val_cov)
    
    alba_mean, alba_std = np.mean(alba_results), np.std(alba_results)
    cov_mean, cov_std = np.mean(alba_cov_results), np.std(alba_cov_results)
    
    print(f"ALBA base:     mean={alba_mean:8.2f}, std={alba_std:8.2f}, CV={alba_std/alba_mean*100:.1f}%")
    print(f"ALBA_Cov:      mean={cov_mean:8.2f}, std={cov_std:8.2f}, CV={cov_std/cov_mean*100:.1f}%")
    
    # Count wins
    wins_cov = sum(1 for a, c in zip(alba_results, alba_cov_results) if c < a)
    print(f"\nALBA_Cov wins: {wins_cov}/20 ({wins_cov/20*100:.0f}%)")
    
    # Show per-seed comparison
    print("\nPer-seed comparison:")
    for i, (a, c) in enumerate(zip(alba_results, alba_cov_results)):
        status = "✅" if c < a else "❌"
        print(f"  Seed {i:2d}: ALBA={a:8.2f}, Cov={c:8.2f} | {status} diff={c-a:+8.2f}")


def analyze_worst_case_seed():
    """Deep dive into what goes wrong on bad seeds"""
    print("\n" + "=" * 70)
    print("WORST CASE DEEP DIVE")
    print("=" * 70)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_rosenbrock(dim)
    
    # Seed 6 was the worst in previous run
    for seed in [5, 6]:  # Best and worst from previous
        sampler = CovarianceLocalSearchSampler()
        
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            local_search_ratio=0.3,
            local_search_sampler=sampler,
            seed=seed
        )
        
        # Track convergence
        convergence = []
        for i in range(budget):
            x = opt.ask()
            y = func(x)
            opt.tell(x, y)
            
            if i % 50 == 0 or i == budget - 1:
                convergence.append((i, opt.best_y))
        
        print(f"\nSeed {seed} convergence:")
        for iteration, val in convergence:
            print(f"  iter {iteration:3d}: best_y = {val:.2f}")


if __name__ == "__main__":
    analyze_variance_source()
    compare_with_baseline()
    analyze_worst_case_seed()
