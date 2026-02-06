#!/usr/bin/env python3
"""
Test A/B: ALBA base vs TPE-inspired Gamma

Testa 2 varianti:
1. ALBA base (gamma quantile fisso 20%)
2. ALBA + Gamma TPE-style (sqrt scaling, più selettivo con più dati)

Su funzioni continue e miste.
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.gamma import QuantileAnnealedGammaScheduler


# ============================================================================
# TPE-INSPIRED GAMMA SCHEDULERS
# ============================================================================

@dataclass(frozen=True)
class TPEStyleGammaScheduler:
    """
    Gamma scheduler ispirato a Optuna TPE.
    
    n_good = ceil(0.25 * sqrt(n_trials))
    
    Diventa più selettivo con più dati.
    """
    min_good: int = 3  # Almeno 3 punti buoni
    max_good: int = 25  # Cap come Optuna
    
    def compute(
        self,
        y_all,
        iteration: int,
        exploration_budget: int,
    ) -> float:
        y_arr = np.asarray(y_all, dtype=float)
        finite_mask = np.isfinite(y_arr)
        y_finite = y_arr[finite_mask]
        n = len(y_finite)
        
        if n < 10:
            return 0.0
        
        # TPE formula: n_good = ceil(0.25 * sqrt(n))
        n_good = int(np.ceil(0.25 * np.sqrt(n)))
        n_good = max(self.min_good, min(n_good, self.max_good))
        n_good = min(n_good, n - 1)  # Non può superare n-1
        
        # Gamma = valore del n_good-esimo migliore (ALBA massimizza)
        y_sorted = np.sort(y_finite)[::-1]  # Decrescente (migliori prima)
        
        # Gamma è il valore sotto cui stare per essere nei top n_good
        gamma = y_sorted[n_good - 1] if n_good <= len(y_sorted) else y_sorted[-1]
        
        return float(gamma)


@dataclass(frozen=True)
class HybridGammaScheduler:
    """
    Ibrido: inizia con quantile fisso (esplorazione), poi passa a TPE-style (exploitation).
    
    Prima metà: 20% quantile (come ALBA base)
    Seconda metà: sqrt scaling (come TPE)
    """
    gamma_quantile: float = 0.20
    min_good: int = 3
    max_good: int = 25
    switch_ratio: float = 0.5  # Quando passare a TPE-style
    
    def compute(
        self,
        y_all,
        iteration: int,
        exploration_budget: int,
    ) -> float:
        y_arr = np.asarray(y_all, dtype=float)
        finite_mask = np.isfinite(y_arr)
        y_finite = y_arr[finite_mask]
        n = len(y_finite)
        
        if n < 10:
            return 0.0
        
        progress = iteration / max(1, exploration_budget)
        
        if progress < self.switch_ratio:
            # Prima metà: quantile fisso (esplorazione)
            return float(np.percentile(y_finite, 100 * (1 - self.gamma_quantile)))
        else:
            # Seconda metà: TPE sqrt (exploitation)
            n_good = int(np.ceil(0.25 * np.sqrt(n)))
            n_good = max(self.min_good, min(n_good, self.max_good))
            n_good = min(n_good, n - 1)
            
            y_sorted = np.sort(y_finite)[::-1]
            gamma = y_sorted[n_good - 1] if n_good <= len(y_sorted) else y_sorted[-1]
            return float(gamma)


# ============================================================================
# TEST FUNCTIONS (MINIMIZATION)
# ============================================================================

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = sum((w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2) for i in range(len(x) - 1))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def mixed_categorical(x):
    """2 categoriali + 3 continue."""
    cats = x[:2].astype(int)
    conts = x[2:]
    # Penalità: categoria ottimale è [0, 1]
    cat_penalty = 5.0 * (abs(cats[0] - 0) + abs(cats[1] - 1))
    cont_value = sum((c - 0.3)**2 for c in conts)
    return cat_penalty + cont_value


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

def run_experiment(func, bounds, gamma_scheduler, n_trials, seed, categorical_dims=None):
    """Run optimization and return best found (for minimization)."""
    
    # ALBA massimizza internamente, quindi passiamo -func
    def objective(x):
        return -func(x)  # Nega per massimizzazione interna
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        gamma_scheduler=gamma_scheduler,
        total_budget=n_trials,
        categorical_dims=categorical_dims,
        maximize=True,  # Vogliamo massimizzare -f(x) = minimizzare f(x)
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        y = objective(x)
        opt.tell(x, y)
    
    # best_y è il massimo di -f(x), quindi -best_y = minimo di f(x)
    return -opt.best_y


def main():
    print("="*70)
    print("  TEST A/B: ALBA base vs TPE-inspired Gamma")
    print("="*70)
    
    N_TRIALS = 100
    N_SEEDS = 20  # 20 seed per robustezza statistica
    DIM = 5
    
    bounds_std = [(-5.0, 5.0)] * DIM
    
    # Test cases
    test_cases = [
        ("Sphere", sphere, bounds_std, None),
        ("Rosenbrock", rosenbrock, bounds_std, None),
        ("Rastrigin", rastrigin, bounds_std, None),
        ("Ackley", ackley, bounds_std, None),
        ("Levy", levy, bounds_std, None),
        ("Mixed_Cat", mixed_categorical, [(0, 3), (0, 4), (-2, 2), (-2, 2), (-2, 2)], [(0, 4), (1, 5)]),
    ]
    
    # Schedulers
    schedulers = {
        "ALBA_base": QuantileAnnealedGammaScheduler(gamma_quantile=0.20, gamma_quantile_start=0.15),
        "TPE_sqrt": TPEStyleGammaScheduler(min_good=3, max_good=25),
        "Hybrid": HybridGammaScheduler(gamma_quantile=0.20, switch_ratio=0.5),
    }
    
    all_winners = {}
    
    for func_name, func, bounds, cat_dims in test_cases:
        print(f"\n{'='*60}")
        print(f"  {func_name} (dim={len(bounds)})")
        print(f"{'='*60}")
        
        results = {name: [] for name in schedulers}
        
        for seed in range(N_SEEDS):
            for name, scheduler in schedulers.items():
                try:
                    val = run_experiment(func, bounds, scheduler, N_TRIALS, seed, cat_dims)
                    results[name].append(val)
                except Exception as e:
                    print(f"  Error {name} seed {seed}: {e}")
                    results[name].append(float('inf'))
        
        # Print results
        print(f"\n{'Method':<15} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 55)
        
        means = {}
        all_vals = {}
        for name, vals in results.items():
            vals = [v for v in vals if np.isfinite(v)]
            all_vals[name] = vals
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                best_val = np.min(vals)
                means[name] = mean_val
                print(f"{name:<15} | {mean_val:>10.4f} | {std_val:>10.4f} | {best_val:>10.4f}")
            else:
                means[name] = float('inf')
                print(f"{name:<15} | {'N/A':>10}")
        
        # Statistical test (Mann-Whitney U)
        from scipy import stats
        if len(all_vals.get("ALBA_base", [])) >= 3 and len(all_vals.get("TPE_sqrt", [])) >= 3:
            stat, pvalue = stats.mannwhitneyu(
                all_vals["ALBA_base"], all_vals["TPE_sqrt"], alternative='two-sided'
            )
            print(f"\n  Mann-Whitney U p-value: {pvalue:.4f}", end="")
            if pvalue < 0.05:
                print(" *")
            else:
                print(" (n.s.)")
        
        winner = min(means, key=means.get)
        all_winners[func_name] = winner
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {"ALBA_base": 0, "TPE_sqrt": 0, "Hybrid": 0}
    for fn, w in all_winners.items():
        print(f"  {fn:<20}: {w}")
        counts[w] += 1
    
    print("\n" + "-"*40)
    for name, c in counts.items():
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get)
    print(f"\n  🏆 Overall Winner: {overall}")


if __name__ == "__main__":
    main()
