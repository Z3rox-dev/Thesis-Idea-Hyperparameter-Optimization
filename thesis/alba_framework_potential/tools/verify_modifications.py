"""
Verifica che le modifiche introdotte non abbiano rotto nulla:
1. ThompsonSamplingLeafSelector come default
2. Density-based potential in coherence.py

Confrontiamo:
- ALBA attuale (Thompson + Density)
- ALBA vecchio (UCBSoftmax + good_ratio) - simulato
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector, ThompsonSamplingLeafSelector


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def sphere(x): return np.sum(x**2)
def rosenbrock(x): return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
def rastrigin(x): return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
def ackley(x):
    n = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e
def levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0])**2 + np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2)) + (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
def griewank(x): return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark():
    print("=" * 70)
    print("VERIFICA: ALBA con modifiche vs ALBA legacy")
    print("=" * 70)
    print("\nModifiche testate:")
    print("  1. ThompsonSamplingLeafSelector (default)")
    print("  2. Density-based potential in coherence.py")
    print()
    
    functions = [
        ("Sphere-5D", sphere, 5, [(-5, 5)] * 5),
        ("Sphere-10D", sphere, 10, [(-5, 5)] * 10),
        ("Rosenbrock-3D", rosenbrock, 3, [(-2, 2)] * 3),
        ("Rosenbrock-5D", rosenbrock, 5, [(-2, 2)] * 5),
        ("Rastrigin-5D", rastrigin, 5, [(-5.12, 5.12)] * 5),
        ("Ackley-5D", ackley, 5, [(-5, 5)] * 5),
        ("Levy-5D", levy, 5, [(-10, 10)] * 5),
        ("Griewank-5D", griewank, 5, [(-10, 10)] * 5),
    ]
    
    seeds = [42, 123, 456, 789, 1011, 2022]
    budget = 100
    
    results_new = {}
    results_legacy = {}
    wins = {'new': 0, 'legacy': 0, 'tie': 0}
    
    for fname, func, dim, bounds in functions:
        print(f"{fname}:", end=" ", flush=True)
        results_new[fname] = []
        results_legacy[fname] = []
        
        for seed in seeds:
            # NEW: Thompson + Density (current default)
            opt_new = ALBA(
                bounds=bounds, 
                seed=seed, 
                total_budget=budget,
                use_potential_field=True  # Uses density now
            )
            for _ in range(budget):
                x = opt_new.ask()
                y = func(x)
                opt_new.tell(x, -y)
            results_new[fname].append(-opt_new.best_y)
            
            # LEGACY: UCBSoftmax + no potential (simulates old behavior)
            opt_legacy = ALBA(
                bounds=bounds,
                seed=seed,
                total_budget=budget,
                use_potential_field=False,
                leaf_selector=UCBSoftmaxLeafSelector()  # Old default
            )
            for _ in range(budget):
                x = opt_legacy.ask()
                y = func(x)
                opt_legacy.tell(x, -y)
            results_legacy[fname].append(-opt_legacy.best_y)
        
        new_mean = np.mean(results_new[fname])
        legacy_mean = np.mean(results_legacy[fname])
        
        # Determine winner
        improvement = (legacy_mean - new_mean) / abs(legacy_mean + 1e-9) * 100
        
        if abs(improvement) < 1:
            winner = "TIE"
            wins['tie'] += 1
        elif new_mean < legacy_mean:
            winner = "NEW ✓"
            wins['new'] += 1
        else:
            winner = "LEGACY"
            wins['legacy'] += 1
        
        print(f"New={new_mean:.4f}, Legacy={legacy_mean:.4f}, Δ={improvement:+.1f}% → {winner}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"NEW (Thompson+Density) wins:  {wins['new']}")
    print(f"LEGACY (UCBSoftmax) wins:     {wins['legacy']}")
    print(f"Ties:                         {wins['tie']}")
    
    total = wins['new'] + wins['legacy'] + wins['tie']
    
    if wins['new'] >= wins['legacy']:
        print(f"\n✓ MODIFICHE OK! New vince o pareggia {wins['new'] + wins['tie']}/{total}")
    else:
        print(f"\n⚠ ATTENZIONE: Legacy vince {wins['legacy']}/{total} - verificare!")
    
    # Detailed comparison
    print("\n" + "-" * 70)
    print("Dettaglio per funzione:")
    print(f"{'Function':<20} {'New':<12} {'Legacy':<12} {'Δ%':<10} {'Winner':<10}")
    print("-" * 70)
    
    for fname in results_new:
        new_mean = np.mean(results_new[fname])
        legacy_mean = np.mean(results_legacy[fname])
        improvement = (legacy_mean - new_mean) / abs(legacy_mean + 1e-9) * 100
        
        if abs(improvement) < 1:
            winner = "TIE"
        elif new_mean < legacy_mean:
            winner = "NEW ✓"
        else:
            winner = "LEGACY"
        
        print(f"{fname:<20} {new_mean:<12.4f} {legacy_mean:<12.4f} {improvement:+8.1f}%  {winner}")
    
    return results_new, results_legacy, wins


if __name__ == "__main__":
    results_new, results_legacy, wins = run_benchmark()
