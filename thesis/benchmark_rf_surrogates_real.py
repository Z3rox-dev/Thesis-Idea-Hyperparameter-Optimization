#!/usr/bin/env python3
"""
Benchmark COV vs PF_orig su surrogati RF REALI.

Approccio: Creo surrogati RF trainati su funzioni sintetiche,
poi ottimizza il surrogato (simula scenario reale HPO).
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# FUNZIONI BASE PER TRAINING RF
# ============================================================================

def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    return float(sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)))

def branin(x):
    # 2D classic
    x1, x2 = x[0], x[1]
    a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
    r, s, t = 6, 10, 1/(8*np.pi)
    return float(a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)

def hartmann6(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    outer = 0
    for i in range(4):
        inner = 0
        for j in range(6):
            inner += A[i, j] * (x[j] - P[i, j])**2
        outer += alpha[i] * np.exp(-inner)
    return float(-outer)


# ============================================================================
# RF SURROGATE WRAPPER
# ============================================================================

class RFSurrogate:
    """Surrogato RF trainato su campioni della funzione originale."""
    
    def __init__(self, func, bounds, n_train=500, n_trees=100, seed=42):
        self.func = func
        self.bounds = bounds
        self.dim = len(bounds)
        
        # Genera training data
        rng = np.random.default_rng(seed)
        X_train = np.zeros((n_train, self.dim))
        for i, (lo, hi) in enumerate(bounds):
            X_train[:, i] = rng.uniform(lo, hi, n_train)
        
        y_train = np.array([func(x) for x in X_train])
        
        # Train RF
        self.rf = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        self.rf.fit(X_train, y_train)
        
        # Stats
        self.y_min = y_train.min()
        self.y_max = y_train.max()
    
    def evaluate(self, x) -> float:
        """Valuta usando il surrogato RF."""
        x_arr = np.array(x).reshape(1, -1)
        return float(self.rf.predict(x_arr)[0])


# ============================================================================
# BENCHMARK
# ============================================================================

def run_comparison(surrogate: RFSurrogate, n_trials: int, seed: int) -> Dict:
    """Confronta COV vs PF_orig."""
    
    results = {}
    
    for use_pf, label in [(False, 'COV'), (True, 'PF')]:
        opt = ALBA(
            bounds=surrogate.bounds,
            seed=seed,
            maximize=False,
            total_budget=n_trials,
            use_potential_field=use_pf,
            use_coherence_gating=True,
        )
        
        best = float('inf')
        for _ in range(n_trials):
            x = opt.ask()
            y = surrogate.evaluate(x)
            opt.tell(x, y)
            best = min(best, y)
        
        results[label] = best
        results[f'{label}_coh'] = opt._coherence_tracker.global_coherence
    
    return results


def main():
    print("=" * 75)
    print("  BENCHMARK: COV vs PF su Surrogati RF Reali")
    print("=" * 75)
    
    # Configurazione
    FUNCTIONS = {
        'sphere_5D': (sphere, [(-5.0, 5.0)] * 5),
        'sphere_10D': (sphere, [(-5.0, 5.0)] * 10),
        'rosenbrock_5D': (rosenbrock, [(-2.0, 2.0)] * 5),
        'rosenbrock_10D': (rosenbrock, [(-2.0, 2.0)] * 10),
        'branin_2D': (branin, [(-5.0, 10.0), (0.0, 15.0)]),
        'hartmann_6D': (hartmann6, [(0.0, 1.0)] * 6),
    }
    
    N_TRIALS = 200
    N_SEEDS = 15
    RF_TREES = 100
    RF_TRAIN = 1000
    
    print(f"\nConfig:")
    print(f"  RF: {RF_TREES} trees, {RF_TRAIN} training points")
    print(f"  Optimization: {N_TRIALS} trials, {N_SEEDS} seeds\n")
    
    all_results = []
    pf_wins = 0
    cov_wins = 0
    ties = 0
    
    for func_name, (func, bounds) in FUNCTIONS.items():
        print(f"\n{'='*50}")
        print(f"  {func_name}")
        print(f"{'='*50}")
        
        func_pf_wins = 0
        func_cov_wins = 0
        
        for seed_off in range(N_SEEDS):
            seed = 42 + seed_off * 1000
            
            # Crea surrogato RF
            surrogate = RFSurrogate(func, bounds, n_train=RF_TRAIN, n_trees=RF_TREES, seed=seed)
            
            # Run comparison
            results = run_comparison(surrogate, N_TRIALS, seed)
            
            # Winner
            pf_best = results['PF']
            cov_best = results['COV']
            
            rel_diff = abs(pf_best - cov_best) / (abs(min(pf_best, cov_best)) + 1e-10)
            
            if rel_diff < 0.01:  # 1% tolerance
                ties += 1
                winner = "TIE"
            elif pf_best < cov_best:
                pf_wins += 1
                func_pf_wins += 1
                winner = "PF ✓"
            else:
                cov_wins += 1
                func_cov_wins += 1
                winner = "COV ✓"
            
            coh = results['PF_coh']
            print(f"  s={seed}: PF={pf_best:.4f} COV={cov_best:.4f} (coh={coh:.2f}) → {winner}")
            
            all_results.append({
                'func': func_name,
                'seed': seed,
                'pf': pf_best,
                'cov': cov_best,
                'coherence': coh,
            })
        
        print(f"\n  {func_name} summary: PF={func_pf_wins}/{N_SEEDS}, COV={func_cov_wins}/{N_SEEDS}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    total = pf_wins + cov_wins + ties
    
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY - Surrogati RF Reali")
    print("=" * 75)
    
    print(f"\n  PF wins:  {pf_wins}/{total} ({100*pf_wins/total:.1f}%)")
    print(f"  COV wins: {cov_wins}/{total} ({100*cov_wins/total:.1f}%)")
    print(f"  Ties:     {ties}/{total}")
    
    # Media
    pf_avg = np.mean([r['pf'] for r in all_results])
    cov_avg = np.mean([r['cov'] for r in all_results])
    
    print(f"\n  Media PF:  {pf_avg:.4f}")
    print(f"  Media COV: {cov_avg:.4f}")
    
    improvement = 100 * (cov_avg - pf_avg) / cov_avg
    print(f"\n  ⚡ PF improvement: {improvement:+.1f}% vs COV")
    
    if pf_wins > cov_wins + 5:
        print("\n  ✅ PF VINCE su surrogati RF!")
    elif cov_wins > pf_wins + 5:
        print("\n  ❌ COV vince su surrogati RF")
    else:
        print("\n  ➖ Risultato neutro")


if __name__ == "__main__":
    main()
