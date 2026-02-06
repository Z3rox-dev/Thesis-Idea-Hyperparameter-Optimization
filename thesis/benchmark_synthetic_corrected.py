#!/usr/bin/env python3
"""
Benchmark COV vs COV+PF su funzioni sintetiche - Versione corretta.

Problema del test precedente: con 150 trial le traiettorie non divergono.
Soluzione: 300 trial, spazio più piccolo, funzioni con gradiente chiaro.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# FUNZIONI SINTETICHE
# ============================================================================

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray) -> float:
    result = 0.0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return float(result)

def rastrigin(x: np.ndarray) -> float:
    A = 10
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def ackley(x: np.ndarray) -> float:
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e)

def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return float(term1 + term2 + term3)

def styblinski_tang(x: np.ndarray) -> float:
    return float(0.5 * np.sum(x**4 - 16*x**2 + 5*x))


# ============================================================================
# DISCRETIZZAZIONE RF-STYLE (più fine)
# ============================================================================

def discretize_rf_style(f: Callable, n_bins: int = 200) -> Callable:
    """
    Discretizza l'output in n_bins livelli uniformi.
    Usa un range dinamico basato sui valori osservati.
    """
    observed_min = [float('inf')]
    observed_max = [float('-inf')]
    
    def wrapped(x: np.ndarray) -> float:
        y = f(x)
        # Aggiorna range osservato
        observed_min[0] = min(observed_min[0], y)
        observed_max[0] = max(observed_max[0], y)
        
        # Discretizza solo se abbiamo un range valido
        if observed_max[0] > observed_min[0]:
            y_range = observed_max[0] - observed_min[0]
            bin_size = y_range / n_bins
            y_discretized = observed_min[0] + np.floor((y - observed_min[0]) / bin_size) * bin_size
            return float(y_discretized)
        return float(y)
    return wrapped


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_single_comparison(
    func: Callable,
    bounds: List[Tuple[float, float]],
    n_trials: int,
    seed: int,
) -> Dict:
    """Run COV vs COV+PF comparison."""
    
    results = {}
    
    for use_pf in [False, True]:
        label = "PF" if use_pf else "COV"
        
        opt = ALBA(
            bounds=bounds,
            seed=seed,
            maximize=False,
            total_budget=n_trials,
            use_potential_field=use_pf,
            use_coherence_gating=True,
        )
        
        best_y = float('inf')
        for _ in range(n_trials):
            x = opt.ask()
            if isinstance(x, dict):
                x_arr = np.array(list(x.values()))
            else:
                x_arr = np.array(x)
            y = func(x_arr)
            opt.tell(x, y)
            best_y = min(best_y, y)
        
        # Anche info sulla coherence finale
        final_coh = opt._coherence_tracker.global_coherence
        n_leaves = len(opt.leaves)
        results[label] = {'best': best_y, 'coherence': final_coh, 'leaves': n_leaves}
    
    return results


def main():
    print("=" * 75)
    print("  BENCHMARK: COV vs COV+PF - Versione corretta")
    print("  - 300 trial per run")
    print("  - Spazio più piccolo per favorire convergenza")
    print("=" * 75)
    
    # Configurazione
    FUNCTIONS = {
        'sphere': (sphere, [(-2.0, 2.0)]),
        'rosenbrock': (rosenbrock, [(-2.0, 2.0)]),
        'rastrigin': (rastrigin, [(-3.0, 3.0)]),
        'ackley': (ackley, [(-3.0, 3.0)]),
        'levy': (levy, [(-5.0, 5.0)]),
        'styblinski': (styblinski_tang, [(-5.0, 5.0)]),
    }
    
    DIMENSIONS = [5, 10]
    DISCRETIZE = [False, True]  # Smooth vs Stepped
    N_TRIALS = 300
    N_SEEDS = 10
    
    print(f"\nConfigurazione:")
    print(f"  Funzioni: {list(FUNCTIONS.keys())}")
    print(f"  Dimensioni: {DIMENSIONS}")
    print(f"  Trial per run: {N_TRIALS}")
    print(f"  Seed per config: {N_SEEDS}")
    print()
    
    all_results = []
    exp_idx = 0
    total = len(FUNCTIONS) * len(DIMENSIONS) * len(DISCRETIZE) * N_SEEDS
    
    for func_name, (base_func, base_bounds) in FUNCTIONS.items():
        for dim in DIMENSIONS:
            bounds = base_bounds * dim
            
            for discretize in DISCRETIZE:
                if discretize:
                    func = discretize_rf_style(base_func, n_bins=100)
                    mode = "STEPPED"
                else:
                    func = base_func
                    mode = "SMOOTH"
                
                pf_wins = 0
                cov_wins = 0
                ties = 0
                
                for seed_offset in range(N_SEEDS):
                    seed = 42 + seed_offset * 1000
                    exp_idx += 1
                    
                    results = run_single_comparison(func, bounds, N_TRIALS, seed)
                    
                    pf_best = results["PF"]['best']
                    cov_best = results["COV"]['best']
                    
                    # Compare con tolleranza relativa
                    rel_diff = abs(pf_best - cov_best) / (abs(cov_best) + 1e-10)
                    
                    if rel_diff < 0.001:  # 0.1% tolerance
                        ties += 1
                        winner = "TIE"
                    elif pf_best < cov_best:
                        pf_wins += 1
                        winner = "PF"
                    else:
                        cov_wins += 1
                        winner = "COV"
                    
                    coh = results["PF"]['coherence']
                    print(f"[{exp_idx}/{total}] {func_name} {dim}D {mode} s={seed}: "
                          f"PF={pf_best:.4f} vs COV={cov_best:.4f} (coh={coh:.2f}) → {winner}")
                
                all_results.append({
                    'function': func_name,
                    'dim': dim,
                    'mode': mode,
                    'pf_wins': pf_wins,
                    'cov_wins': cov_wins,
                    'ties': ties,
                })
                
                print(f"  ➤ {func_name} {dim}D {mode}: PF={pf_wins} COV={cov_wins} TIE={ties}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    
    # Per modo
    for mode in ["SMOOTH", "STEPPED"]:
        subset = [r for r in all_results if r['mode'] == mode]
        total_pf = sum(r['pf_wins'] for r in subset)
        total_cov = sum(r['cov_wins'] for r in subset)
        total_tie = sum(r['ties'] for r in subset)
        total_n = total_pf + total_cov + total_tie
        
        print(f"\n{mode}:")
        print(f"  PF wins: {total_pf}/{total_n} ({100*total_pf/total_n:.1f}%)")
        print(f"  COV wins: {total_cov}/{total_n} ({100*total_cov/total_n:.1f}%)")
        print(f"  Ties: {total_tie}/{total_n}")
    
    # Per funzione
    print("\n" + "-" * 40)
    print("Per funzione:")
    for func_name in FUNCTIONS.keys():
        subset = [r for r in all_results if r['function'] == func_name]
        total_pf = sum(r['pf_wins'] for r in subset)
        total_cov = sum(r['cov_wins'] for r in subset)
        total_tie = sum(r['ties'] for r in subset)
        total_n = total_pf + total_cov + total_tie
        status = "✅ PF" if total_pf > total_cov + 2 else ("❌ COV" if total_cov > total_pf + 2 else "➖")
        print(f"  {func_name}: PF={total_pf}/{total_n} | COV={total_cov}/{total_n} {status}")
    
    # Totale
    total_pf = sum(r['pf_wins'] for r in all_results)
    total_cov = sum(r['cov_wins'] for r in all_results)
    total_tie = sum(r['ties'] for r in all_results)
    total_n = total_pf + total_cov + total_tie
    
    print("\n" + "=" * 75)
    print(f"  TOTALE: PF={total_pf}/{total_n} ({100*total_pf/total_n:.1f}%) | COV={total_cov}/{total_n} ({100*total_cov/total_n:.1f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
