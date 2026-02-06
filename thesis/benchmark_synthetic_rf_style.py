#!/usr/bin/env python3
"""
Benchmark COV vs COV+PF su funzioni sintetiche con output RF-style (stepped).

Funzioni testate:
- Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Levy
- Dimensioni: 5D, 10D, 20D
- Output: discretizzato in N step (simula surrogato RF)

Ipotesi: PF aiuta su superfici stepped perché stabilizza la ricerca.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# FUNZIONI SINTETICHE CLASSICHE
# ============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function: sum(x^2). Optimum at origin."""
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. Optimum at (1,1,...,1)."""
    result = 0.0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return float(result)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function: multimodal. Optimum at origin."""
    A = 10
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function: multimodal with many local minima."""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e)


def griewank(x: np.ndarray) -> float:
    """Griewank function: multimodal."""
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_sq - prod_cos + 1)


def levy(x: np.ndarray) -> float:
    """Levy function: multimodal. Optimum at (1,1,...,1)."""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return float(term1 + term2 + term3)


def styblinski_tang(x: np.ndarray) -> float:
    """Styblinski-Tang: simple multimodal. Min at x=-2.903534."""
    return float(0.5 * np.sum(x**4 - 16*x**2 + 5*x))


def schwefel(x: np.ndarray) -> float:
    """Schwefel function: deceptive multimodal."""
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


# ============================================================================
# DISCRETIZZAZIONE RF-STYLE
# ============================================================================

def discretize_rf_style(f: Callable, n_steps: int = 50) -> Callable:
    """
    Wrap a function to produce RF-style stepped output.
    
    Simula un surrogato Random Forest che produce output a gradini.
    """
    def wrapped(x: np.ndarray) -> float:
        y = f(x)
        # Discretizza in n_steps livelli
        # Prima normalizziamo rispetto a un range tipico
        y_clipped = np.clip(y, 0, 1000)  # Evita overflow
        step_size = 1000 / n_steps
        y_discretized = np.floor(y_clipped / step_size) * step_size
        return float(y_discretized)
    return wrapped


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_single_comparison(
    func: Callable,
    dim: int,
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
        
        results[label] = best_y
    
    return results


def run_benchmark(
    functions: Dict[str, Callable],
    dimensions: List[int],
    n_steps_list: List[int],
    n_trials: int,
    n_seeds: int,
    base_seed: int = 42,
) -> Dict:
    """Run full benchmark."""
    
    all_results = []
    
    total_experiments = len(functions) * len(dimensions) * len(n_steps_list) * n_seeds
    exp_idx = 0
    
    for func_name, base_func in functions.items():
        for dim in dimensions:
            for n_steps in n_steps_list:
                # Discretize function
                if n_steps == 0:
                    func = base_func  # Smooth (no discretization)
                    steps_label = "SMOOTH"
                else:
                    func = discretize_rf_style(base_func, n_steps)
                    steps_label = f"{n_steps}steps"
                
                # Bounds
                if func_name == "schwefel":
                    bounds = [(-500.0, 500.0)] * dim
                elif func_name == "styblinski_tang":
                    bounds = [(-5.0, 5.0)] * dim
                else:
                    bounds = [(-5.0, 5.0)] * dim
                
                pf_wins = 0
                cov_wins = 0
                ties = 0
                
                for seed_offset in range(n_seeds):
                    seed = base_seed + seed_offset * 1000
                    exp_idx += 1
                    
                    results = run_single_comparison(func, dim, bounds, n_trials, seed)
                    
                    pf_best = results["PF"]
                    cov_best = results["COV"]
                    
                    # Compare (tolerance for discretized values)
                    if abs(pf_best - cov_best) < 1e-6:
                        ties += 1
                        winner = "TIE"
                    elif pf_best < cov_best:
                        pf_wins += 1
                        winner = "PF"
                    else:
                        cov_wins += 1
                        winner = "COV"
                    
                    print(f"[{exp_idx}/{total_experiments}] {func_name} {dim}D {steps_label} seed={seed}: "
                          f"PF={pf_best:.2f} vs COV={cov_best:.2f} → {winner}")
                
                all_results.append({
                    'function': func_name,
                    'dim': dim,
                    'steps': n_steps,
                    'pf_wins': pf_wins,
                    'cov_wins': cov_wins,
                    'ties': ties,
                })
                
                print(f"  ➤ {func_name} {dim}D {steps_label}: PF={pf_wins} COV={cov_wins} TIE={ties}")
    
    return all_results


def main():
    print("=" * 75)
    print("  BENCHMARK: COV vs COV+PF su funzioni sintetiche RF-style")
    print("  Ipotesi: PF aiuta su superfici stepped (discretizzate)")
    print("=" * 75)
    
    # Configurazione
    FUNCTIONS = {
        'sphere': sphere,
        'rosenbrock': rosenbrock,
        'rastrigin': rastrigin,
        'ackley': ackley,
        'griewank': griewank,
        'levy': levy,
        'styblinski': styblinski_tang,
    }
    
    DIMENSIONS = [5, 10, 20]
    N_STEPS_LIST = [0, 25, 50, 100]  # 0 = smooth, altri = RF-style
    N_TRIALS = 150
    N_SEEDS = 5
    
    print(f"\nConfigurazione:")
    print(f"  Funzioni: {list(FUNCTIONS.keys())}")
    print(f"  Dimensioni: {DIMENSIONS}")
    print(f"  Step RF: {N_STEPS_LIST} (0=smooth)")
    print(f"  Trial per run: {N_TRIALS}")
    print(f"  Seed per config: {N_SEEDS}")
    print(f"  Totale esperimenti: {len(FUNCTIONS) * len(DIMENSIONS) * len(N_STEPS_LIST) * N_SEEDS}")
    print()
    
    results = run_benchmark(
        FUNCTIONS, DIMENSIONS, N_STEPS_LIST, N_TRIALS, N_SEEDS
    )
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    
    # Per tipo di discretizzazione
    for n_steps in N_STEPS_LIST:
        subset = [r for r in results if r['steps'] == n_steps]
        total_pf = sum(r['pf_wins'] for r in subset)
        total_cov = sum(r['cov_wins'] for r in subset)
        total_tie = sum(r['ties'] for r in subset)
        total = total_pf + total_cov + total_tie
        
        steps_label = "SMOOTH" if n_steps == 0 else f"{n_steps} steps"
        print(f"\n{steps_label}:")
        print(f"  PF wins: {total_pf}/{total} ({100*total_pf/total:.1f}%)")
        print(f"  COV wins: {total_cov}/{total} ({100*total_cov/total:.1f}%)")
        print(f"  Ties: {total_tie}/{total}")
    
    # Per dimensione
    print("\n" + "-" * 40)
    print("Per dimensione:")
    for dim in DIMENSIONS:
        subset = [r for r in results if r['dim'] == dim]
        total_pf = sum(r['pf_wins'] for r in subset)
        total_cov = sum(r['cov_wins'] for r in subset)
        total_tie = sum(r['ties'] for r in subset)
        total = total_pf + total_cov + total_tie
        print(f"  {dim}D: PF={total_pf}/{total} ({100*total_pf/total:.1f}%) | COV={total_cov}/{total}")
    
    # Per funzione
    print("\n" + "-" * 40)
    print("Per funzione:")
    for func_name in FUNCTIONS.keys():
        subset = [r for r in results if r['function'] == func_name]
        total_pf = sum(r['pf_wins'] for r in subset)
        total_cov = sum(r['cov_wins'] for r in subset)
        total_tie = sum(r['ties'] for r in subset)
        total = total_pf + total_cov + total_tie
        print(f"  {func_name}: PF={total_pf}/{total} | COV={total_cov}/{total}")
    
    # Stepped vs Smooth
    print("\n" + "-" * 40)
    print("STEPPED (RF-style) vs SMOOTH:")
    
    smooth = [r for r in results if r['steps'] == 0]
    stepped = [r for r in results if r['steps'] > 0]
    
    smooth_pf = sum(r['pf_wins'] for r in smooth)
    smooth_cov = sum(r['cov_wins'] for r in smooth)
    smooth_total = smooth_pf + smooth_cov + sum(r['ties'] for r in smooth)
    
    stepped_pf = sum(r['pf_wins'] for r in stepped)
    stepped_cov = sum(r['cov_wins'] for r in stepped)
    stepped_total = stepped_pf + stepped_cov + sum(r['ties'] for r in stepped)
    
    print(f"  SMOOTH:  PF={smooth_pf}/{smooth_total} ({100*smooth_pf/smooth_total:.1f}%) | COV={smooth_cov}")
    print(f"  STEPPED: PF={stepped_pf}/{stepped_total} ({100*stepped_pf/stepped_total:.1f}%) | COV={stepped_cov}")
    
    pf_boost = (stepped_pf/stepped_total) - (smooth_pf/smooth_total)
    print(f"\n  ⚡ PF boost on STEPPED: {pf_boost*100:+.1f}%")
    
    if pf_boost > 0.05:
        print("  ✅ IPOTESI CONFERMATA: PF aiuta su superfici RF-style!")
    elif pf_boost < -0.05:
        print("  ❌ IPOTESI SMENTITA: PF peggiora su RF-style")
    else:
        print("  ➖ Nessuna differenza significativa")


if __name__ == "__main__":
    main()
