#!/usr/bin/env python3
"""
Test dell'intuizione: il Potential Field aiuta su funzioni discretizzate?

Ipotesi: ParamNet usa RF che produce output "a gradini". Il PF potrebbe
aiutare a stabilizzare la ricerca su queste superfici discretizzate.

Test:
1. Funzione SMOOTH (continua, liscia)
2. Funzione DISCRETIZED (stessa base, ma output quantizzato come una RF)

Se PF aiuta di più sulla versione discretizzata, l'intuizione è confermata.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# FUNZIONI TEST
# ============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function - smooth, convex."""
    return np.sum(x**2)

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock - smooth, narrow valley."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin - smooth with many local minima."""
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def ackley(x: np.ndarray) -> float:
    """Ackley - smooth with global minimum at origin."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


def discretize_rf_style(y: float, n_bins: int = 20, y_range: Tuple[float, float] = (0, 100)) -> float:
    """
    Discretizza l'output come farebbe una Random Forest.
    Simula il fatto che RF restituisce la media di un subset di alberi,
    producendo un numero finito di valori possibili.
    """
    y_min, y_max = y_range
    y_clipped = np.clip(y, y_min, y_max)
    bin_size = (y_max - y_min) / n_bins
    bin_idx = int((y_clipped - y_min) / bin_size)
    bin_idx = min(bin_idx, n_bins - 1)
    # Return center of bin (simulating RF averaging)
    return y_min + (bin_idx + 0.5) * bin_size


class SmoothFunction:
    """Wrapper per funzione liscia."""
    def __init__(self, func: Callable, dim: int, scale: float = 1.0):
        self.func = func
        self.dim = dim
        self.scale = scale
        self.name = func.__name__
    
    def __call__(self, x: np.ndarray) -> float:
        return self.func(x) * self.scale


class DiscretizedFunction:
    """Wrapper che discretizza l'output (simula RF surrogate)."""
    def __init__(self, func: Callable, dim: int, scale: float = 1.0, 
                 n_bins: int = 20, y_range: Tuple[float, float] = (0, 100)):
        self.func = func
        self.dim = dim
        self.scale = scale
        self.n_bins = n_bins
        self.y_range = y_range
        self.name = f"{func.__name__}_discretized"
    
    def __call__(self, x: np.ndarray) -> float:
        y_smooth = self.func(x) * self.scale
        return discretize_rf_style(y_smooth, self.n_bins, self.y_range)


# ============================================================================
# BENCHMARK
# ============================================================================

def run_alba(
    func: Callable,
    dim: int,
    n_trials: int,
    seed: int,
    use_potential_field: bool,
) -> float:
    """Run ALBA, return best value found."""
    
    bounds = [(-5.0, 5.0)] * dim
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        maximize=False,
        total_budget=n_trials,
        use_potential_field=use_potential_field,
        use_coherence_gating=True,
    )
    
    best_y = np.inf
    for _ in range(n_trials):
        x = opt.ask()
        # x può essere dict o array
        if isinstance(x, dict):
            x_arr = np.array(list(x.values()))
        else:
            x_arr = np.array(x)
        y = func(x_arr)
        opt.tell(x, y)
        if y < best_y:
            best_y = y
    
    return best_y


def test_function_pair(
    base_func: Callable,
    dim: int,
    scale: float,
    n_bins: int,
    y_range: Tuple[float, float],
    budget: int,
    n_seeds: int,
) -> dict:
    """Test SMOOTH vs DISCRETIZED version of same function."""
    
    smooth = SmoothFunction(base_func, dim, scale)
    discretized = DiscretizedFunction(base_func, dim, scale, n_bins, y_range)
    
    results = {
        'smooth': {'cov': [], 'pf': [], 'wins_cov': 0, 'wins_pf': 0},
        'discretized': {'cov': [], 'pf': [], 'wins_cov': 0, 'wins_pf': 0},
    }
    
    print(f"\n{'='*70}")
    print(f"  {base_func.__name__.upper()} (dim={dim}, bins={n_bins})")
    print(f"{'='*70}")
    
    # Test SMOOTH
    print(f"\n  SMOOTH version:")
    for seed in range(n_seeds):
        val_cov = run_alba(smooth, dim, budget, 100 + seed, use_potential_field=False)
        val_pf = run_alba(smooth, dim, budget, 100 + seed, use_potential_field=True)
        
        results['smooth']['cov'].append(val_cov)
        results['smooth']['pf'].append(val_pf)
        
        if val_cov < val_pf:
            results['smooth']['wins_cov'] += 1
            winner = "COV"
        elif val_pf < val_cov:
            results['smooth']['wins_pf'] += 1
            winner = "PF"
        else:
            winner = "TIE"
        
        print(f"    Seed {seed}: COV={val_cov:.4f} PF={val_pf:.4f} → {winner}")
    
    # Test DISCRETIZED
    print(f"\n  DISCRETIZED version ({n_bins} bins):")
    for seed in range(n_seeds):
        val_cov = run_alba(discretized, dim, budget, 100 + seed, use_potential_field=False)
        val_pf = run_alba(discretized, dim, budget, 100 + seed, use_potential_field=True)
        
        results['discretized']['cov'].append(val_cov)
        results['discretized']['pf'].append(val_pf)
        
        if val_cov < val_pf:
            results['discretized']['wins_cov'] += 1
            winner = "COV"
        elif val_pf < val_cov:
            results['discretized']['wins_pf'] += 1
            winner = "PF"
        else:
            winner = "TIE"
        
        print(f"    Seed {seed}: COV={val_cov:.4f} PF={val_pf:.4f} → {winner}")
    
    # Summary
    for version in ['smooth', 'discretized']:
        r = results[version]
        r['mean_cov'] = np.mean(r['cov'])
        r['mean_pf'] = np.mean(r['pf'])
        r['delta_pct'] = (r['mean_cov'] - r['mean_pf']) / abs(r['mean_cov']) * 100 if r['mean_cov'] != 0 else 0
    
    print(f"\n  Summary {base_func.__name__}:")
    print(f"    SMOOTH:      COV wins {results['smooth']['wins_cov']}/{n_seeds}, PF wins {results['smooth']['wins_pf']}/{n_seeds} (Δ={results['smooth']['delta_pct']:+.1f}%)")
    print(f"    DISCRETIZED: COV wins {results['discretized']['wins_cov']}/{n_seeds}, PF wins {results['discretized']['wins_pf']}/{n_seeds} (Δ={results['discretized']['delta_pct']:+.1f}%)")
    
    # PF advantage on discretized vs smooth
    pf_advantage_smooth = results['smooth']['wins_pf'] - results['smooth']['wins_cov']
    pf_advantage_disc = results['discretized']['wins_pf'] - results['discretized']['wins_cov']
    pf_boost = pf_advantage_disc - pf_advantage_smooth
    
    results['pf_boost_on_discretized'] = pf_boost
    
    if pf_boost > 0:
        print(f"    → PF ha +{pf_boost} vantaggio in più su DISCRETIZED ✅")
    elif pf_boost < 0:
        print(f"    → PF ha {pf_boost} vantaggio in meno su DISCRETIZED ❌")
    else:
        print(f"    → PF ha stesso vantaggio su entrambe ➖")
    
    return results


def main():
    print("=" * 75)
    print("  TEST: Potential Field su funzioni SMOOTH vs DISCRETIZED")
    print("  Ipotesi: PF aiuta di più su funzioni discretizzate (RF-style)")
    print("=" * 75)
    
    DIM = 8
    BUDGET = 200
    N_SEEDS = 10
    N_BINS = 15  # Numero di "gradini" (come RF con pochi alberi)
    
    # Test functions con parametri appropriati
    test_configs = [
        (sphere, 1.0, (0, 200)),      # sphere: range tipico [0, 200]
        (rosenbrock, 0.01, (0, 100)), # rosenbrock: scalato
        (rastrigin, 1.0, (0, 300)),   # rastrigin: range tipico
        (ackley, 1.0, (0, 25)),       # ackley: range [0, ~22]
    ]
    
    all_results = {}
    
    for func, scale, y_range in test_configs:
        results = test_function_pair(
            base_func=func,
            dim=DIM,
            scale=scale,
            n_bins=N_BINS,
            y_range=y_range,
            budget=BUDGET,
            n_seeds=N_SEEDS,
        )
        all_results[func.__name__] = results
    
    # ========================================================================
    # FINAL ANALYSIS
    # ========================================================================
    print("\n" + "=" * 75)
    print("  ANALISI FINALE")
    print("=" * 75)
    
    print(f"\n{'Function':<15} {'SMOOTH':<25} {'DISCRETIZED':<25} {'PF Boost':<10}")
    print(f"{'':<15} {'COV wins / PF wins':<25} {'COV wins / PF wins':<25} {'on disc.':<10}")
    print("-" * 75)
    
    total_smooth_cov = 0
    total_smooth_pf = 0
    total_disc_cov = 0
    total_disc_pf = 0
    total_boost = 0
    
    for func_name, r in all_results.items():
        s = r['smooth']
        d = r['discretized']
        boost = r['pf_boost_on_discretized']
        
        status = "✅" if boost > 0 else ("❌" if boost < 0 else "➖")
        
        print(f"{func_name:<15} {s['wins_cov']:>5} / {s['wins_pf']:<5} (Δ={s['delta_pct']:+.1f}%)   "
              f"{d['wins_cov']:>5} / {d['wins_pf']:<5} (Δ={d['delta_pct']:+.1f}%)   "
              f"{boost:>+3} {status}")
        
        total_smooth_cov += s['wins_cov']
        total_smooth_pf += s['wins_pf']
        total_disc_cov += d['wins_cov']
        total_disc_pf += d['wins_pf']
        total_boost += boost
    
    print("-" * 75)
    n_funcs = len(all_results)
    total_runs = n_funcs * N_SEEDS
    
    print(f"{'TOTALE':<15} {total_smooth_cov:>5} / {total_smooth_pf:<5} "
          f"({total_smooth_pf/total_runs*100:.0f}% PF)      "
          f"{total_disc_cov:>5} / {total_disc_pf:<5} "
          f"({total_disc_pf/total_runs*100:.0f}% PF)      "
          f"{total_boost:>+3}")
    
    print("\n" + "=" * 75)
    if total_boost > 0:
        print(f"🔥 IPOTESI CONFERMATA: PF guadagna +{total_boost} wins su funzioni discretizzate!")
        print("   Il Potential Field stabilizza l'ottimizzazione su superfici 'a gradini'.")
    elif total_boost < 0:
        print(f"❌ IPOTESI RIFIUTATA: PF perde {abs(total_boost)} wins su funzioni discretizzate.")
    else:
        print("➖ IPOTESI NON SUPPORTATA: PF ha stesso effetto su smooth e discretized.")
    print("=" * 75)
    
    # Extra: test con diversi livelli di discretizzazione
    print("\n\n" + "=" * 75)
    print("  EXTRA: Effetto del numero di bins (livello di discretizzazione)")
    print("=" * 75)
    
    bins_levels = [5, 10, 20, 50, 100]
    
    print(f"\n  Funzione: Rosenbrock (dim={DIM})")
    print(f"  Più bins = più liscio, meno bins = più a gradini")
    print()
    
    for n_bins in bins_levels:
        discretized = DiscretizedFunction(rosenbrock, DIM, 0.01, n_bins, (0, 100))
        
        wins_cov = 0
        wins_pf = 0
        
        for seed in range(N_SEEDS):
            val_cov = run_alba(discretized, DIM, BUDGET, 100 + seed, use_potential_field=False)
            val_pf = run_alba(discretized, DIM, BUDGET, 100 + seed, use_potential_field=True)
            
            if val_cov < val_pf:
                wins_cov += 1
            elif val_pf < val_cov:
                wins_pf += 1
        
        pf_rate = wins_pf / N_SEEDS * 100
        status = "🔥" if wins_pf > wins_cov else ("❌" if wins_cov > wins_pf else "➖")
        print(f"    {n_bins:>3} bins: COV={wins_cov}, PF={wins_pf} ({pf_rate:.0f}% PF wins) {status}")
    
    print("\n  Se l'ipotesi è corretta: meno bins → più PF wins")


if __name__ == "__main__":
    main()
