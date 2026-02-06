#!/usr/bin/env python3
"""
Benchmark: CopulaHPO Elite vs Latent-CMA modes.

Confronto diretto tra le due modalità su funzioni continue.
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from copula_hpo_v2 import CopulaHPO_Continuous, HyperparameterSpec

# Try to import Optuna and Nevergrad
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import nevergrad as ng
    HAS_NEVERGRAD = True
except ImportError:
    HAS_NEVERGRAD = False


# =============================================================================
# Test Functions
# =============================================================================

def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def rastrigin(x):
    d = len(x)
    return float(10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def ackley(x):
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(sum_sq/d)) - np.exp(sum_cos/d) + 20 + np.e

def schwefel(x):
    d = len(x)
    return 418.9829*d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# =============================================================================
# Runners
# =============================================================================

def run_copula(fn, bounds, budget, seed, mode="elite", **kwargs):
    """Run CopulaHPO with specified mode."""
    # Pass budget hint for adaptive M in latent_cma
    opt = CopulaHPO_Continuous(bounds, seed=seed, mode=mode, budget=budget, **kwargs)
    for _ in range(budget):
        x = opt.ask()
        y = fn(x)
        opt.tell(x, y)
    return opt.best_y


def run_random(fn, bounds, budget, seed):
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    bounds_arr = np.asarray(bounds)
    lower, upper = bounds_arr[:, 0], bounds_arr[:, 1]
    
    best_y = float("inf")
    for _ in range(budget):
        x = lower + rng.random(len(lower)) * (upper - lower)
        y = fn(x)
        if y < best_y:
            best_y = y
    return best_y


def run_optuna(fn, bounds, budget, seed):
    """Optuna TPE."""
    if not HAS_OPTUNA:
        return float("nan")
    
    bounds_arr = np.asarray(bounds)
    d = len(bounds_arr)
    
    def objective(trial):
        x = np.array([
            trial.suggest_float(f"x{i}", bounds_arr[i, 0], bounds_arr[i, 1])
            for i in range(d)
        ])
        return fn(x)
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return study.best_value


def run_cma(fn, bounds, budget, seed):
    """CMA-ES via Nevergrad."""
    if not HAS_NEVERGRAD:
        return float("nan")
    
    bounds_arr = np.asarray(bounds)
    d = len(bounds_arr)
    
    param = ng.p.Array(shape=(d,)).set_bounds(bounds_arr[:, 0], bounds_arr[:, 1])
    opt = ng.optimizers.CMA(parametrization=param, budget=budget)
    opt.parametrization.random_state = np.random.RandomState(seed)
    
    best_y = float("inf")
    for _ in range(budget):
        x = opt.ask()
        y = fn(x.value)
        opt.tell(x, y)
        if y < best_y:
            best_y = y
    return best_y


# =============================================================================
# Benchmark
# =============================================================================

def benchmark():
    """Benchmark Elite vs Latent-CMA."""
    print("=" * 80)
    print("BENCHMARK: CopulaHPO Elite vs Latent-CMA")
    print("=" * 80)
    
    functions = [
        ("Sphere", sphere, [(-5, 5)] * 10),
        ("Rosenbrock", rosenbrock, [(-5, 10)] * 10),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * 10),
        ("Ackley", ackley, [(-32, 32)] * 10),
        ("Schwefel", schwefel, [(-500, 500)] * 10),
    ]
    
    budget = 400
    n_seeds = 5
    seeds = [42, 123, 456, 789, 999]
    
    all_results = {}
    
    for fn_name, fn, bounds in functions:
        print(f"\n{'='*70}")
        print(f"  {fn_name} (d={len(bounds)}, budget={budget})")
        print("=" * 70)
        
        results = {
            "Elite": [],
            "Latent-CMA": [],
            "Latent-CMA-Active": [],
            "Random": [],
        }
        if HAS_OPTUNA:
            results["Optuna"] = []
        if HAS_NEVERGRAD:
            results["CMA-ES"] = []
        
        for seed in seeds:
            print(f"  Seed {seed}:", end=" ", flush=True)
            
            d = len(bounds)
            
            # Elite mode: M auto (128)
            y_elite = run_copula(fn, bounds, budget, seed, mode="elite")
            results["Elite"].append(y_elite)
            print(f"Elite={y_elite:.4f}", end=" ", flush=True)
            
            # Latent-CMA mode: M auto (dimension-based with budget check)
            y_cma = run_copula(fn, bounds, budget, seed, mode="latent_cma")
            results["Latent-CMA"].append(y_cma)
            print(f"L-CMA={y_cma:.4f}", end=" ", flush=True)

            # Latent-CMA (active covariance update using worst samples)
            y_cma_active = run_copula(fn, bounds, budget, seed, mode="latent_cma", cma_active=True, cma_active_eta=0.3)
            results["Latent-CMA-Active"].append(y_cma_active)
            print(f"aCMA={y_cma_active:.4f}", end=" ", flush=True)

            # Random
            y_rand = run_random(fn, bounds, budget, seed)
            results["Random"].append(y_rand)
            print(f"Rand={y_rand:.4f}", end=" ", flush=True)
            
            # Optuna
            if HAS_OPTUNA:
                y_opt = run_optuna(fn, bounds, budget, seed)
                results["Optuna"].append(y_opt)
                print(f"Opt={y_opt:.4f}", end=" ", flush=True)
            
            # CMA-ES
            if HAS_NEVERGRAD:
                y_cma_ng = run_cma(fn, bounds, budget, seed)
                results["CMA-ES"].append(y_cma_ng)
                print(f"CMA={y_cma_ng:.4f}", end="", flush=True)
            
            print()
        
        # Summary
        print(f"\n  Summary {fn_name}:")
        print("  " + "-" * 50)
        for method, vals in results.items():
            arr = np.array(vals)
            mean, std = arr.mean(), arr.std()
            print(f"    {method:12s}: {mean:10.4f} ± {std:.4f}")
        
        # Find winner
        means = {k: np.mean(v) for k, v in results.items()}
        winner = min(means, key=means.get)
        print(f"  >>> Winner: {winner}")
        
        all_results[fn_name] = results
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    wins = {k: 0 for k in ["Elite", "Latent-CMA", "Random"]}
    wins["Latent-CMA-Active"] = 0
    if HAS_OPTUNA:
        wins["Optuna"] = 0
    if HAS_NEVERGRAD:
        wins["CMA-ES"] = 0
    
    for fn_name, results in all_results.items():
        means = {k: np.mean(v) for k, v in results.items()}
        winner = min(means, key=means.get)
        wins[winner] += 1
        
        elite_mean = np.mean(results["Elite"])
        lcma_mean = np.mean(results["Latent-CMA"])
        acma_mean = np.mean(results["Latent-CMA-Active"])
        best_name, best_val = min(
            [("Elite", elite_mean), ("Latent-CMA", lcma_mean), ("Latent-CMA-Active", acma_mean)],
            key=lambda t: t[1],
        )
        print(f"  {fn_name:15s}: Elite={elite_mean:10.4f}, L-CMA={lcma_mean:10.4f}, aCMA={acma_mean:10.4f} -> {best_name}")
    
    print(f"\n  Wins: {wins}")


if __name__ == "__main__":
    benchmark()
