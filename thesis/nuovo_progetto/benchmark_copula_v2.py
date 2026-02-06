#!/usr/bin/env python3
"""
Benchmark: CopulaHPO v2 vs Optuna vs CMA-ES vs Random.

Tests on:
1. Pure continuous functions (Sphere, Rosenbrock, Rastrigin)
2. Mixed-space synthetic problem
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from copula_hpo_v2 import CopulaHPO, CopulaHPO_Continuous, HyperparameterSpec

# Try to import Optuna and Nevergrad
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not available")

try:
    import nevergrad as ng
    HAS_NEVERGRAD = True
except ImportError:
    HAS_NEVERGRAD = False
    print("Warning: Nevergrad not available")


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


def mixed_objective(config: dict) -> float:
    """
    Mixed-space problem.
    
    Optimal: lr=0.001, optimizer=adam, n_layers=3, dropout=0.1, batch_size=64
    """
    score = 0.0
    
    # Learning rate: log-scale, best at 0.001
    lr = config["lr"]
    score += (np.log10(lr) + 3) ** 2 * 10
    
    # Optimizer: adam is best
    opt_scores = {"adam": 0, "sgd": 2, "rmsprop": 1, "adamw": 0.5}
    score += opt_scores.get(config["optimizer"], 1)
    
    # n_layers: 3 is best
    score += (config["n_layers"] - 3) ** 2
    
    # Dropout: 0.1 is best
    score += (config["dropout"] - 0.1) ** 2 * 5
    
    # Batch size: 64 is best
    bs_scores = {16: 2, 32: 1, 64: 0, 128: 1, 256: 2}
    score += bs_scores.get(config["batch_size"], 1)
    
    # Add interaction: adam works better with low lr
    if config["optimizer"] == "adam" and lr < 0.01:
        score -= 0.5
    
    return score


MIXED_SPECS = [
    HyperparameterSpec("lr", "continuous", (1e-5, 1e-1)),
    HyperparameterSpec("optimizer", "categorical", ["adam", "sgd", "rmsprop", "adamw"]),
    HyperparameterSpec("n_layers", "integer", (1, 6)),
    HyperparameterSpec("dropout", "continuous", (0.0, 0.5)),
    HyperparameterSpec("batch_size", "categorical", [16, 32, 64, 128, 256]),
]


# =============================================================================
# Runners
# =============================================================================

def run_copula_continuous(fn, bounds, budget, seed):
    """Run CopulaHPO on continuous problem."""
    opt = CopulaHPO_Continuous(bounds, seed=seed)
    for _ in range(budget):
        x = opt.ask()
        y = fn(x)
        opt.tell(x, y)
    return opt.best_y


def run_copula_mixed(fn, specs, budget, seed):
    """Run CopulaHPO on mixed problem."""
    opt = CopulaHPO(specs, seed=seed)
    for _ in range(budget):
        x = opt.ask()
        y = fn(x)
        opt.tell(x, y)
    return opt.best_y


def run_random_continuous(fn, bounds, budget, seed):
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


def run_random_mixed(fn, specs, budget, seed):
    """Random search on mixed space."""
    rng = np.random.default_rng(seed)
    
    best_y = float("inf")
    for _ in range(budget):
        x = {}
        for spec in specs:
            if spec.type == "continuous":
                x[spec.name] = rng.uniform(spec.bounds[0], spec.bounds[1])
            elif spec.type == "categorical":
                x[spec.name] = rng.choice(spec.bounds)
            elif spec.type == "integer":
                x[spec.name] = rng.integers(spec.bounds[0], spec.bounds[1] + 1)
        y = fn(x)
        if y < best_y:
            best_y = y
    return best_y


def run_optuna_continuous(fn, bounds, budget, seed):
    """Optuna TPE on continuous problem."""
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


def run_optuna_mixed(fn, specs, budget, seed):
    """Optuna TPE on mixed problem."""
    if not HAS_OPTUNA:
        return float("nan")
    
    def objective(trial):
        x = {}
        for spec in specs:
            if spec.type == "continuous":
                x[spec.name] = trial.suggest_float(spec.name, spec.bounds[0], spec.bounds[1])
            elif spec.type == "categorical":
                x[spec.name] = trial.suggest_categorical(spec.name, spec.bounds)
            elif spec.type == "integer":
                x[spec.name] = trial.suggest_int(spec.name, spec.bounds[0], spec.bounds[1])
        return fn(x)
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    return study.best_value


def run_cma_continuous(fn, bounds, budget, seed):
    """CMA-ES via Nevergrad on continuous problem."""
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

def benchmark_continuous():
    """Benchmark on continuous functions."""
    print("\n" + "=" * 70)
    print("BENCHMARK: CONTINUOUS FUNCTIONS")
    print("=" * 70)
    
    functions = [
        ("Sphere", sphere, [(-5, 5)] * 10),
        ("Rosenbrock", rosenbrock, [(-5, 10)] * 10),
        ("Rastrigin", rastrigin, [(-5.12, 5.12)] * 10),
    ]
    
    budget = 200
    n_seeds = 5
    
    methods = ["CopulaHPO", "Random"]
    if HAS_OPTUNA:
        methods.append("Optuna")
    if HAS_NEVERGRAD:
        methods.append("CMA-ES")
    
    for fn_name, fn, bounds in functions:
        print(f"\n  {fn_name} (d={len(bounds)}, budget={budget}):")
        print("  " + "-" * 55)
        
        results = {}
        
        # CopulaHPO
        scores = []
        for seed in range(n_seeds):
            scores.append(run_copula_continuous(fn, bounds, budget, seed))
        results["CopulaHPO"] = (np.mean(scores), np.std(scores))
        
        # Random
        scores = []
        for seed in range(n_seeds):
            scores.append(run_random_continuous(fn, bounds, budget, seed))
        results["Random"] = (np.mean(scores), np.std(scores))
        
        # Optuna
        if HAS_OPTUNA:
            scores = []
            for seed in range(n_seeds):
                scores.append(run_optuna_continuous(fn, bounds, budget, seed))
            results["Optuna"] = (np.mean(scores), np.std(scores))
        
        # CMA-ES
        if HAS_NEVERGRAD:
            scores = []
            for seed in range(n_seeds):
                scores.append(run_cma_continuous(fn, bounds, budget, seed))
            results["CMA-ES"] = (np.mean(scores), np.std(scores))
        
        # Print results
        best_method = min(results.keys(), key=lambda k: results[k][0])
        for method in methods:
            if method in results:
                mean, std = results[method]
                marker = " ✓" if method == best_method else ""
                print(f"    {method:15s}: {mean:12.4f} ± {std:8.4f}{marker}")


def benchmark_mixed():
    """Benchmark on mixed-space problem."""
    print("\n" + "=" * 70)
    print("BENCHMARK: MIXED SPACE (continuous + categorical + integer)")
    print("=" * 70)
    
    budget = 100
    n_seeds = 5
    
    print(f"\n  Mixed HPO Problem (budget={budget}):")
    print("  Params: lr(cont), optimizer(cat), n_layers(int), dropout(cont), batch_size(cat)")
    print("  " + "-" * 60)
    
    results = {}
    
    # CopulaHPO
    scores = []
    for seed in range(n_seeds):
        scores.append(run_copula_mixed(mixed_objective, MIXED_SPECS, budget, seed))
    results["CopulaHPO"] = (np.mean(scores), np.std(scores))
    
    # Random
    scores = []
    for seed in range(n_seeds):
        scores.append(run_random_mixed(mixed_objective, MIXED_SPECS, budget, seed))
    results["Random"] = (np.mean(scores), np.std(scores))
    
    # Optuna
    if HAS_OPTUNA:
        scores = []
        for seed in range(n_seeds):
            scores.append(run_optuna_mixed(mixed_objective, MIXED_SPECS, budget, seed))
        results["Optuna"] = (np.mean(scores), np.std(scores))
    
    # Print
    best_method = min(results.keys(), key=lambda k: results[k][0])
    for method, (mean, std) in results.items():
        marker = " ✓" if method == best_method else ""
        print(f"    {method:15s}: {mean:12.4f} ± {std:8.4f}{marker}")


def main():
    print("=" * 70)
    print("CopulaHPO v2 BENCHMARK")
    print("=" * 70)
    
    start = time.time()
    
    benchmark_continuous()
    benchmark_mixed()
    
    elapsed = time.time() - start
    print(f"\n\nTotal time: {elapsed:.1f}s")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
