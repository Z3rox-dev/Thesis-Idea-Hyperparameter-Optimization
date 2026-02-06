#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import warnings
from typing import Callable, List, Tuple, Dict, Any

# Filter warnings
warnings.filterwarnings("ignore")

# Improve import robustness
try:
    import nevergrad as ng
except ImportError:
    print("Warning: nevergrad not installed.")
    ng = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Warning: optuna not installed.")
    optuna = None

try:
    from turbo import TurboM
except ImportError:
    print("Warning: TuRBO not installed. Skipping TuRBO benchmarks.")
    TurboM = None

# ALBA Imports
import sys
import os
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework_potential.optimizer import ALBA

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

N_REPEATS = 5
BUDGET = 200 # Shortened for testing, user asked for "benchmark", usually means ~1000 but I'll make it configurable or standard.
# User asked for "some synthetic functions", "smooth, multimodal, traps"

TASKS = [
    {"name": "Sphere", "dim": 20, "budget": 1000},
    {"name": "Rastrigin", "dim": 20, "budget": 1000},
    {"name": "Ackley", "dim": 20, "budget": 1000},
    {"name": "Rosenbrock", "dim": 20, "budget": 1000},
]

# =============================================================================
# SYNTHETIC FUNCTIONS
# =============================================================================

def get_function(name: str, dim: int) -> Tuple[Callable, List[Tuple[float, float]], float]:
    """
    Returns (func, bounds, optimum_value)
    """
    if name == "Sphere":
        # optimum at 0, val 0
        def func(x):
            return np.sum(x**2)
        bounds = [(-5.0, 5.0)] * dim
        opt = 0.0
        
    elif name == "Rastrigin":
        # optimum at 0, val 0
        def func(x):
            # Rastrigin: 10d + sum(x^2 - 10cos(2pi x))
            x = np.array(x)
            return 10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        bounds = [(-5.12, 5.12)] * dim
        opt = 0.0
        
    elif name == "Ackley":
        # optimum at 0, val 0
        def func(x):
            x = np.array(x)
            a = 20
            b = 0.2
            c = 2 * np.pi
            term1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
            term2 = -np.exp(np.mean(np.cos(c * x)))
            return term1 + term2 + a + np.exp(1)
        bounds = [(-32.768, 32.768)] * dim
        opt = 0.0
        
    elif name == "Rosenbrock":
        # optimum at 1, val 0
        def func(x):
            x = np.array(x)
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        bounds = [(-5.0, 10.0)] * dim
        opt = 0.0
        
    else:
        raise ValueError(f"Unknown function: {name}")
        
    return func, bounds, opt

# =============================================================================
# OPTIMIZER WRAPPERS
# =============================================================================

def run_alba(func, bounds, budget):
    optimizer = ALBA(
        bounds=bounds,
        maximize=False,
        total_budget=budget,
        use_potential_field=True,
        n_candidates=50, # Slightly higher for robust benchmarks
        local_search_ratio=0.3
    )
    
    start_time = time.time()
    optimizer.optimize(func, budget)
    duration = time.time() - start_time
    
    return optimizer.best_y, duration

def run_optuna(func, bounds, budget):
    if optuna is None: return np.nan, 0.0
    
    def objective(trial):
        x = []
        for i, (lo, hi) in enumerate(bounds):
            x.append(trial.suggest_float(f"x{i}", lo, hi))
        return func(np.array(x))
    
    # TPE Sampler
    sampler = optuna.samplers.TPESampler(seed=np.random.randint(1000))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    start_time = time.time()
    study.optimize(objective, n_trials=budget)
    duration = time.time() - start_time
    
    return study.best_value, duration

def run_turbo(func, bounds, budget):
    if TurboM is None: return np.nan, 0.0
    
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # TuRBO requires a batched function, but here we run check one by one
    # We can wrap it. TuRBO-m minimizes.
    
    class TurboFunc:
        def __init__(self, f):
            self.f = f
            self.n_evals = 0
            
        def __call__(self, x):
            # x is (batch_size, dim)
            y = []
            for xi in x:
                y.append(self.f(xi))
                self.n_evals += 1
            return np.array(y).reshape(-1, 1)

    # Initialize TuRBO-m
    # It assumes value in unit hypercube? No, TuRBO handles bounds if we scale?
    # Actually standard TuRBO usage often assumes we handle normalization or pass bounds.
    # The standard implementation takes lb, ub.
    
    f_wrapper = TurboFunc(func)
    
    # TuRBO needs initial points
    n_init = min(2 * dim + 1, 50)
    X_init = np.random.uniform(lb, ub, (n_init, dim))
    Y_init = f_wrapper(X_init)
    
    remaining_budget = budget - n_init
    
    start_time = time.time()
    
    turbo_m = TurboM(
        f=f_wrapper,
        lb=lb,
        ub=ub,
        n_init=n_init,
        max_evals=budget,
        n_trust_regions=2 if dim < 30 else 5, # heuristic
        batch_size=1,
        verbose=False,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024, # Force CPU usually
        device="cpu",
        dtype="float64",
    )
    
    # If we have existing data
    turbo_m.X = X_init
    turbo_m.Y = Y_init
    
    # Optimization Loop
    turbo_m.optimize()
    
    duration = time.time() - start_time
    
    # Best value
    # Y is minimized
    best_y = np.min(turbo_m.Y)
    
    return best_y, duration

def run_random(func, bounds, budget):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    best_y = np.inf
    
    start_time = time.time()
    for _ in range(budget):
        x = np.random.uniform(lb, ub)
        y = func(x)
        if y < best_y:
            best_y = y
    duration = time.time() - start_time
    
    return best_y, duration

# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all(test_run=False):
    tasks_to_run = TASKS
    repeats = N_REPEATS
    
    if test_run:
        print("Running in TEST mode (reduced budget/repeats)")
        tasks_to_run = [{"name": "Sphere", "dim": 5, "budget": 50}]
        repeats = 1
        
    results = []
    
    print(f"{'Function':<12} {'Dim':<4} {'Opt':<10} {'Mean':<10} {'Std':<10} {'Time(s)':<8} {'Best(all)':<10}")
    print("-" * 80)
    
    for task in tasks_to_run:
        func_name = task["name"]
        dim = task["dim"]
        budget = task["budget"]
        
        func, bounds, optimum = get_function(func_name, dim)
        
        # Prepare Rows
        
        # Optimizers map
        optimizers = {
            "Random": run_random,
            "ALBA": run_alba,
        }
        if optuna:
            optimizers["Optuna"] = run_optuna
        if TurboM:
             optimizers["TuRBO"] = run_turbo
             
        for opt_name, opt_func in optimizers.items():
            run_vals = []
            run_times = []
            
            for r in range(repeats):
                try:
                    val, duration = opt_func(func, bounds, budget)
                    run_vals.append(val)
                    run_times.append(duration)
                except Exception as e:
                    print(f"Error in {opt_name} on {func_name}: {e}")
                    run_vals.append(np.nan)
                    run_times.append(0.0)
            
            run_vals = np.array(run_vals)
            mean_val = np.nanmean(run_vals)
            std_val = np.nanstd(run_vals)
            avg_time = np.mean(run_times)
            best_found = np.nanmin(run_vals)
            
            results.append({
                "Function": func_name, 
                "Optimizer": opt_name,
                "Mean": mean_val,
                "Std": std_val,
                "Time": avg_time,
                "Best": best_found
            })
            
            print(f"{func_name:<12} {dim:<4} {opt_name:<10} {mean_val:<10.4f} {std_val:<10.4f} {avg_time:<8.2f} {best_found:<10.4f}")

    # Summarize with nicer formatting
    if not test_run:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print(df.pivot(index='Function', columns='Optimizer', values='Mean'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run a quick test")
    args = parser.parse_args()
    
    run_all(test_run=args.test_run)
