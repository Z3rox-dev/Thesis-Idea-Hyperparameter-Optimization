#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
import warnings
import traceback
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

# Try importing ALBA, if fails, might need to fix path
try:
    from alba_framework_potential.optimizer import ALBA
except ImportError:
    print("Error: Could not import ALBA. Check python path.")
# Try importing HPO Minimal (assume in thesis folder)
try:
    from hpo_minimal import HPOptimizer
except ImportError:
    print("Warning: HPOptimizer not found in path.")
    HPOptimizer = None

# ... (Previous imports)

def run_hpo_minimal(func, bounds, budget):
    if HPOptimizer is None: return np.nan, 0.0
    
    # HPOptimizer maximizes by default, so set maximize=False
    opt = HPOptimizer(bounds=bounds, maximize=False, seed=np.random.randint(100000))
    
    start_time = time.time()
    # It returns (best_x, best_score).
    # HPOptimizer already un-signs the score in its return statement.
    _, best_score = opt.optimize(func, budget)
    duration = time.time() - start_time
    
    return best_score, duration



N_REPEATS = 5
BUDGET = 200 # Shortened for testing, user asked for "benchmark", usually means ~1000 but I'll make it configurable or standard.
# User asked for "some synthetic functions", "smooth, multimodal, traps"

TASKS = [
    {"name": "IntegerSphere", "dim": 20, "budget": 500},
    {"name": "IntegerRastrigin", "dim": 20, "budget": 500},
]

# =============================================================================
# SYNTHETIC FUNCTIONS (Via Nevergrad)
# =============================================================================

try:
    from nevergrad import functions as ng_funcs
    NG_FUNCS = True
except ImportError:
    NG_FUNCS = False
    
# Fallback Generators if NG missing
def _make_sphere(dim):
    return lambda x: float(np.sum(np.array(x)**2))

def _make_rastrigin(dim):
    return lambda x: float(10 * dim + np.sum(np.array(x)**2 - 10 * np.cos(2 * np.pi * np.array(x))))

def _make_ackley(dim):
    return lambda x: float(-20 * np.exp(-0.2 * np.sqrt(np.sum(np.array(x)**2)/dim)) - 
                           np.exp(np.sum(np.cos(2*np.pi*np.array(x)))/dim) + 20 + np.exp(1))
                           
def _make_rosenbrock(dim):
    return lambda x: float(np.sum(100.0*(np.array(x)[1:]-np.array(x)[:-1]**2)**2 + (1-np.array(x)[:-1])**2))

def get_function(name: str, dim: int) -> Tuple[Callable, List[Tuple[float, float]], float]:
    """
    Returns (func_wrapper, bounds, optimum_value)
    
    Uses nevergrad.functions.ArtificialFunction for the core logic,
    but applies simple numpy wrapping and explicit bounds for consistency.
    """
    
    # 1. Define standard bounds
    is_integer = False
    base_name = name
    
    if name.startswith("Integer"):
        is_integer = True
        base_name = name.replace("Integer", "")

    if base_name == "Sphere":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "Rastrigin":
        bounds = [(-5.12, 5.12)] * dim
        opt_val = 0.0
    elif base_name == "Ackley":
        bounds = [(-32.768, 32.768)] * dim
        opt_val = 0.0
    elif base_name == "Rosenbrock":
        bounds = [(-5.0, 10.0)] * dim
        opt_val = 0.0
    elif base_name == "Lunacek":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "Griewank":
        bounds = [(-600.0, 600.0)] * dim
        opt_val = 0.0
    elif base_name == "DeceptiveMultimodal":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "StepDoubleLinearSlope":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "StepEllipsoid":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "Schwefel_1_2":
        bounds = [(-100.0, 100.0)] * dim
        opt_val = 0.0
    elif base_name == "Cigar":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "Discus":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    elif base_name == "BentCigar":
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0
    else:
        # Default safety
        bounds = [(-5.0, 5.0)] * dim
        opt_val = 0.0


    # 2. Get function logic
    ng_success = False
    
    if NG_FUNCS:
        try:
            ng_func = ng_funcs.ArtificialFunction(base_name.lower(), block_dimension=dim, rotation=False, translation_factor=0.0)
            def func_wrapper(x):
                if is_integer: x = np.round(np.array(x))
                val = ng_func(x)
                if hasattr(val, "item"): return val.item()
                return float(val)
            return func_wrapper, bounds, opt_val
        except Exception:
            ng_success = False

    # Local Fallback
    print(f"Fallback to local implementation for {base_name}")
    if base_name == "Sphere":
        f = _make_sphere(dim)
    elif base_name == "Rastrigin":
        f = _make_rastrigin(dim)
    elif base_name == "Ackley":
        f = _make_ackley(dim)
    elif base_name == "Rosenbrock":
        f = _make_rosenbrock(dim)
    else:
        print(f"Unknown function {base_name}, defaulting to Sphere.")
        f = _make_sphere(dim)
        
    def func_wrapper(x):
        if is_integer: x = np.round(np.array(x))
        return f(x)
        
    return func_wrapper, bounds, opt_val

# =============================================================================
# OPTIMIZER WRAPPERS
# =============================================================================

try:
    from alba_framework_potential.optimizer import ALBA
    from alba_framework_potential.local_search import CovarianceLocalSearchSampler
except ImportError:
    ALBA = None
    CovarianceLocalSearchSampler = None
    print("ALBA_Framework not found in path. ALBA will be skipped.")

def run_alba(func, bounds, budget, drilling=False):
    """Specific wrapper for ALBA execution."""
    if ALBA is None: return np.nan, 0.0
    t0 = time.time()
    
    optimizer = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        local_search_ratio=0.3,
        use_drilling=drilling
    )
    
    _, val = optimizer.optimize(func, budget)
    return val, time.time() - t0

def run_alba_covariance(func, bounds, budget, drilling=False):
    """Wrapper for ALBA with Covariance Local Search."""
    if ALBA is None: return np.nan, 0.0
    t0 = time.time()
    
    # Inject weighted covariance sampler
    cov_sampler = CovarianceLocalSearchSampler(
        radius_start=0.15,
        radius_end=0.01,
        top_k_fraction=0.15, # Weighted logic
        min_points_fit=10
        # dim will be inferred
    )
    
    optimizer = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=cov_sampler,
        use_drilling=drilling,
        use_potential_field=True
    )
    
    _, val = optimizer.optimize(func, budget)
    return val, time.time() - t0

# ECFS Import
try:
    sys.path.append('/mnt/workspace/thesis/nuovo_progetto')
    from ecfs import minimize as ecfs_minimize
    ECFS_AVAILABLE = True
except ImportError:
    ECFS_AVAILABLE = False
    print("Warning: ECFS not found")

def run_ecfs(func, bounds, budget):
    """Wrapper for ECFS."""
    if not ECFS_AVAILABLE: return np.nan, 0.0
    t0 = time.time()
    # ECFS expects raw function (scalar) and bounds as list of tuples
    res = ecfs_minimize(
        func, 
        bounds, 
        T=budget, 
        seed=np.random.randint(10000),
        M=128,          # Default candidates
        top_eval=10     # Slight batch
    )
    duration = time.time() - t0
    return res['best_y'], duration

# CopulaTPE Import
try:
    sys.path.append('/mnt/workspace/thesis/nuovo_progetto')
    from copula_tpe import minimize as copula_tpe_minimize
    COPULA_TPE_AVAILABLE = True
except ImportError:
    COPULA_TPE_AVAILABLE = False
    print("Warning: CopulaTPE not found")

def run_copula_tpe(func, bounds, budget):
    """Wrapper for CopulaTPE (Gaussian copula + TPE marginals)."""
    if not COPULA_TPE_AVAILABLE:
        return np.nan, 0.0
    t0 = time.time()
    res = copula_tpe_minimize(
        func,
        bounds,
        T=budget,
        seed=np.random.randint(10000),
        M=256,
        top_eval=8,
        local_k=0,
    )
    duration = time.time() - t0
    return res["best_y"], duration

# CopulaHPO v2 Import
try:
    sys.path.append('/mnt/workspace/thesis/nuovo_progetto')
    from copula_hpo_v2 import CopulaHPO_Continuous
    COPULA_V2_AVAILABLE = True
except ImportError:
    COPULA_V2_AVAILABLE = False
    print("Warning: CopulaHPO v2 not found")

def run_copula_v2(func, bounds, budget):
    """Wrapper for CopulaHPO v2 (mixed-space copula, used here in continuous mode)."""
    if not COPULA_V2_AVAILABLE:
        return np.nan, 0.0
    t0 = time.time()
    opt = CopulaHPO_Continuous(
        bounds,
        seed=np.random.randint(10000),
        mode="latent_cma",
        gamma=0.2,
        M=32,
        top_eval=1,  # avoid pending/batching during sequential evaluation
        eps_explore=0.05,
        alpha_corr=0.1,
        reg=1e-6,
        u_clip=1e-6,
    )
    for _ in range(int(budget)):
        x = opt.ask()
        y = float(func(x))
        opt.tell(x, y)
    return float(opt.best_y), time.time() - t0

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

import contextlib
import os

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ... (wrapper functions)

def run_turbo(func, bounds, budget):
    if TurboM is None: return np.nan, 0.0
    
    # ... (setup code remains same) ...
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    class TurboFunc:
        def __init__(self, f):
            self.f = f
            self.n_evals = 0   
        def __call__(self, x):
            x = np.array(x)
            if x.ndim == 1:
                self.n_evals += 1
                return self.f(x)
            y = []
            for xi in x:
                y.append(self.f(xi))
                self.n_evals += 1
            return np.array(y).reshape(-1, 1)

    f_wrapper = TurboFunc(func)
    n_init = min(2 * dim + 1, 50)
    start_time = time.time()
    lb = lb.astype(np.float64)
    ub = ub.astype(np.float64)
    import torch
    torch.set_num_threads(1)
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Suppress TuRBO initialization prints
    with suppress_stdout():
        turbo_m = TurboM(
            f=f_wrapper,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=budget,
            n_trust_regions=5,
            batch_size=10,
            verbose=False,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=30,
            min_cuda=0,
            device=device,
            dtype="float32",
        )
        turbo_m.optimize()
    
    duration = time.time() - start_time
    best_y = np.min(turbo_m.fX)
    return best_y, duration

def run_ng_optimizer(optimizer_cls, func, bounds, budget, name="NGOpt"):
    if ng is None: return np.nan, 0.0
    
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    with suppress_stdout():
        parametrization = ng.p.Array(shape=(dim,)).set_bounds(lower=lb, upper=ub)
        optimizer = optimizer_cls(parametrization=parametrization, budget=budget)
        start_time = time.time()
        recommendation = optimizer.minimize(func)
        duration = time.time() - start_time
    
    best_y = func(recommendation.value)
    return best_y, duration

def run_random(func, bounds, budget):
    if ng is None:
        # Fallback to numpy if ng missing
        return run_random_numpy(func, bounds, budget)
    return run_ng_optimizer(ng.optimizers.RandomSearch, func, bounds, budget, name="Random")

def run_cma(func, bounds, budget):
    return run_ng_optimizer(ng.optimizers.CMA, func, bounds, budget, name="CMA")

def run_random_numpy(func, bounds, budget):
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
        tasks_to_run = [{"name": "Sphere", "dim": 5, "budget": 100}]
        repeats = 1
    else:
        # Run 17: Mega Suite (10 Functions)
        tasks_to_run = [
            # Standard
            {"name": "Sphere", "dim": 20, "budget": 500},
            {"name": "Rastrigin", "dim": 20, "budget": 500},
            {"name": "Ackley", "dim": 20, "budget": 500},
            {"name": "Rosenbrock", "dim": 20, "budget": 500},
            {"name": "Ellipsoid", "dim": 20, "budget": 500},
            # New / Hard / Anisotropic
            {"name": "Cigar", "dim": 20, "budget": 500},
            {"name": "Discus", "dim": 20, "budget": 500},
            {"name": "BentCigar", "dim": 20, "budget": 500},
        ]
        repeats = 3

    print(f"Starting Benchmark Suite (Test={test_run})...")
    
    results = []
    
    print(f"{'Function':<12} {'Dim':<4} {'Opt':<10} {'Mean':<10} {'Std':<10} {'Time(s)':<8} {'Best(all)':<10}")
    print("-" * 80)
    
    for task in tasks_to_run:
        name = task["name"]
        dim = task["dim"]
        budget = task["budget"]
        
        func, bounds, optimum = get_function(name, dim)
        func_name = name # Alias for compatibility
        
        print(f"\n--- Benchmark: {name} ({dim}D), Budget {budget} ---")
        
        # Define optimizers to test
        optimizers = {
            "Random": run_random,
            "ALBA": lambda f, b, bu: run_alba(f, b, bu, drilling=False),
            "ALBA_Cov": lambda f, b, bu: run_alba_covariance(f, b, bu, drilling=False),
            "ALBA_Drill": lambda f, b, bu: run_alba(f, b, bu, drilling=True),
            "ALBA_Hybrid_Drill": lambda f, b, bu: run_alba_covariance(f, b, bu, drilling=True),
        }
        if ng:
            optimizers["CMA"] = run_cma
            
        if optuna:
            optimizers["Optuna"] = run_optuna
            
        if ECFS_AVAILABLE:
            optimizers["ECFS"] = run_ecfs

        if COPULA_TPE_AVAILABLE:
            optimizers["CopulaTPE"] = run_copula_tpe

        if COPULA_V2_AVAILABLE:
            optimizers["CopulaV2"] = run_copula_v2
        
        # User requested to remove TuRBO for this run to save time
        # if TurboM:
        #      optimizers["TuRBO"] = run_turbo
             
        for opt_name, opt_func in optimizers.items():
            run_vals = []
            run_times = []
            
            for r in range(repeats):
                try:
                    val, duration = opt_func(func, bounds, budget)
                    # print(f"DEBUG: {opt_name} rep {r} val={val} type={type(val)}")
                    run_vals.append(val)
                    run_times.append(duration)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
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
        df.to_csv("benchmark_results.csv", index=False)
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print(df.pivot(index='Function', columns='Optimizer', values='Mean'))
        print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run a quick test")
    args = parser.parse_args()
    
    run_all(test_run=args.test_run)
