#!/usr/bin/env python3
"""
Benchmark ALBA Coherence vs TuRBO-m on synthetic functions.
Dimensions: 10, 20.
"""

import sys
import os
import time
import json
import numpy as np
import argparse
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

# Add workspace root to path to import ParamSpace
sys.path.insert(0, "/mnt/workspace")
# Add thesis to path to import alba_framework_coherence
sys.path.insert(0, "/mnt/workspace/thesis")

try:
    from ParamSpace import FUNS
except ImportError:
    print("Could not import ParamSpace. Make sure you are in the right directory.")
    sys.exit(1)

try:
    from alba_framework_coherence.optimizer import ALBA
except ImportError:
    print("Could not import ALBA from alba_framework_coherence.")
    sys.exit(1)

try:
    from turbo import TurboM
except ImportError:
    TurboM = None
    print("Warning: TurboM not found. TuRBO-m benchmarks will be skipped.")

try:
    from alba_framework.optimizer import ALBA as ALBA_V1
except ImportError:
    print("Could not import ALBA from alba_framework.optimizer.")
    ALBA_V1 = None

try:
    from hpo_minimal_improved import HPOptimizer as HPOptimizer_Minimal
except ImportError:
    print("Could not import HPOptimizer from hpo_minimal_improved.")
    HPOptimizer_Minimal = None

try:
    from hpo_lgs_v3 import HPOptimizer as HPOptimizer_LGS
except ImportError:
    print("Could not import HPOptimizer from hpo_lgs_v3.")
    HPOptimizer_LGS = None

try:
    from alba_framework_potential.optimizer import ALBA as ALBA_Potential
except ImportError:
    print("Could not import ALBA from alba_framework_potential.optimizer.")
    ALBA_Potential = None

@dataclass
class BenchmarkResult:
    method: str
    function: str
    dim: int
    seed: int
    best_score: float
    time_seconds: float
    evaluations: int

def run_alba_coherence(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    
    start_time = time.time()
    
    # ALBA minimizes by default (maximize=False)
    opt = ALBA(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
    )
    
    evals = 0
    
    def objective_wrapper(x):
        nonlocal evals
        evals += 1
        return func(x)

    best_x, best_y = opt.optimize(objective_wrapper, budget=budget)
    
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="ALBA-Coherence",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=evals
    )

def run_alba_potential(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    if ALBA_Potential is None:
        raise ImportError("ALBA Potential not available")
        
    start_time = time.time()
    
    # ALBA minimizes by default (maximize=False)
    opt = ALBA_Potential(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_potential_field=True
    )
    
    evals = 0
    
    def objective_wrapper(x):
        nonlocal evals
        evals += 1
        return func(x)

    best_x, best_y = opt.optimize(objective_wrapper, budget=budget)
    
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="ALBA-Potential",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=evals
    )

def run_alba_v1(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    if ALBA_V1 is None:
        raise ImportError("ALBA V1 not available")

    start_time = time.time()
    
    opt = ALBA_V1(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
    )
    
    evals = 0
    def objective_wrapper(x):
        nonlocal evals
        evals += 1
        return func(x)

    best_x, best_y = opt.optimize(objective_wrapper, budget=budget)
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="ALBA-V1",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=evals
    )

def run_hpo_minimal(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    if HPOptimizer_Minimal is None:
        raise ImportError("HPO Minimal not available")

    start_time = time.time()
    
    # HPO Minimal maximizes by default, so we wrap objective to negate
    opt = HPOptimizer_Minimal(
        bounds=bounds,
        maximize=True, # We will flip sign in wrapper
        seed=seed
    )
    
    evals = 0
    def objective_wrapper(x):
        nonlocal evals
        evals += 1
        return -func(x) # Negate since HPO optimizes for max

    best_x, best_y_max = opt.optimize(objective_wrapper, budget=budget)
    best_y = -best_y_max # Flip back for reporting
    
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="HPO-Minimal",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=evals
    )

def run_hpo_lgs(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    if HPOptimizer_LGS is None:
        raise ImportError("HPO LGS not available")

    start_time = time.time()
    
    # HPO LGS supports maximize=False in init
    opt = HPOptimizer_LGS(
        bounds=bounds,
        maximize=False, 
        seed=seed
    )
    
    evals = 0
    def objective_wrapper(x):
        nonlocal evals
        evals += 1
        return func(x)

    best_x, best_y = opt.optimize(objective_wrapper, budget=budget)
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="HPO-LGS",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=evals
    )

def run_random(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    best_y = float('inf')
    evals = 0
    
    for _ in range(budget):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        y = func(x)
        evals += 1
        if y < best_y:
            best_y = y
            
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="Random",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=evals
    )

def run_turbo_m(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    budget: int,
    seed: int
) -> BenchmarkResult:
    
    start_time = time.time()
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    eval_count = 0
    
    def f(x):
        nonlocal eval_count
        # Turbo might pass x as (1, dim) or (dim,). Flatten it.
        x_eval = x.flatten()
        val = func(x_eval)
        eval_count += 1
        return val

    # Initialize TuRBO
    # n_init must be larger than batch size? usually 2*dim
    n_init = min(2 * dim, budget // 2)
    
    turbo = TurboM(
        f=f,
        lb=lb,
        ub=ub,
        n_init=n_init,
        max_evals=budget,
        n_trust_regions=5,
        batch_size=1,
        verbose=False,
        use_ard=True,
        device="cpu",
        dtype="float64"
    )
    
    turbo.optimize()
    
    # turbo.fX contains the observations (best is min)
    best_y = turbo.fX.min().item()
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        method="TuRBO-m",
        function=func.__name__ if hasattr(func, '__name__') else "unknown",
        dim=dim,
        seed=seed,
        best_score=best_y,
        time_seconds=elapsed,
        evaluations=eval_count
    )

def main():
    parser = argparse.ArgumentParser(description="Benchmark ALBA Coherence vs TuRBO-m vs Others")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20], help="Dimensions to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds to run")
    parser.add_argument("--budget", type=int, default=200, help="Evaluation budget")
    parser.add_argument("--functions", type=str, nargs="+", default=None, help="Specific functions to run (default: all)")
    args = parser.parse_args()
    
    available_funs = FUNS 
    func_names = args.functions if args.functions else list(available_funs.keys())
    
    print(f"Running Benchmark: ALBA-Coherence vs ALBA-V1 vs HPO-Minimal vs HPO-LGS vs Random vs TuRBO-m")
    print(f"Dimensions: {args.dims}")
    print(f"Seeds: {args.seeds}")
    print(f"Budget: {args.budget}")
    print(f"Functions: {func_names}")
    print("-" * 60)
    
    # Create results directory
    results_dir = "/mnt/workspace/thesis/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"benchmark_alba_full_{timestamp}.json")
    
    all_results_data = []

    methods = [
        # ("TuRBO-m", run_turbo_m),
        ("ALBA-Coherence", run_alba_coherence),
        ("ALBA-Potential", run_alba_potential),
        ("ALBA-V1", run_alba_v1),
        # ("HPO-Minimal", run_hpo_minimal),
        ("HPO-LGS", run_hpo_lgs),
        ("Random", run_random)
    ]

    for dim in args.dims:
        print(f"\n{'='*20} Dimension: {dim} {'='*20}")
        
        for fname in func_names:
            if fname not in available_funs:
                print(f"Warning: Function {fname} not found in ParamSpace. Skipping.")
                continue
                
            func_obj, default_bounds_10d = available_funs[fname]
            single_bound = default_bounds_10d[0]
            current_bounds = [single_bound] * dim
            
            print(f"\n--- Function: {fname} ({dim}D) ---")
            
            scores_by_method = {m[0]: [] for m in methods}
            
            for seed in args.seeds:
                print(f"  Seed {seed}:")
                for name, runner in methods:
                    res_dict = None
                    try:
                        res = runner(func_obj, dim, current_bounds, args.budget, seed)
                        scores_by_method[name].append(res.best_score)
                        res_dict = res.__dict__
                        print(f"    {name:<15}: {res.best_score:.6f} ({res.time_seconds:.2f}s)")
                    except Exception as e:
                        print(f"    {name:<15}: FAILED ({e})")
                        res_dict = {"method": name, "function": fname, "dim": dim, "seed": seed, "error": str(e)}
                    
                    if res_dict: all_results_data.append(res_dict)
                
                # Save immediately
                with open(results_file, "w") as f:
                    json.dump(all_results_data, f, indent=2)

            # Summary for this function
            print(f"  Summary:")
            avgs = []
            for name in scores_by_method:
                s = scores_by_method[name]
                if s:
                    avg = np.mean(s)
                    avgs.append((name, avg))
            
            avgs.sort(key=lambda x: x[1]) # Sort by best score (min)
            for i, (name, avg) in enumerate(avgs):
                 print(f"    {i+1}. {name:<15}: {avg:.6f}")

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
