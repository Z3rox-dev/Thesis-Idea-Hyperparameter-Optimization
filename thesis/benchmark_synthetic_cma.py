#!/usr/bin/env python3
"""
Benchmark CubeHPO vs Optuna vs Random on synthetic test functions.
Fast evaluation for algorithm validation.
"""
import numpy as np
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List

# Import synthetic functions
from synthetic_functions import (
    sphere, rosenbrock, rastrigin, ackley, levy, griewank
)

# Import optimizers
sys.path.insert(0, str(Path(__file__).parent))
from hpo_curvature import QuadHPO

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("Warning: pycma (cma) not available")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    method: str
    function: str
    seed: int
    best_score: float
    best_params: Dict[str, float]
    eval_count: int
    time_seconds: float
    all_scores: List[float]


def run_curvature(func: Callable, dim: int, bounds: tuple, 
                  budget: int, seed: int, func_name: str, verbose: bool = False) -> BenchmarkResult:
    """Run QuadHPO with curvature-based splitting."""
    np.random.seed(seed)
    
    # IMPORTANT: QuadHPO has a bug with non-unit bounds
    # Use [0,1] bounds and handle conversion in objective
    bounds_normalized = [(0.0, 1.0) for _ in range(dim)]
    
    # Objective wrapper
    evals = []
    
    # Run QuadHPO with normalized [0,1] bounds
    start = time.time()
    hpo = QuadHPO(
        bounds=bounds_normalized,
        beta=0.05,
        lambda_geo=2.0,
        full_epochs=50,
        maximize=False,  # minimize - QuadHPO will negate scores internally
        rng_seed=seed
    )
    
    # Objective wrapper for QuadHPO
    def objective_wrapper(x_norm, epochs=1):
        # Convert from [0,1] to actual bounds
        x = np.array([bounds[0] + (bounds[1] - bounds[0]) * xi for xi in x_norm])
        score = func(x)
        evals.append(score)
        if verbose:
            params_str = " | ".join([f"x{i}: {x[i]:.4f}" for i in range(min(3, dim))])
            if dim > 3:
                params_str += " | ..."
            # Use hpo.trial_id for logging
            try:
                trial_idx = int(hpo.trial_id)
            except:
                trial_idx = len(evals)
            print(f"{func_name} | seed {seed} | curv trial {trial_idx} | {params_str} | score: {score:.6f}")
        return score
    
    # Run optimization
    hpo.optimize(objective_wrapper, budget=budget)
    
    elapsed = time.time() - start
    
    # Best score: usa direttamente i valori veri che hai valutato
    if evals:
        best_score = float(np.min(evals))
    else:
        best_score = float("inf")
    
    # Best params: controlla solo che l'attributo esista e non sia None
    best_x_norm = getattr(hpo, "best_x_norm", None)
    if best_x_norm is not None:
        best_x_norm = np.asarray(best_x_norm, dtype=float)
        x_best = bounds[0] + (bounds[1] - bounds[0]) * best_x_norm
        best_params = {f"x{i}": float(x_best[i]) for i in range(dim)}
    else:
        best_params = {f"x{i}": 0.0 for i in range(dim)}
    
    return BenchmarkResult(
        method="curvature",
        function=func_name,
        seed=seed,
        best_score=best_score,
        best_params=best_params,
        eval_count=len(evals),
        time_seconds=elapsed,
        all_scores=evals
    )


def run_optuna(func: Callable, dim: int, bounds: tuple,
               budget: int, seed: int, func_name: str, verbose: bool = False) -> BenchmarkResult:
    """Run Optuna TPE."""
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available")
    
    evals = []
    
    def objective(trial):
        params = [trial.suggest_float(f"x{i}", bounds[0], bounds[1]) for i in range(dim)]
        x = np.array(params)
        score = func(x)
        evals.append(score)
        if verbose:
            params_str = " | ".join([f"x{i}: {params[i]:.4f}" for i in range(min(3, dim))])
            if dim > 3:
                params_str += " | ..."
            print(f"{func_name} | seed {seed} | optuna trial {len(evals)} | {params_str} | score: {score:.6f}")
        return score
    
    start = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=budget)
    elapsed = time.time() - start
    
    best_params_dict = {f"x{i}": study.best_params[f"x{i}"] for i in range(dim)}
    
    return BenchmarkResult(
        method="optuna",
        function=func_name,
        seed=seed,
        best_score=study.best_value,
        best_params=best_params_dict,
        eval_count=len(evals),
        time_seconds=elapsed,
        all_scores=evals
    )


def run_random(func: Callable, dim: int, bounds: tuple,
               budget: int, seed: int, func_name: str, verbose: bool = False) -> BenchmarkResult:
    """Run random search baseline."""
    np.random.seed(seed)
    
    evals = []
    best_score = float('inf')
    best_params = None
    
    start = time.time()
    for i in range(budget):
        x = np.random.uniform(bounds[0], bounds[1], dim)
        score = func(x)
        evals.append(score)
        
        if score < best_score:
            best_score = score
            best_params = {f"x{j}": x[j] for j in range(dim)}
        
        if verbose:
            params_str = " | ".join([f"x{j}: {x[j]:.4f}" for j in range(min(3, dim))])
            if dim > 3:
                params_str += " | ..."
            print(f"{func_name} | seed {seed} | random trial {i+1} | {params_str} | score: {score:.6f}")
    
    elapsed = time.time() - start
    
    return BenchmarkResult(
        method="random",
        function=func_name,
        seed=seed,
        best_score=best_score,
        best_params=best_params,
        eval_count=len(evals),
        time_seconds=elapsed,
        all_scores=evals
    )


def run_cmaes(func: Callable, dim: int, bounds: tuple,
              budget: int, seed: int, func_name: str, verbose: bool = False) -> BenchmarkResult:
    """Run BIPOP-CMA-ES (pycma) using cma.fmin2 with restarts.

    We optimize on [0,1]^dim and map to the true bounds.
    """
    if not CMA_AVAILABLE:
        raise RuntimeError("pycma (cma) not available")

    evals = []
    best_score = float('inf')
    best_params = None

    # objective in normalized coordinates
    def objective_cma(x):
        nonlocal best_score, best_params
        # clip to [0,1]
        x_norm = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        # map to real bounds
        low, high = bounds
        x_real = low + (high - low) * x_norm
        score = func(x_real)
        evals.append(score)
        if score < best_score:
            best_score = score
            best_params = {f"x{i}": float(x_real[i]) for i in range(dim)}
        if verbose:
            params_str = " | ".join([f"x{i}: {x_real[i]:.4f}" for i in range(min(3, dim))])
            if dim > 3:
                params_str += " | ..."
            print(f"{func_name} | seed {seed} | cma-es eval {len(evals)} | {params_str} | score: {score:.6f}")
        return score

    x0 = [0.5] * dim
    sigma0 = 0.3
    opts = {
        'seed': seed,
        'bounds': [0.0, 1.0],
        'verbose': -9,
        'CMA_active': True,
        'CMA_elitist': True,
        'maxfevals': budget,
    }

    start = time.time()
    # Use fmin2 with bipop=True for BIPOP-CMA-ES with restarts
    res = cma.fmin2(objective_cma, x0, sigma0, opts, bipop=True)
    elapsed = time.time() - start

    # res is a tuple: (xbest, fbest, evaluations, ...) or a dict-like object
    # We already tracked best in objective_cma, so just use that
    if best_params is None:
        best_params = {f"x{i}": 0.0 for i in range(dim)}

    return BenchmarkResult(
        method="cmaes_bipop",
        function=func_name,
        seed=seed,
        best_score=best_score,
        best_params=best_params,
        eval_count=len(evals),
        time_seconds=elapsed,
        all_scores=evals,
    )


# Benchmark function definitions (name -> (function, bounds))
BENCHMARKS = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-2.0, 2.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-32.768, 32.768)),
    "levy": (levy, (-10.0, 10.0)),
    "griewank": (griewank, (-600.0, 600.0))
}


def main():
    # Config fisso: esegue tutti i benchmark senza CLI
    functions = list(BENCHMARKS.keys())
    dim = 30
    budget = 400
    seeds = [42, 43, 44, 45, 46]
    methods = ["curvature", "optuna", "random", "cmaes_bipop"]
    verbose = False
    
    print("=" * 80)
    print("SYNTHETIC BENCHMARK")
    print("=" * 80)
    print(f"Functions: {functions}")
    print(f"Dimension: {dim}D")
    print(f"Budget: {budget} evaluations")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print("=" * 80)
    print()
    
    all_results = []
    
    for func_name in functions:
        func, bounds = BENCHMARKS[func_name]
        print(f"\n{'='*80}")
        print(f"FUNCTION: {func_name} ({dim}D)")
        print(f"{'='*80}")
        
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            for method in methods:
                print(f"\nRunning {method}...")
                
                try:
                    if method == "curvature":
                        result = run_curvature(func, dim, bounds, budget, seed, func_name, verbose)
                    elif method == "optuna":
                        result = run_optuna(func, dim, bounds, budget, seed, func_name, verbose)
                    elif method == "random":
                        result = run_random(func, dim, bounds, budget, seed, func_name, verbose)
                    elif method == "cmaes_bipop":
                        result = run_cmaes(func, dim, bounds, budget, seed, func_name, verbose)
                    
                    all_results.append(result)
                    
                    if not verbose:
                        print(f"{method:12s} | best: {result.best_score:.6f} | time: {result.time_seconds:.2f}s")
                
                except Exception as e:
                    print(f"ERROR running {method}: {e}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for func_name in functions:
        print(f"\n{func_name} ({dim}D):")
        func_results = [r for r in all_results if r.function == func_name]
        
        for method in methods:
            method_results = [r for r in func_results if r.method == method]
            if not method_results:
                continue
            
            scores = [r.best_score for r in method_results]
            times = [r.time_seconds for r in method_results]
            
            print(f"  {method:12s}: "
                  f"best={np.mean(scores):.6f} ± {np.std(scores):.6f}  "
                  f"time={np.mean(times):.2f}s ± {np.std(times):.2f}s")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"/mnt/workspace/tests/benchmark_synthetic_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        f.write("SYNTHETIC BENCHMARK RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Functions: {functions}\n")
        f.write(f"Dimension: {dim}D\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Methods: {methods}\n")
        f.write("="*80 + "\n\n")

        # Write summary statistics to file (same as console)
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")
        for func_name in functions:
            f.write(f"\n{func_name} ({dim}D):\n")
            func_results = [r for r in all_results if r.function == func_name]

            for method in methods:
                method_results = [r for r in func_results if r.method == method]
                if not method_results:
                    continue

                scores = [r.best_score for r in method_results]
                times = [r.time_seconds for r in method_results]

                f.write(
                    f"  {method:12s}: "
                    f"best={np.mean(scores):.6f} ± {np.std(scores):.6f}  "
                    f"time={np.mean(times):.2f}s ± {np.std(times):.2f}s\n"
                )

        f.write("\n\nDETAILED RESULTS\n")
        f.write("="*80 + "\n")

        for result in all_results:
            f.write(f"\n{result.method} | {result.function} | seed {result.seed}\n")
            f.write(f"  Best score: {result.best_score:.6f}\n")
            f.write(f"  Best params: {result.best_params}\n")
            f.write(f"  Evaluations: {result.eval_count}\n")
            f.write(f"  Time: {result.time_seconds:.2f}s\n")
            f.write(f"  All scores (first 10): {result.all_scores[:10]}\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
