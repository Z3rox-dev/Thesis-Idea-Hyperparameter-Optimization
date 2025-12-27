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

# Ensure local imports work regardless of CWD
THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(WORKSPACE_ROOT))

# Import synthetic functions
try:
    from synthetic_functions import (
        sphere, rosenbrock, rastrigin, ackley, levy, griewank,
        schwefel, zakharov, styblinski_tang, michalewicz, dixon_price, alpine1
    )
except Exception:
    from ParamSpace import (
        sphere, rosenbrock, rastrigin, ackley, levy, griewank,
        schwefel, zakharov, styblinski_tang, michalewicz, dixon_price, alpine1
    )

# Import optimizers
from hpo_debug import QuadHPO as QuadHPO_Debug
from hpo_minimal import HPOptimizer as HPOptimizer_Minimal

try:
    from hpo_v5s_more_novelty_standalone import HPOptimizerV5s as HPOptimizer_V5
    V5_AVAILABLE = True
except Exception:
    V5_AVAILABLE = False
    print("Warning: V5 optimizer not available")

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


def run_curvature_generic(hpo_class, func: Callable, dim: int, bounds: tuple, 
                          budget: int, seed: int, func_name: str, verbose: bool = False, method_name: str = "curvature") -> BenchmarkResult:
    """Run QuadHPO (generic) with curvature-based splitting."""
    np.random.seed(seed)
    
    # IMPORTANT: QuadHPO has a bug with non-unit bounds
    # Use [0,1] bounds and handle conversion in objective
    bounds_normalized = [(0.0, 1.0) for _ in range(dim)]
    
    # Objective wrapper
    evals = []
    
    # Run QuadHPO with normalized [0,1] bounds
    start = time.time()
    hpo = hpo_class(
        bounds=bounds_normalized,
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
            print(f"{func_name} | seed {seed} | {method_name} trial {trial_idx} | {params_str} | score: {score:.6f}")
        return score
    
    # Run optimization
    hpo.optimize(objective_wrapper, budget=budget)
    
    elapsed = time.time() - start
    
    # Extract best parameters - QuadHPO negates scores for minimization
    # so best_score_global is actually -min(scores)
    if hasattr(hpo, 'best_score_global'):
        # Since maximize=False, QuadHPO stored -score, so negate again to get true minimum
        best_score = -hpo.best_score_global if evals else float('inf')
    else:
        best_score = min(evals) if evals else float('inf')
    
    # Handle different attribute names for best solution
    print(f"DEBUG: hpo type: {type(hpo)}")
    print(f"DEBUG: dir(hpo): {dir(hpo)}")
    if hasattr(hpo, 'best_x_norm') and hpo.best_x_norm is not None:
        x_best_norm = hpo.best_x_norm
    elif hasattr(hpo, 'best_x_candidate') and hpo.best_x_candidate is not None:
        x_best_norm = hpo.best_x_candidate
    else:
        x_best_norm = None

    if x_best_norm:
        x_best = np.array([bounds[0] + (bounds[1] - bounds[0]) * xi for xi in x_best_norm])
        best_params = {f"x{i}": x_best[i] for i in range(dim)}
    else:
        best_params = {f"x{i}": 0.0 for i in range(dim)}
    
    return BenchmarkResult(
        method=method_name,
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


def run_minimal(func: Callable, dim: int, bounds: tuple,
                budget: int, seed: int, func_name: str, verbose: bool = False) -> BenchmarkResult:
    """Run HPOptimizer Minimal."""
    evals = []
    
    # Create bounds list for minimal optimizer
    bounds_list = [(bounds[0], bounds[1]) for _ in range(dim)]
    
    def objective_wrapper(x):
        score = func(x)
        evals.append(score)
        if verbose:
            params_str = " | ".join([f"x{i}: {x[i]:.4f}" for i in range(min(3, dim))])
            if dim > 3:
                params_str += " | ..."
            print(f"{func_name} | seed {seed} | minimal trial {len(evals)} | {params_str} | score: {score:.6f}")
        return -score  # HPOptimizer maximizes, so negate for minimization
    
    start = time.time()
    hpo = HPOptimizer_Minimal(bounds=bounds_list, maximize=True, seed=seed)
    best_x, best_score_neg = hpo.optimize(objective_wrapper, budget=budget)
    elapsed = time.time() - start
    
    best_score = -best_score_neg  # Convert back to minimization score
    best_params = {f"x{i}": best_x[i] for i in range(dim)} if best_x is not None else {}
    
    return BenchmarkResult(
        method="minimal",
        function=func_name,
        seed=seed,
        best_score=best_score,
        best_params=best_params,
        eval_count=len(evals),
        time_seconds=elapsed,
        all_scores=[-s for s in evals]  # Convert back all scores
    )


def run_v5(func: Callable, dim: int, bounds: tuple,
           budget: int, seed: int, func_name: str, verbose: bool = False) -> BenchmarkResult:
    """Run HPOptimizer V5 (currently mapped to HPOptimizerV5s)."""
    if not V5_AVAILABLE:
        raise RuntimeError("V5 optimizer not available")

    evals = []
    bounds_list = [(bounds[0], bounds[1]) for _ in range(dim)]

    def objective_wrapper(x: np.ndarray) -> float:
        score = func(x)
        evals.append(score)
        if verbose:
            params_str = " | ".join([f"x{i}: {x[i]:.4f}" for i in range(min(3, dim))])
            if dim > 3:
                params_str += " | ..."
            print(f"{func_name} | seed {seed} | v5 trial {len(evals)} | {params_str} | score: {score:.6f}")
        return score

    start = time.time()
    hpo = HPOptimizer_V5(bounds=bounds_list, maximize=False, seed=seed, total_budget=budget)
    best_x, best_score = hpo.optimize(objective_wrapper, budget=budget)
    elapsed = time.time() - start

    best_params = {f"x{i}": best_x[i] for i in range(dim)} if best_x is not None else {}

    return BenchmarkResult(
        method="v5",
        function=func_name,
        seed=seed,
        best_score=float(best_score),
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
    "griewank": (griewank, (-600.0, 600.0)),
    "schwefel": (schwefel, (-500.0, 500.0)),
    "zakharov": (zakharov, (-5.0, 10.0)),
    "styblinski_tang": (styblinski_tang, (-5.0, 5.0)),
    "michalewicz": (michalewicz, (0.0, np.pi)),
    "dixon_price": (dixon_price, (-10.0, 10.0)),
    "alpine1": (alpine1, (0.0, 10.0))
}

def main():
    # Config fisso: esegue tutti i benchmark senza CLI
    functions = list(BENCHMARKS.keys())
    dims = [6, 3]
    budget = 150
    seeds = [87]
    methods = ["v5", "optuna", "random"]
    verbose = False
    
    print("=" * 80)
    print("SYNTHETIC BENCHMARK")
    print("=" * 80)
    print(f"Functions: {functions}")
    print(f"Dimensions: {dims}D")
    print(f"Budget: {budget} evaluations")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print("=" * 80)
    print()
    
    all_results = []
    
    for dim in dims:
        print(f"\n{'#'*80}")
        print(f"RUNNING DIMENSION: {dim}D")
        print(f"{'#'*80}")
        
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
                        if method == "v5":
                            result = run_v5(func, dim, bounds, budget, seed, func_name, verbose)
                        elif method == "optuna":
                            result = run_optuna(func, dim, bounds, budget, seed, func_name, verbose)
                        elif method == "random":
                            result = run_random(func, dim, bounds, budget, seed, func_name, verbose)
                        
                        # Add dimension to result for filtering later
                        result.dim = dim
                        all_results.append(result)
                        
                        if not verbose:
                            print(f"{method:12s} | best: {result.best_score:.6f} | time: {result.time_seconds:.2f}s")
                    
                    except Exception as e:
                        print(f"ERROR running {method}: {e}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for dim in dims:
        print(f"\n--- DIMENSION {dim}D ---")
        for func_name in functions:
            print(f"\n{func_name} ({dim}D):")
            func_results = [r for r in all_results if r.function == func_name and getattr(r, 'dim', 6) == dim]
            
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
    output_file = f"/mnt/workspace/thesis/results/benchmark_synthetic_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        f.write("SYNTHETIC BENCHMARK RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Functions: {functions}\n")
        f.write(f"Dimensions: {dims}D\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Methods: {methods}\n")
        f.write("="*80 + "\n\n")

        # Write summary statistics to file (same as console)
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")
        for dim in dims:
            f.write(f"\n--- DIMENSION {dim}D ---\n")
            for func_name in functions:
                f.write(f"\n{func_name} ({dim}D):\n")
                func_results = [r for r in all_results if r.function == func_name and getattr(r, 'dim', 6) == dim]

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
            dim_val = getattr(result, 'dim', 'unknown')
            f.write(f"\n{result.method} | {result.function} | {dim_val}D | seed {result.seed}\n")
            f.write(f"  Best score: {result.best_score:.6f}\n")
            f.write(f"  Best params: {result.best_params}\n")
            f.write(f"  Evaluations: {result.eval_count}\n")
            f.write(f"  Time: {result.time_seconds:.2f}s\n")
            f.write(f"  All scores (first 10): {result.all_scores[:10]}\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
