#!/usr/bin/env python3
"""
Benchmark v5s_more_novelty vs HPO-Minimal vs Optuna on synthetic functions.
Structured like benchmark_synthetic, but focused on the v5-simple variant.
"""

import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List

# Synthetic functions
from synthetic_functions import (
    sphere,
    rosenbrock,
    rastrigin,
    ackley,
    levy,
    griewank,
    schwefel,
    zakharov,
    styblinski_tang,
    michalewicz,
    dixon_price,
    alpine1,
)

# Optimizers
from hpo_minimal import HPOptimizer as HPOptimizer_Minimal
from thesis.hpo_lgs_v5_simple import HPOptimizer as HPO_v5s

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")


@dataclass
class RunResult:
    method: str
    function: str
    dim: int
    seed: int
    best_score: float
    time_seconds: float


def run_v5s(func: Callable, dim: int, bounds: tuple, budget: int, seed: int, verbose: bool = False) -> RunResult:
    bounds_list = [(bounds[0], bounds[1]) for _ in range(dim)]
    hpo = HPO_v5s(
        bounds=bounds_list,
        maximize=False,
        seed=seed,
        total_budget=budget,
        n_candidates=25,
        novelty_weight=0.4,
    )
    best = float("inf")

    start = time.time()
    for _ in range(budget):
        x = hpo.ask()
        y = float(func(np.array(x)))
        hpo.tell(x, y)
        if y < best:
            best = y
        if verbose:
            print(f"v5s | seed {seed} | y={y:.6f}")
    elapsed = time.time() - start

    return RunResult("v5s_more_novelty", func.__name__, dim, seed, best, elapsed)


def run_minimal(func: Callable, dim: int, bounds: tuple, budget: int, seed: int, verbose: bool = False) -> RunResult:
    bounds_list = [(bounds[0], bounds[1]) for _ in range(dim)]
    evals: List[float] = []

    def objective(x: np.ndarray) -> float:
        y = float(func(x))
        evals.append(y)
        if verbose:
            print(f"minimal | seed {seed} | y={y:.6f}")
        return -y  # minimal maximizes

    start = time.time()
    hpo = HPOptimizer_Minimal(bounds=bounds_list, maximize=True, seed=seed)
    _, best_neg = hpo.optimize(objective, budget=budget)
    elapsed = time.time() - start

    return RunResult("hpo_minimal", func.__name__, dim, seed, -best_neg, elapsed)


def run_optuna(func: Callable, dim: int, bounds: tuple, budget: int, seed: int, verbose: bool = False) -> RunResult:
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available")

    def objective(trial: "optuna.trial.Trial") -> float:
        x = np.array([trial.suggest_float(f"x{i}", bounds[0], bounds[1]) for i in range(dim)])
        y = float(func(x))
        if verbose:
            print(f"optuna | seed {seed} | y={y:.6f}")
        return y

    start = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    elapsed = time.time() - start

    return RunResult("optuna", func.__name__, dim, seed, study.best_value, elapsed)


# Benchmark catalog (aligns with benchmark_synthetic)
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
    "alpine1": (alpine1, (0.0, 10.0)),
}


def main() -> None:
    functions = list(BENCHMARKS.keys())
    dims = [15]
    budget = 150
    seeds = [87]
    methods = ["v5s_more_novelty", "hpo_minimal", "optuna"]
    verbose = False

    print("=" * 80)
    print("SYNTHETIC BENCHMARK: v5s_more_novelty vs hpo_minimal vs optuna")
    print("=" * 80)
    print(f"Functions: {functions}")
    print(f"Dimensions: {dims}D")
    print(f"Budget: {budget} evaluations")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print("=" * 80)
    print()

    all_results: List[RunResult] = []

    for dim in dims:
        print(f"\n{'#' * 80}")
        print(f"RUNNING DIMENSION: {dim}D")
        print(f"{'#' * 80}")

        for func_name in functions:
            func, bounds = BENCHMARKS[func_name]
            print(f"\n{'=' * 80}")
            print(f"FUNCTION: {func_name} ({dim}D)")
            print(f"{'=' * 80}")

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")

                for method in methods:
                    print(f"\nRunning {method}...")

                    try:
                        if method == "v5s_more_novelty":
                            res = run_v5s(func, dim, bounds, budget, seed, verbose)
                        elif method == "hpo_minimal":
                            res = run_minimal(func, dim, bounds, budget, seed, verbose)
                        elif method == "optuna":
                            res = run_optuna(func, dim, bounds, budget, seed, verbose)
                        else:
                            continue

                        all_results.append(res)
                        print(f"{method:16s} | best: {res.best_score:.6f} | time: {res.time_seconds:.2f}s")

                    except Exception as exc:
                        print(f"ERROR running {method}: {exc}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for dim in dims:
        print(f"\n--- DIMENSION {dim}D ---")
        for func_name in functions:
            print(f"\n{func_name} ({dim}D):")
            func_results = [r for r in all_results if r.function == func_name and r.dim == dim]

            for method in methods:
                m_results = [r for r in func_results if r.method == method]
                if not m_results:
                    continue

                scores = [r.best_score for r in m_results]
                times = [r.time_seconds for r in m_results]
                print(
                    f"  {method:16s}: best={np.mean(scores):.6f} +/- {np.std(scores):.6f}  "
                    f"time={np.mean(times):.2f}s +/- {np.std(times):.2f}s"
                )

    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = "/mnt/workspace/thesis/results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"benchmark_v5s_vs_minimal_optuna_{timestamp}.txt")

    with open(output_file, "w") as f:
        f.write("SYNTHETIC BENCHMARK: v5s_more_novelty vs hpo_minimal vs optuna\n")
        f.write("=" * 80 + "\n")
        f.write(f"Functions: {functions}\n")
        f.write(f"Dimensions: {dims}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Methods: {methods}\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        for dim in dims:
            f.write(f"\n--- DIMENSION {dim}D ---\n")
            for func_name in functions:
                f.write(f"\n{func_name} ({dim}D):\n")
                func_results = [r for r in all_results if r.function == func_name and r.dim == dim]
                for method in methods:
                    m_results = [r for r in func_results if r.method == method]
                    if not m_results:
                        continue
                    scores = [r.best_score for r in m_results]
                    times = [r.time_seconds for r in m_results]
                    f.write(
                        f"  {method:16s}: best={np.mean(scores):.6f} +/- {np.std(scores):.6f}  "
                        f"time={np.mean(times):.2f}s +/- {np.std(times):.2f}s\n"
                    )

        f.write("\n\nDETAILED RESULTS\n")
        f.write("=" * 80 + "\n")
        for r in all_results:
            f.write(
                f"{r.method} | {r.function} | {r.dim}D | seed {r.seed} | best={r.best_score:.6f} | "
                f"time={r.time_seconds:.2f}s\n"
            )

    print(f"\nDetailed results saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
