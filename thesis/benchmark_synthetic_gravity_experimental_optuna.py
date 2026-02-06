#!/usr/bin/env python3
"""
Synthetic functions benchmark: alba_experimental vs Optuna vs alba_framework_gravity.

Tests continuous-only optimization across multiple dimensions and function types.

Usage:
  python3 thesis/benchmark_synthetic_gravity_experimental_optuna.py --budget 200 --seeds 70-73
  python3 thesis/benchmark_synthetic_gravity_experimental_optuna.py --budget 400 --seeds 70-79 --dims 5,10,20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Ensure local thesis/ is importable.
sys.path.insert(0, str(Path(__file__).parent))

from hpo_minimal import HPOptimizer as HPO_MINIMAL  # noqa: E402
from hpo_minimal_improved import HPOptimizer as HPO_IMPROVED  # noqa: E402

# Import synthetic functions
try:
    from ParamSpace import (  # noqa: E402
        sphere, rosenbrock, rastrigin, ackley, levy, griewank,
        schwefel, zakharov, styblinski_tang, michalewicz
    )
except ImportError:
    # Fallback: define functions inline
    def sphere(x):
        return np.sum(x**2)
    
    def rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    def ackley(x):
        n = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e
    
    def levy(x):
        w = 1 + (x - 1) / 4
        return (np.sin(np.pi * w[0])**2 + 
                np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)) +
                (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2))
    
    def griewank(x):
        return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    
    def schwefel(x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def zakharov(x):
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum1 + sum2**2 + sum2**4
    
    def styblinski_tang(x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
    
    michalewicz = None  # Skip if not available

RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"


def _save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _parse_seeds(arg: str) -> List[int]:
    arg = str(arg or "").strip()
    if not arg:
        return []
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_dims(arg: str) -> List[int]:
    arg = str(arg or "").strip()
    if not arg:
        return []
    return sorted(set(int(x.strip()) for x in arg.split(",") if x.strip()))


@dataclass
class RunSummary:
    best_score: float
    time_seconds: float = 0.0


# Synthetic function definitions with bounds
SYNTHETIC_FUNCTIONS: Dict[str, Tuple[Callable, Tuple[float, float]]] = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-5.0, 5.0)),
    "levy": (levy, (-10.0, 10.0)),
    "griewank": (griewank, (-600.0, 600.0)),
    "schwefel": (schwefel, (-500.0, 500.0)),
    "zakharov": (zakharov, (-5.0, 10.0)),
    "styblinski_tang": (styblinski_tang, (-5.0, 5.0)),
}


def _run_hpo_synthetic(
    func: Callable,
    dim: int,
    bounds_range: Tuple[float, float],
    seed: int,
    budget: int,
    hpo_cls: Any,
) -> RunSummary:
    """Run HPO (minimal/improved) on synthetic function."""
    bounds = [bounds_range] * dim

    opt = hpo_cls(bounds=bounds, maximize=False, seed=int(seed))
    
    t0 = time.time()
    best_x, best_score = opt.optimize(func, budget=budget)
    elapsed = time.time() - t0
    
    return RunSummary(best_score=best_score, time_seconds=elapsed)


def _run_optuna_synthetic(
    func: Callable,
    dim: int,
    bounds_range: Tuple[float, float],
    seed: int,
    budget: int,
) -> RunSummary:
    """Run Optuna on synthetic function."""
    try:
        import optuna
    except ImportError:
        return RunSummary(best_score=float("inf"), time_seconds=0.0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        x = [trial.suggest_float(f"x{i}", bounds_range[0], bounds_range[1]) for i in range(dim)]
        return func(np.array(x))

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    elapsed = time.time() - t0

    return RunSummary(best_score=study.best_value, time_seconds=elapsed)


def run_benchmark(
    budget: int,
    seeds: List[int],
    dims: List[int],
) -> Dict[str, Any]:
    """Run synthetic benchmark across multiple functions and dimensions."""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"synthetic_gravity_exp_optuna_b{budget}_{timestamp}.json")

    results: Dict[str, Any] = {
        "benchmark": "synthetic_functions_multidim",
        "methods": {
            "hpo_minimal": {"kind": "hpo_minimal"},
            "optuna": {"kind": "optuna_TPE"},
            "hpo_improved": {"kind": "hpo_minimal_improved"},
        },
        "config": {
            "budget": budget,
            "seeds": seeds,
            "dimensions": dims,
            "functions": list(SYNTHETIC_FUNCTIONS.keys()),
        },
        "tasks": {},
    }

    print("=" * 78)
    print(f"Synthetic Functions benchmark: hpo_minimal vs Optuna vs hpo_improved")
    print(f"budget={budget} | seeds={seeds} | dimensions={dims}")
    print(f"out={out_file}")
    print("=" * 78)
    print()

    # Run each function at each dimensionality
    for func_name, (func, bounds_range) in SYNTHETIC_FUNCTIONS.items():
        for dim in dims:
            task_key = f"{func_name}_{dim}D"
            rows: List[Dict[str, Any]] = []

            print(f"== {func_name} ({dim}D) bounds={bounds_range} ==")
            print("seed | hpo_minimal |    optuna | hpo_improved | time (s)")

            for seed in seeds:
                # HPO minimal
                res_min = _run_hpo_synthetic(func, dim, bounds_range, seed, budget, HPO_MINIMAL)
                
                # Optuna
                res_opt = _run_optuna_synthetic(func, dim, bounds_range, seed, budget)
                
                # HPO improved
                res_imp = _run_hpo_synthetic(func, dim, bounds_range, seed, budget, HPO_IMPROVED)

                min_score = res_min.best_score
                opt_score = res_opt.best_score
                imp_score = res_imp.best_score

                print(f"  {seed:2d} |  {min_score:11.6e} | {opt_score:9.6e} |  {imp_score:11.6e} | "
                      f"{res_min.time_seconds:.1f}/{res_opt.time_seconds:.1f}/{res_imp.time_seconds:.1f}")

                rows.append({
                    "seed": seed,
                    "hpo_minimal": min_score,
                    "optuna": opt_score,
                    "hpo_improved": imp_score,
                    "time_min": res_min.time_seconds,
                    "time_opt": res_opt.time_seconds,
                    "time_imp": res_imp.time_seconds,
                })

            # Compute statistics
            min_scores = [r["hpo_minimal"] for r in rows]
            opt_scores = [r["optuna"] for r in rows]
            imp_scores = [r["hpo_improved"] for r in rows]

            m_min = float(np.mean(min_scores))
            s_min = float(np.std(min_scores))
            m_opt = float(np.mean(opt_scores))
            s_opt = float(np.std(opt_scores))
            m_imp = float(np.mean(imp_scores))
            s_imp = float(np.std(imp_scores))

            # Count wins (lower is better for minimization)
            wins_min = sum(1 for r in rows if r["hpo_minimal"] <= min(r["optuna"], r["hpo_improved"]))
            wins_opt = sum(1 for r in rows if r["optuna"] <= min(r["hpo_minimal"], r["hpo_improved"]))
            wins_imp = sum(1 for r in rows if r["hpo_improved"] <= min(r["hpo_minimal"], r["optuna"]))

            print(f"  Summary (lower is better):")
            print(f"  hpo_minimal      mean={m_min:.6e} ± {s_min:.6e} | wins={wins_min}/{len(rows)}")
            print(f"  optuna          mean={m_opt:.6e} ± {s_opt:.6e} | wins={wins_opt}/{len(rows)}")
            print(f"  hpo_improved    mean={m_imp:.6e} ± {s_imp:.6e} | wins={wins_imp}/{len(rows)}")
            print()

            results["tasks"][task_key] = {
                "function": func_name,
                "dimension": dim,
                "bounds": bounds_range,
                "seeds": rows,
                "summary": {
                    "hpo_minimal": {"mean": m_min, "std": s_min, "wins": wins_min},
                    "optuna": {"mean": m_opt, "std": s_opt, "wins": wins_opt},
                    "hpo_improved": {"mean": m_imp, "std": s_imp, "wins": wins_imp},
                },
            }

            _save_results(out_file, results)

    print(f"✓ Results saved to {out_file}")
    print()

    # Print overall summary
    print("=" * 78)
    print("OVERALL SUMMARY")
    print("=" * 78)
    
    total_wins_min = sum(t["summary"]["hpo_minimal"]["wins"] for t in results["tasks"].values())
    total_wins_opt = sum(t["summary"]["optuna"]["wins"] for t in results["tasks"].values())
    total_wins_imp = sum(t["summary"]["hpo_improved"]["wins"] for t in results["tasks"].values())
    total_tasks = len(results["tasks"])
    
    print(f"Total wins across {total_tasks} tasks:")
    print(f"  hpo_minimal:      {total_wins_min}")
    print(f"  optuna:          {total_wins_opt}")
    print(f"  hpo_improved:    {total_wins_imp}")
    print()

    return results


def main():
    p = argparse.ArgumentParser(
        description="Synthetic functions benchmark: hpo_minimal vs Optuna vs hpo_improved"
    )
    p.add_argument("--budget", type=int, default=200, help="Optimization budget per seed")
    p.add_argument("--seeds", type=str, default="70-73", help="Seeds (e.g., '70-73' or '70,71,72')")
    p.add_argument("--dims", type=str, default="5,10,20", help="Dimensions to test (e.g., '5,10,20')")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [70, 71, 72, 73]

    dims = _parse_dims(args.dims)
    if not dims:
        dims = [5, 10, 20]

    run_benchmark(budget=args.budget, seeds=seeds, dims=dims)


if __name__ == "__main__":
    main()
