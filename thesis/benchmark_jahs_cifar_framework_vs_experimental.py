#!/usr/bin/env python3
"""
JAHS cifar10 benchmark: alba_framework vs alba_experimental.

Uses the same JAHS interface as benchmark_gravity_gravjump_jahs_paramnet.py.

Usage (requires py39 env):
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_jahs_cifar_framework_vs_experimental.py --budget 400 --seeds 70-79
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Reduce noisy warning spam from dependencies.
warnings.filterwarnings("ignore")

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

# Ensure local thesis/ is importable.
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework import ALBA as ALBA_FRAMEWORK  # noqa: E402
from ALBA_V1_experimental import ALBA as ALBA_EXP  # noqa: E402


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


@dataclass
class RunSummary:
    best: float
    unique_keys: int = 0


def _make_jahs_objective(task: str, *, nepochs: int) -> Callable:
    """Create JAHS objective using same interface as working benchmark."""
    try:
        from jahs_bench import Benchmark  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "jahs_bench not available. Run with conda env py39."
        ) from e

    bench = Benchmark(
        task=str(task),
        kind="surrogate",
        download=True,
        save_dir="/mnt/workspace/jahs_bench_data",
        metrics=["valid-acc"],
    )

    def objective(x_norm: np.ndarray) -> float:
        x_norm = np.asarray(x_norm, dtype=float)
        config = {
            "LearningRate": float(
                10 ** (np.log10(0.001) + float(x_norm[0]) * (np.log10(1.0) - np.log10(0.001)))
            ),
            "WeightDecay": float(
                10 ** (np.log10(1e-5) + float(x_norm[1]) * (np.log10(0.01) - np.log10(1e-5)))
            ),
            "N": [1, 3, 5][int(np.clip(int(float(x_norm[2]) * 2.99), 0, 2))],
            "W": [4, 8, 16][int(np.clip(int(float(x_norm[3]) * 2.99), 0, 2))],
            "Resolution": [0.25, 0.5, 1.0][int(np.clip(int(float(x_norm[4]) * 2.99), 0, 2))],
            "Activation": ["ReLU", "Hardswish", "Mish"][int(np.clip(int(float(x_norm[5]) * 2.99), 0, 2))],
            "TrivialAugment": [True, False][int(np.clip(int(float(x_norm[6]) * 1.99), 0, 1))],
            "Op1": int(np.clip(int(float(x_norm[7]) * 4.99), 0, 4)),
            "Op2": int(np.clip(int(float(x_norm[8]) * 4.99), 0, 4)),
            "Op3": int(np.clip(int(float(x_norm[9]) * 4.99), 0, 4)),
            "Op4": int(np.clip(int(float(x_norm[10]) * 4.99), 0, 4)),
            "Op5": int(np.clip(int(float(x_norm[11]) * 4.99), 0, 4)),
            "Op6": int(np.clip(int(float(x_norm[12]) * 4.99), 0, 4)),
            "Optimizer": "SGD",
            "epoch": int(nepochs),
        }

        result = bench(config)
        last_epoch = max(result.keys())
        valid_acc = float(result[last_epoch]["valid-acc"])
        return 1.0 - valid_acc / 100.0

    return objective


def _cat_key_from_x(x: np.ndarray, *, categorical_dims: List[Tuple[int, int]]) -> Tuple[int, ...]:
    """Extract categorical key from x."""
    key: List[int] = []
    x = np.asarray(x, dtype=float)
    for dim_idx, n_ch in categorical_dims:
        v = float(np.clip(x[dim_idx], 0.0, 1.0))
        idx = int(round(v * float(n_ch - 1)))
        idx = int(np.clip(idx, 0, n_ch - 1))
        key.append(idx)
    return tuple(key)


def _run_alba(
    objective: Callable,
    bounds: List[Tuple[float, float]],
    categorical_dims: List[Tuple[int, int]],
    seed: int,
    budget: int,
    alba_cls: Any,
) -> RunSummary:
    """Run ALBA optimizer."""
    kwargs: Dict[str, Any] = {
        "bounds": bounds,
        "maximize": False,
        "seed": int(seed),
        "total_budget": int(budget),
        "split_depth_max": 8,
        "global_random_prob": 0.05,
        "stagnation_threshold": 50,
        "categorical_dims": categorical_dims,
    }

    try:
        opt = alba_cls(**kwargs)
    except TypeError:
        kwargs_min = {
            "bounds": bounds,
            "maximize": False,
            "seed": int(seed),
            "total_budget": int(budget),
            "categorical_dims": categorical_dims,
        }
        opt = alba_cls(**kwargs_min)

    best = float("inf")
    seen_keys = set()

    for _ in range(int(budget)):
        x = opt.ask()
        y = float(objective(x))
        opt.tell(x, y)
        best = min(best, y)
        seen_keys.add(_cat_key_from_x(np.asarray(x), categorical_dims=categorical_dims))

    return RunSummary(best=best, unique_keys=len(seen_keys))


def run_benchmark(
    budget: int,
    seeds: List[int],
    task: str = "cifar10",
    nepochs: int = 200,
) -> Dict[str, Any]:
    """Run JAHS benchmark: alba_framework vs alba_experimental."""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"jahs_{task}_framework_exp_b{budget}_{timestamp}.json")

    # JAHS: 13 dims, first 2 continuous, rest categorical
    jahs_bounds = [(0.0, 1.0)] * 13
    jahs_cat_dims = [
        (2, 3),   # N
        (3, 3),   # W
        (4, 3),   # Resolution
        (5, 3),   # Activation
        (6, 2),   # TrivialAugment
        (7, 5),   # Op1
        (8, 5),   # Op2
        (9, 5),   # Op3
        (10, 5),  # Op4
        (11, 5),  # Op5
        (12, 5),  # Op6
    ]

    results: Dict[str, Any] = {
        "benchmark": f"jahs_{task}",
        "methods": {
            "alba_framework": {"kind": "alba_framework"},
            "alba_experimental": {"kind": "ALBA_V1_experimental"},
        },
        "config": {
            "budget": budget,
            "seeds": seeds,
            "task": task,
            "nepochs": nepochs,
        },
        "tasks": {},
    }

    print("=" * 78)
    print(f"JAHS {task} benchmark: alba_framework vs alba_experimental")
    print(f"budget={budget} | seeds={seeds} | nepochs={nepochs}")
    print(f"out={out_file}")
    print("=" * 78)
    print()

    # Create objective once
    print(f"Loading JAHS {task} surrogate...")
    objective = _make_jahs_objective(task, nepochs=nepochs)
    print("Loaded. Starting runs...")
    print()

    rows: List[Dict[str, Any]] = []

    print(f"== JAHS {task} (nepochs={nepochs}) ==")

    for seed in seeds:
        # ALBA framework
        t0 = time.time()
        res_framework = _run_alba(objective, jahs_bounds, jahs_cat_dims, seed, budget, ALBA_FRAMEWORK)
        t_framework = time.time() - t0

        # ALBA experimental
        t0 = time.time()
        res_exp = _run_alba(objective, jahs_bounds, jahs_cat_dims, seed, budget, ALBA_EXP)
        t_exp = time.time() - t0

        framework_score = res_framework.best
        exp_score = res_exp.best

        # Optuna TPE on x in [0,1]^13 (objective expects normalized vector)
        optuna_best = float("inf")

        def optuna_objective(trial: optuna.Trial) -> float:
            nonlocal optuna_best
            x = np.array(
                [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(13)],
                dtype=float,
            )
            y = float(objective(x))
            if y < optuna_best:
                optuna_best = y
            return y

        sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=int(budget), show_progress_bar=False)

        print(f"seed {seed}")
        pair = [
            ("alba_framework", float(framework_score)),
            ("alba_experimental", float(exp_score)),
            ("optuna_tpe", float(optuna_best)),
        ]
        pair.sort(key=lambda t: t[1])
        for name, score in pair:
            print(f"  {name} {score:.6f}")

        rows.append({
            "seed": seed,
            "alba_framework": framework_score,
            "alba_experimental": exp_score,
            "optuna_tpe": optuna_best,
            "uniq_framework": res_framework.unique_keys,
            "uniq_exp": res_exp.unique_keys,
            "time_framework": t_framework,
            "time_exp": t_exp,
        })

        # Save incrementally
        results["tasks"][task] = {"seeds": rows}
        _save_results(out_file, results)

    results["tasks"][task] = {
        "seeds": rows,
    }

    _save_results(out_file, results)
    print(f"✓ Results saved to {out_file}")
    print()

    return results


def main():
    p = argparse.ArgumentParser(
        description="JAHS cifar10 benchmark: alba_framework vs alba_experimental"
    )
    p.add_argument("--budget", type=int, default=400, help="Optimization budget per seed")
    p.add_argument("--seeds", type=str, default="70-79", help="Seeds (e.g., '70-79' or '70,71,72')")
    p.add_argument("--task", type=str, default="cifar10", choices=["cifar10", "fashion_mnist", "colorectal_histology"])
    p.add_argument("--nepochs", type=int, default=200, help="JAHS nepochs")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

    run_benchmark(budget=args.budget, seeds=seeds, task=args.task, nepochs=args.nepochs)


if __name__ == "__main__":
    main()
