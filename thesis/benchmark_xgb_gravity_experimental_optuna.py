#!/usr/bin/env python3
"""
XGBoost Tabular benchmark: alba_experimental vs Optuna vs alba_framework_gravity.

XGBoost has 20 continuous dimensions (no categorical) - tests continuous handling.

Usage:
  # Default: budget 200, seeds 70-73
  python thesis/benchmark_xgb_gravity_experimental_optuna.py

  # Extended run
  python thesis/benchmark_xgb_gravity_experimental_optuna.py --budget 400 --seeds 70-79

Output: JSON to thesis/benchmark_results/
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure local thesis/ is importable.
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework_gravity import ALBA as ALBA_GRAV  # noqa: E402
from ALBA_V1_experimental import ALBA as ALBA_EXP  # noqa: E402

# Import XGBoost benchmark
from benchmark_xgboost_tabular import xgboost_tabular  # noqa: E402


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
    best_accuracy: float
    time_seconds: float = 0.0


def _run_alba_xgb(
    seed: int,
    budget: int,
    alba_cls: Any,
    use_gpu: bool = False,
) -> RunSummary:
    """Run ALBA on XGBoost tabular (20D continuous)."""
    dim = 20
    bounds = [(0.0, 1.0)] * dim

    kwargs: Dict[str, Any] = {
        "bounds": bounds,
        "maximize": True,  # maximize accuracy
        "seed": int(seed),
        "total_budget": int(budget),
        "split_depth_max": 8,
        "global_random_prob": 0.05,
        "stagnation_threshold": 50,
        "categorical_dims": None,  # all continuous
    }

    try:
        opt = alba_cls(**kwargs)
    except TypeError:
        kwargs_min = {
            "bounds": bounds,
            "maximize": True,
            "seed": int(seed),
            "total_budget": int(budget),
            "categorical_dims": None,
        }
        opt = alba_cls(**kwargs_min)

    best_acc = -np.inf
    t0 = time.time()

    for _ in range(int(budget)):
        x = opt.ask()
        metrics = xgboost_tabular(x, use_gpu=use_gpu, trial_seed=seed)
        acc = metrics["accuracy"]
        opt.tell(x, acc)
        best_acc = max(best_acc, acc)

    elapsed = time.time() - t0
    return RunSummary(best_accuracy=best_acc, time_seconds=elapsed)


def _run_optuna_xgb(seed: int, budget: int, use_gpu: bool = False) -> RunSummary:
    """Run Optuna on XGBoost tabular."""
    try:
        import optuna
    except ImportError:
        return RunSummary(best_accuracy=0.0, time_seconds=0.0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        x = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(20)]
        metrics = xgboost_tabular(np.array(x), use_gpu=use_gpu, trial_seed=seed)
        return metrics["accuracy"]

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    elapsed = time.time() - t0

    return RunSummary(best_accuracy=study.best_value, time_seconds=elapsed)


def run_benchmark(
    budget: int,
    seeds: List[int],
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """Run XGBoost benchmark: alba_experimental vs optuna vs alba_framework_gravity."""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"xgb_gravity_exp_optuna_b{budget}_{timestamp}.json")

    results: Dict[str, Any] = {
        "benchmark": "xgboost_tabular",
        "methods": {
            "alba_experimental": {"kind": "ALBA_V1_experimental"},
            "optuna": {"kind": "optuna_TPE"},
            "alba_framework_gravity": {"kind": "alba_framework_gravity"},
        },
        "config": {
            "budget": budget,
            "seeds": seeds,
            "use_gpu": use_gpu,
            "xgb_dim": 20,
            "task": "XGBoost tabular classification (20D continuous)",
        },
        "tasks": {},
    }

    print("=" * 78)
    print(f"XGBoost Tabular benchmark: alba_experimental vs Optuna vs alba_framework_gravity")
    print(f"budget={budget} | seeds={seeds} | out={out_file}")
    print(f"Task: 20D continuous hyperparameters")
    print("=" * 78)
    print()

    task_key = "xgboost_tabular"
    rows: List[Dict[str, Any]] = []

    print(f"== XGBoost Tabular (20D continuous) ==")
    print("seed | alba_experimental |    optuna | alba_framework_gravity | time (s)")

    for seed in seeds:
        # ALBA experimental
        t0 = time.time()
        res_exp = _run_alba_xgb(seed, budget, ALBA_EXP, use_gpu=use_gpu)
        t_exp = time.time() - t0

        # Optuna
        t0 = time.time()
        res_opt = _run_optuna_xgb(seed, budget, use_gpu=use_gpu)
        t_opt = time.time() - t0

        # ALBA gravity
        t0 = time.time()
        res_grav = _run_alba_xgb(seed, budget, ALBA_GRAV, use_gpu=use_gpu)
        t_grav = time.time() - t0

        exp_acc = res_exp.best_accuracy
        opt_acc = res_opt.best_accuracy
        grav_acc = res_grav.best_accuracy

        print(f"  {seed:2d} |          {exp_acc:.6f} |  {opt_acc:.6f} |          {grav_acc:.6f} | {t_exp:.1f}/{t_opt:.1f}/{t_grav:.1f}")

        rows.append({
            "seed": seed,
            "alba_experimental": exp_acc,
            "optuna": opt_acc,
            "alba_framework_gravity": grav_acc,
            "time_exp": t_exp,
            "time_opt": t_opt,
            "time_grav": t_grav,
        })

    # Compute statistics
    exp_scores = [r["alba_experimental"] for r in rows]
    opt_scores = [r["optuna"] for r in rows]
    grav_scores = [r["alba_framework_gravity"] for r in rows]

    m_exp = float(np.mean(exp_scores))
    s_exp = float(np.std(exp_scores))
    m_opt = float(np.mean(opt_scores))
    s_opt = float(np.std(opt_scores))
    m_grav = float(np.mean(grav_scores))
    s_grav = float(np.std(grav_scores))

    # Count wins (higher accuracy is better)
    wins_exp = sum(1 for r in rows if r["alba_experimental"] >= max(r["optuna"], r["alba_framework_gravity"]))
    wins_opt = sum(1 for r in rows if r["optuna"] >= max(r["alba_experimental"], r["alba_framework_gravity"]))
    wins_grav = sum(1 for r in rows if r["alba_framework_gravity"] >= max(r["alba_experimental"], r["optuna"]))

    print(f"  Summary (higher accuracy is better):")
    print(f"  alba_experimental      mean={m_exp:.6f} ± {s_exp:.6f} | wins={wins_exp}/{len(rows)}")
    print(f"  optuna                mean={m_opt:.6f} ± {s_opt:.6f} | wins={wins_opt}/{len(rows)}")
    print(f"  alba_framework_gravity mean={m_grav:.6f} ± {s_grav:.6f} | wins={wins_grav}/{len(rows)}")
    print()

    results["tasks"][task_key] = {
        "seeds": rows,
        "summary": {
            "alba_experimental": {"mean": m_exp, "std": s_exp, "wins": wins_exp},
            "optuna": {"mean": m_opt, "std": s_opt, "wins": wins_opt},
            "alba_framework_gravity": {"mean": m_grav, "std": s_grav, "wins": wins_grav},
        },
    }

    _save_results(out_file, results)
    print(f"✓ Results saved to {out_file}")
    print()

    return results


def main():
    p = argparse.ArgumentParser(
        description="XGBoost Tabular benchmark: alba_experimental vs Optuna vs alba_framework_gravity"
    )
    p.add_argument("--budget", type=int, default=200, help="Optimization budget per seed")
    p.add_argument("--seeds", type=str, default="70-73", help="Seeds (e.g., '70-73' or '70,71,72')")
    p.add_argument("--gpu", action="store_true", help="Use GPU for XGBoost")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [70, 71, 72, 73]

    run_benchmark(budget=args.budget, seeds=seeds, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
