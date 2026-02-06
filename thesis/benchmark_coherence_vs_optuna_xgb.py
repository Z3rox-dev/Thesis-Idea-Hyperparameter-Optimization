#!/usr/bin/env python3
"""
XGBoost Tabular benchmark: ALBA Coherence vs Optuna TPE.

XGBoost has 20 continuous dimensions (no categorical) - tests continuous handling.

Usage:
  python thesis/benchmark_coherence_vs_optuna_xgb.py --budget 400 --seeds 0-9

Output: JSON to thesis/benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure local thesis/ is importable.
sys.path.insert(0, str(Path(__file__).parent))

from alba_framework_coherence import ALBA as ALBA_COH  # noqa: E402

# Import XGBoost benchmark
from benchmark_xgboost_tabular import xgboost_tabular  # noqa: E402

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_checkpoints(arg: str, budget: int) -> List[int]:
    cps = [int(x.strip()) for x in arg.split(",") if x.strip()]
    cps = sorted(set(c for c in cps if 1 <= c <= budget))
    if budget not in cps:
        cps.append(budget)
    return cps


def run_coherence(seed: int, budget: int, checkpoints: List[int], use_gpu: bool = False) -> Dict[str, float]:
    """Run ALBA Coherence on XGBoost tabular (20D continuous)."""
    dim = 20
    bounds = [(0.0, 1.0)] * dim

    opt = ALBA_COH(
        bounds=bounds,
        maximize=True,  # maximize accuracy
        seed=int(seed),
        total_budget=int(budget),
        use_coherence_gating=True,
    )

    best_acc = -np.inf
    cp_results: Dict[str, float] = {}

    for it in range(int(budget)):
        x = opt.ask()
        metrics = xgboost_tabular(x, use_gpu=use_gpu, trial_seed=seed)
        acc = metrics["accuracy"]
        opt.tell(x, acc)
        best_acc = max(best_acc, acc)
        if (it + 1) in checkpoints:
            cp_results[str(it + 1)] = float(best_acc)

    return cp_results


def run_optuna(seed: int, budget: int, checkpoints: List[int], use_gpu: bool = False) -> Dict[str, float]:
    """Run Optuna TPE on XGBoost tabular."""
    best_acc = -np.inf
    cp_results: Dict[str, float] = {}
    trial_count = [0]

    def objective(trial):
        nonlocal best_acc
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(20)])
        metrics = xgboost_tabular(x, use_gpu=use_gpu, trial_seed=seed)
        acc = metrics["accuracy"]
        best_acc = max(best_acc, acc)
        trial_count[0] += 1
        if trial_count[0] in checkpoints:
            cp_results[str(trial_count[0])] = float(best_acc)
        return acc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    if str(budget) not in cp_results:
        cp_results[str(budget)] = float(best_acc)

    return cp_results


def main() -> int:
    p = argparse.ArgumentParser(description="Coherence vs Optuna on XGBoost Tabular")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--checkpoints", default="100,200,400")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--use_gpu", action="store_true")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    checkpoints = _parse_checkpoints(args.checkpoints, args.budget)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/coherence_vs_optuna_xgb_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json"

    results: Dict[str, Any] = {
        "config": {
            "budget": args.budget,
            "checkpoints": checkpoints,
            "seeds": seeds,
            "task": "XGBoost tabular classification (20D continuous)",
            "timestamp": timestamp,
        },
        "runs": {"coherence": {}, "optuna": {}},
    }
    _save_results(out_path, results)

    print("=" * 80)
    print(f"XGBoost BENCHMARK | budget={args.budget} | seeds={seeds}")
    print(f"Task: 20D continuous hyperparameters")
    print(f"checkpoints={checkpoints}")
    print(f"save={out_path}")
    print("=" * 80)

    total_wins_coh = 0
    total_wins_opt = 0
    total_ties = 0

    for seed in seeds:
        t0 = time.time()
        try:
            cp_coh = run_coherence(seed, args.budget, checkpoints, args.use_gpu)
            cp_opt = run_optuna(seed, args.budget, checkpoints, args.use_gpu)

            results["runs"]["coherence"][str(seed)] = cp_coh
            results["runs"]["optuna"][str(seed)] = cp_opt
            _save_results(out_path, results)

            final_coh = cp_coh.get(str(args.budget), 0.0)
            final_opt = cp_opt.get(str(args.budget), 0.0)

            # maximize accuracy: higher is better
            if final_coh - 1e-8 > final_opt:
                winner = "COH"
                total_wins_coh += 1
            elif final_opt - 1e-8 > final_coh:
                winner = "OPT"
                total_wins_opt += 1
            else:
                winner = "TIE"
                total_ties += 1

            elapsed = time.time() - t0
            print(f"seed={seed:2d} | COH={final_coh:.6f} OPT={final_opt:.6f} -> {winner} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"seed={seed:2d} | ERROR: {e}")

    total = total_wins_coh + total_wins_opt + total_ties
    winrate_coh = total_wins_coh / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"FINAL: Coherence wins={total_wins_coh}, Optuna wins={total_wins_opt}, ties={total_ties}")
    print(f"Coherence winrate: {winrate_coh:.1%}")
    print(f"Saved: {out_path}")

    results["summary"] = {
        "coherence_wins": total_wins_coh,
        "optuna_wins": total_wins_opt,
        "ties": total_ties,
        "coherence_winrate": winrate_coh,
    }
    _save_results(out_path, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
