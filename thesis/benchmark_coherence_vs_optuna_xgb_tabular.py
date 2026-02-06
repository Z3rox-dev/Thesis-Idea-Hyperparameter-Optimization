#!/usr/bin/env python3
"""
Benchmark: ALBA Coherence vs Optuna TPE on HPOBench XGBoost Tabular (surrogate).

Uses TabularBenchmark with model='xgb' - fast surrogate lookup.
Runs on all locally available task_ids.

Example:
  python thesis/benchmark_coherence_vs_optuna_xgb_tabular.py --budget 400 --seeds 0-9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Compatibilità numpy
if not hasattr(np, "float"):
    np.float = float  # type: ignore
if not hasattr(np, "int"):
    np.int = int  # type: ignore
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore

# HPOBench (vendored)
sys.path.insert(0, "/mnt/workspace/HPOBench")

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, "/mnt/workspace/thesis")
from alba_framework_coherence import ALBA as ALBA_COH

RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"


def _enable_fast_tabular_lookup(benchmark: TabularBenchmark) -> bool:
    """Speed up TabularBenchmark with indexed lookup."""
    try:
        import pandas as pd
        import types
    except Exception:
        return False

    df = getattr(benchmark, "table", None)
    if df is None or "result" not in df.columns:
        return False

    key_cols = [c for c in df.columns if c != "result"]
    if not key_cols:
        return False

    indexed = df.set_index(key_cols, drop=False)

    def _fast_search_dataframe(self, row_dict, _df_ignored):
        key = tuple(row_dict[c] for c in key_cols)
        row = indexed.loc[key]
        if isinstance(row, pd.Series):
            return row["result"]
        return row.iloc[0]["result"]

    benchmark._search_dataframe = types.MethodType(_fast_search_dataframe, benchmark)
    return True


def _parse_seeds(arg: str) -> List[int]:
    arg = arg.strip()
    if "-" in arg and "," not in arg:
        a, b = arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_checkpoints(arg: str, budget: int) -> List[int]:
    cps = [int(x.strip()) for x in arg.split(",") if x.strip()]
    cps = sorted(set(c for c in cps if 1 <= c <= budget))
    if budget not in cps:
        cps.append(budget)
    return cps


def _available_task_ids(model: str = "xgb") -> List[int]:
    """Return available task_ids for HPOBench TabularBenchmark."""
    try:
        import hpobench
    except Exception:
        return []

    base_dir = Path(hpobench.config_file.data_dir) / "TabularData"
    model_dir = base_dir / model
    if not model_dir.exists():
        return []

    task_ids: List[int] = []
    for child in model_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        tid = int(child.name)
        parquet = child / f"{model}_{tid}_data.parquet.gzip"
        meta = child / f"{model}_{tid}_metadata.json"
        if parquet.exists() and meta.exists():
            task_ids.append(tid)
    return sorted(task_ids)


def _x_to_config(x: np.ndarray, hp_names: List[str], hp_seqs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Map x in [0,1]^d to a discrete config dict."""
    cfg: Dict[str, Any] = {}
    for i, name in enumerate(hp_names):
        seq = hp_seqs[name]
        k = len(seq)
        if k <= 1:
            cfg[name] = seq[0]
            continue
        idx = int(np.floor(float(x[i]) * k))
        idx = max(0, min(k - 1, idx))
        cfg[name] = seq[idx]
    return cfg


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_coherence(
    benchmark: TabularBenchmark,
    hp_names: List[str],
    hp_seqs: Dict[str, List[Any]],
    max_fidelity: dict,
    seed: int,
    budget: int,
    checkpoints: List[int],
) -> Dict[str, float]:
    dim = len(hp_names)
    categorical_dims: List[Tuple[int, int]] = [(i, len(hp_seqs[name])) for i, name in enumerate(hp_names)]

    opt = ALBA_COH(
        bounds=[(0.0, 1.0)] * dim,
        categorical_dims=categorical_dims,
        maximize=False,  # minimize error
        seed=seed,
        total_budget=budget,
        use_coherence_gating=True,
    )

    best = float("inf")
    cp_results: Dict[str, float] = {}

    for it in range(budget):
        x = opt.ask()
        cfg = _x_to_config(x, hp_names, hp_seqs)
        res = benchmark.objective_function(configuration=cfg, fidelity=max_fidelity)
        y = float(res["function_value"])
        opt.tell(x, y)
        best = min(best, y)
        if (it + 1) in checkpoints:
            cp_results[str(it + 1)] = float(best)

    return cp_results


def run_optuna(
    benchmark: TabularBenchmark,
    hp_names: List[str],
    hp_seqs: Dict[str, List[Any]],
    max_fidelity: dict,
    seed: int,
    budget: int,
    checkpoints: List[int],
) -> Dict[str, float]:
    best = float("inf")
    cp_results: Dict[str, float] = {}
    trial_count = [0]

    def objective(trial):
        nonlocal best
        cfg = {}
        for name in hp_names:
            seq = hp_seqs[name]
            cfg[name] = trial.suggest_categorical(name, seq)
        res = benchmark.objective_function(configuration=cfg, fidelity=max_fidelity)
        y = float(res["function_value"])
        best = min(best, y)
        trial_count[0] += 1
        if trial_count[0] in checkpoints:
            cp_results[str(trial_count[0])] = float(best)
        return y

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    if str(budget) not in cp_results:
        cp_results[str(budget)] = float(best)

    return cp_results


def main() -> int:
    p = argparse.ArgumentParser(description="Coherence vs Optuna on XGB Tabular (surrogate)")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--checkpoints", default="100,200,400")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--task_ids", default="", help="Comma-separated task IDs; default=all available")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    checkpoints = _parse_checkpoints(args.checkpoints, args.budget)

    if args.task_ids.strip():
        task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = _available_task_ids(model="xgb")

    if not task_ids:
        raise SystemExit("No HPOBench xgb/tabular task_ids found on disk.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/coherence_vs_optuna_xgb_tabular_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json"

    results: Dict[str, Any] = {
        "config": {
            "budget": args.budget,
            "checkpoints": checkpoints,
            "seeds": seeds,
            "task_ids": task_ids,
            "timestamp": timestamp,
        },
        "tasks": {},
    }
    save_results(out_path, results)

    print(f"XGB TABULAR BATTERY | budget={args.budget} | seeds={seeds}")
    print(f"task_ids={task_ids}")
    print(f"checkpoints={checkpoints}")
    print(f"save={out_path}")
    print("=" * 80)

    total_wins_coh = 0
    total_wins_opt = 0
    total_ties = 0

    for task_id in task_ids:
        benchmark = TabularBenchmark(model="xgb", task_id=int(task_id), rng=42)
        fast_ok = _enable_fast_tabular_lookup(benchmark)
        print(f"task {task_id}: fast_lookup={'ON' if fast_ok else 'OFF'}")

        cs = benchmark.get_configuration_space(seed=42)
        hps = list(cs.values())
        hp_names = [hp.name for hp in hps]
        hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
        max_fidelity = benchmark.get_max_fidelity()

        results["tasks"].setdefault(str(task_id), {"coherence": {}, "optuna": {}})
        wins_coh = 0
        wins_opt = 0
        ties = 0

        for seed in seeds:
            t0 = time.time()
            try:
                cp_coh = run_coherence(benchmark, hp_names, hp_seqs, max_fidelity, seed, args.budget, checkpoints)
                cp_opt = run_optuna(benchmark, hp_names, hp_seqs, max_fidelity, seed, args.budget, checkpoints)

                results["tasks"][str(task_id)]["coherence"][str(seed)] = cp_coh
                results["tasks"][str(task_id)]["optuna"][str(seed)] = cp_opt
                save_results(out_path, results)

                final_coh = cp_coh.get(str(args.budget), float("inf"))
                final_opt = cp_opt.get(str(args.budget), float("inf"))

                if final_coh + 1e-8 < final_opt:
                    winner = "COH"
                    wins_coh += 1
                elif final_opt + 1e-8 < final_coh:
                    winner = "OPT"
                    wins_opt += 1
                else:
                    winner = "TIE"
                    ties += 1

                elapsed = time.time() - t0
                print(f"task {task_id} seed={seed:2d} | COH={final_coh:.6f} OPT={final_opt:.6f} -> {winner} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"task {task_id} seed={seed:2d} | ERROR: {e}")

        total_wins_coh += wins_coh
        total_wins_opt += wins_opt
        total_ties += ties
        print(f"task {task_id} SUMMARY: COH wins={wins_coh}, OPT wins={wins_opt}, ties={ties}")
        print("-" * 80)

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
    save_results(out_path, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
