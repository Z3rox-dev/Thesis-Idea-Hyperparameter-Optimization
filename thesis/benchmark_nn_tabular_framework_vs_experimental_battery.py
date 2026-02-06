#!/usr/bin/env python3
"""HPOBench NN-Tabular battery: alba_framework_grid (param_space) vs ALBA_V1_experimental.

Runs on all locally available HPOBench TabularBenchmark datasets for model='nn'
(task IDs present on disk), with a fixed evaluation budget per (task_id, seed).

Intended env: conda `py39`.

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_nn_tabular_framework_vs_experimental_battery.py --budget 400 --seeds 70-79
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

# HPOBench (vendored)
sys.path.insert(0, "/mnt/workspace/HPOBench")

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"


def _enable_fast_tabular_lookup(benchmark: TabularBenchmark) -> bool:
    """Speed up TabularBenchmark by replacing the O(n) dataframe scan with an indexed lookup.

    HPOBench's TabularBenchmark uses a boolean mask scan over the full table for every evaluation.
    For large tables this dominates runtime.

    This patch preserves behavior by only changing how rows are retrieved.
    """
    try:
        import pandas as pd  # type: ignore
        import types
    except Exception:
        return False

    df = getattr(benchmark, "table", None)
    if df is None:
        return False
    if not hasattr(df, "columns"):
        return False
    if "result" not in df.columns:
        return False

    key_cols = [c for c in df.columns if c != "result"]
    if not key_cols:
        return False

    # MultiIndex for fast exact-match lookup
    indexed = df.set_index(key_cols, drop=False)

    def _fast_search_dataframe(self: TabularBenchmark, row_dict: Dict[str, Any], _df_ignored):
        key = tuple(row_dict[c] for c in key_cols)
        row = indexed.loc[key]
        # Unique match expected; if multiple matches, take first.
        if isinstance(row, pd.Series):
            return row["result"]
        return row.iloc[0]["result"]

    benchmark._search_dataframe = types.MethodType(_fast_search_dataframe, benchmark)
    return True


def _parse_seeds(arg: str) -> List[int]:
    arg = arg.strip()
    if not arg:
        return []
    if "-" in arg and "," not in arg:
        a, b = arg.split("-", 1)
        lo = int(a)
        hi = int(b)
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
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


def _parse_checkpoints(arg: str, budget: int) -> List[int]:
    cps: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        cps.append(int(part))
    cps = [c for c in sorted(set(cps)) if 1 <= c <= budget]
    if budget not in cps:
        cps.append(budget)
    return cps


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _available_hpobench_tabular_task_ids(model: str = "nn") -> List[int]:
    """Return available task_ids for HPOBench TabularBenchmark datasets already present on disk."""
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
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue

        tid = int(child.name)
        parquet = child / f"{model}_{tid}_data.parquet.gzip"
        meta = child / f"{model}_{tid}_metadata.json"
        if parquet.exists() and meta.exists():
            task_ids.append(tid)

    task_ids.sort()
    return task_ids


def _x_to_config_dict(x: np.ndarray, hp_names: List[str], hp_seqs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Map x in [0,1]^d to a discrete config dict via uniform binning."""
    cfg: Dict[str, Any] = {}
    for i, name in enumerate(hp_names):
        seq = hp_seqs[name]
        k = len(seq)
        if k <= 1:
            cfg[name] = seq[0]
            continue
        idx = int(np.floor(float(x[i]) * k))
        if idx < 0:
            idx = 0
        elif idx >= k:
            idx = k - 1
        cfg[name] = seq[idx]
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description="NN Tabular battery: alba_framework_grid(param_space) vs ALBA_V1_experimental"
    )
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--checkpoints", default="100,200,400")
    p.add_argument("--seeds", default="70-79")
    p.add_argument(
        "--task_ids",
        default="",
        help="Optional comma-separated list of task IDs; default uses all available on disk.",
    )
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    checkpoints = _parse_checkpoints(args.checkpoints, args.budget)

    if args.task_ids.strip():
        task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = _available_hpobench_tabular_task_ids(model="nn")

    if not task_ids:
        raise SystemExit("No HPOBench nn/tabular task_ids found on disk.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        f"{RESULTS_DIR}/nn_tabular_framework_grid_vs_experimental_b{args.budget}_"
        f"s{seeds[0]}-{seeds[-1]}_{timestamp}.json"
    )

    # Import optimizers
    sys.path.insert(0, "/mnt/workspace/thesis")
    from alba_framework_grid import ALBA as ALBA_FW
    from ALBA_V1_experimental import ALBA as ALBA_EXP

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

    print(f"NN TABULAR BATTERY | budget={args.budget} | seeds={seeds}")
    print(f"task_ids={task_ids}")
    print(f"checkpoints={checkpoints}")
    print(f"save={out_path}")

    for task_id in task_ids:
        benchmark = TabularBenchmark(model="nn", task_id=int(task_id))
        fast_ok = _enable_fast_tabular_lookup(benchmark)
        print(f"task {task_id}: fast_lookup={'ON' if fast_ok else 'OFF'}")
        cs = benchmark.get_configuration_space()
        hps = list(cs.values())
        hp_names = [hp.name for hp in hps]
        hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
        dim = len(hp_names)
        categorical_dims: List[Tuple[int, int]] = [
            (i, len(hp_seqs[name])) for i, name in enumerate(hp_names)
        ]
        max_fidelity = benchmark.get_max_fidelity()

        # param_space for framework: each hp is categorical sequence
        param_space = {name: hp_seqs[name] for name in hp_names}

        results["tasks"].setdefault(str(task_id), {"framework": {}, "experimental": {}, "errors": {}})
        save_results(out_path, results)

        for seed in seeds:
            t0 = time.time()
            try:
                # ---------------- Framework (param_space) ----------------
                opt_fw = ALBA_FW(
                    param_space=param_space,
                    seed=seed,
                    maximize=False,
                    total_budget=args.budget,
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                best = float("inf")
                cp: Dict[str, float] = {}
                for it in range(args.budget):
                    cfg = opt_fw.ask()  # dict, already typed
                    res = benchmark.objective_function(configuration=cfg, fidelity=max_fidelity)
                    y = float(res["function_value"])
                    opt_fw.tell(cfg, y)
                    best = min(best, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best)
                results["tasks"][str(task_id)]["framework"][str(seed)] = cp
                save_results(out_path, results)

                # -------------- Experimental (array) --------------
                opt_exp = ALBA_EXP(
                    bounds=[(0.0, 1.0)] * dim,
                    categorical_dims=categorical_dims,
                    seed=seed,
                    maximize=False,
                    total_budget=args.budget,
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                best = float("inf")
                cp = {}
                for it in range(args.budget):
                    x = opt_exp.ask()
                    cfg = _x_to_config_dict(x, hp_names, hp_seqs)
                    res = benchmark.objective_function(configuration=cfg, fidelity=max_fidelity)
                    y = float(res["function_value"])
                    opt_exp.tell(x, y)
                    best = min(best, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best)
                results["tasks"][str(task_id)]["experimental"][str(seed)] = cp
                save_results(out_path, results)

                elapsed = time.time() - t0
                fw_final = results["tasks"][str(task_id)]["framework"][str(seed)].get(str(args.budget))
                exp_final = results["tasks"][str(task_id)]["experimental"][str(seed)].get(str(args.budget))
                winner = (
                    "FW"
                    if (fw_final is not None and exp_final is not None and fw_final < exp_final)
                    else "EXP"
                )
                print(
                    f"task {task_id} seed {seed}: FW={fw_final:.6f} EXP={exp_final:.6f} -> {winner} ({elapsed:.1f}s)"
                )

            except Exception as e:
                results["tasks"][str(task_id)]["errors"][str(seed)] = str(e)
                save_results(out_path, results)
                print(f"task {task_id} seed {seed}: ERROR: {e}")

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
