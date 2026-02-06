#!/usr/bin/env python3
"""JAHS-Bench-201 battery: ALBA_V1 vs ALBA meta-objective (budget 2000, 10 seeds).

Runs the same protocol as `thesis/benchmark_jahs_battery.py`, but swaps Optuna
for the meta-objective variant implemented in `thesis/ALBA_V1 copy 2.py`.

Intended environment: conda `py39` (JAHS-Bench-201).
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Any, Dict

import numpy as np


# Config
BUDGET = 2000
CHECKPOINTS = [100, 250, 500, 1000, 1500, 2000]
N_SEEDS = 10
SEED_START = 42
SEEDS = list(range(SEED_START, SEED_START + N_SEEDS))
JAHS_TASKS = ["cifar10", "fashion_mnist", "colorectal_histology"]

RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULTS_FILE = f"{RESULTS_DIR}/jahs_benchmark_meta_vs_v1_{timestamp}.json"


def _load_alba_from_path(module_name: str, file_path: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure dataclasses can resolve the module during class processing.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ALBA"):
        raise RuntimeError(f"Module {file_path} does not export ALBA")
    return mod.ALBA


def save_results(results_file: str, obj: Dict[str, Any]) -> None:
    with open(results_file, "w") as f:
        json.dump(obj, f, indent=2)


def load_results_if_any(results_file: str) -> Dict[str, Any]:
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}


def _is_seed_done(task_data: Dict[str, Any], algo_key: str, seed: int) -> bool:
    seed_key = str(seed)
    cp = task_data.get(algo_key, {}).get(seed_key, {})
    if not cp:
        return False
    # keys are serialized as strings
    return str(BUDGET) in cp or BUDGET in cp


def main() -> None:
    parser = argparse.ArgumentParser(description="JAHS meta vs V1 battery")
    parser.add_argument(
        "--results_file",
        type=str,
        default=DEFAULT_RESULTS_FILE,
        help="Path to results JSON (use an existing file to resume)",
    )
    parser.add_argument(
        "--meta_variant",
        type=str,
        default="default",
        choices=["default", "lexi_topk"],
        help="Which meta-ALBA variant to run",
    )
    args = parser.parse_args()
    results_file = args.results_file
    meta_variant = args.meta_variant

    if meta_variant == "lexi_topk":
        meta_kwargs = dict(
            leaf_selection_mode="lexi_topk",
            leaf_reward_topk=5,
            cat_selection_mode="lexi_topk",
            cat_reward_topk=3,
        )
        meta_key = "alba_meta_lexi_topk"
    else:
        meta_kwargs = {}
        meta_key = "alba_meta"

    print("Starting JAHS Meta vs V1 Battery")
    print(f"Tasks: {JAHS_TASKS}")
    print(f"Budget: {BUDGET}, Seeds: {N_SEEDS}")
    print(f"Results file: {results_file}")
    print(f"Meta variant: {meta_variant}")
    print("=" * 60)

    # Use the existing wrapper already used across the repo.
    sys.path.insert(0, "/mnt/workspace/thesis")
    from benchmark_jahs import JAHSBenchWrapper
    from ALBA_V1 import ALBA as ALBA_V1

    ALBA_META = _load_alba_from_path(
        module_name="alba_meta_copy2",
        file_path="/mnt/workspace/thesis/ALBA_V1 copy 2.py",
    )

    # Categorical dims for JAHS in wrapper space.
    # (dim_idx, n_choices)
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]

    existing = load_results_if_any(results_file)
    if existing:
        results = existing
        # Keep config but ensure it matches current constants.
        results.setdefault("config", {})
        results["config"].update(
            {
                "budget": BUDGET,
                "checkpoints": CHECKPOINTS,
                "n_seeds": N_SEEDS,
                "seeds": SEEDS,
                "timestamp": results["config"].get("timestamp", timestamp),
                "meta_variant": meta_variant,
                "meta_kwargs": meta_kwargs,
            }
        )
        results.setdefault("jahs", {})
        print(f"Resuming from existing results file: {results_file}")
    else:
        results = {
            "config": {
                "budget": BUDGET,
                "checkpoints": CHECKPOINTS,
                "n_seeds": N_SEEDS,
                "seeds": SEEDS,
                "timestamp": timestamp,
                "meta_variant": meta_variant,
                "meta_kwargs": meta_kwargs,
            },
            "jahs": {},
        }
        save_results(results_file, results)

    for task in JAHS_TASKS:
        print(f"\n>>> Running {task.upper()}")

        wrapper = JAHSBenchWrapper(task=task)
        dim = wrapper.dim

        results["jahs"].setdefault(task, {})
        results["jahs"][task].setdefault("dim", dim)
        results["jahs"][task].setdefault("alba_v1", {})
        results["jahs"][task].setdefault(meta_key, {})
        results["jahs"][task].setdefault("errors", {})
        task_bucket = results["jahs"][task]

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            start = time.time()

            try:
                # Skip if already completed for both algorithms
                if _is_seed_done(task_bucket, "alba_v1", seed) and _is_seed_done(task_bucket, meta_key, seed):
                    print("SKIP (already done)")
                    continue

                # --- ALBA_V1 ---
                opt_v1 = ALBA_V1(
                    bounds=[(0.0, 1.0)] * dim,
                    maximize=False,
                    seed=seed,
                    split_depth_max=8,
                    total_budget=BUDGET,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                    categorical_dims=categorical_dims,
                )

                wrapper.reset()
                v1_best = float("inf")
                v1_checkpoints: Dict[int, float] = {}
                for it in range(BUDGET):
                    x = opt_v1.ask()
                    y = float(wrapper.evaluate_array(x))
                    opt_v1.tell(x, y)
                    v1_best = min(v1_best, y)
                    if (it + 1) in CHECKPOINTS:
                        v1_checkpoints[it + 1] = float(v1_best)

                task_bucket["alba_v1"][str(seed)] = v1_checkpoints

                # --- ALBA_META (copy2) ---
                opt_meta = ALBA_META(
                    bounds=[(0.0, 1.0)] * dim,
                    maximize=False,
                    seed=seed,
                    split_depth_max=8,
                    total_budget=BUDGET,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                    categorical_dims=categorical_dims,
                    **meta_kwargs,
                )

                wrapper.reset()
                meta_best = float("inf")
                meta_checkpoints: Dict[int, float] = {}
                for it in range(BUDGET):
                    x = opt_meta.ask()
                    y = float(wrapper.evaluate_array(x))
                    opt_meta.tell(x, y)
                    meta_best = min(meta_best, y)
                    if (it + 1) in CHECKPOINTS:
                        meta_checkpoints[it + 1] = float(meta_best)

                task_bucket[meta_key][str(seed)] = meta_checkpoints

                v1_final = v1_checkpoints.get(BUDGET, v1_best)
                meta_final = meta_checkpoints.get(BUDGET, meta_best)

                v1_acc = (1.0 - v1_final) * 100
                meta_acc = (1.0 - meta_final) * 100
                winner = "META" if meta_final < v1_final else "V1"

                elapsed = time.time() - start
                print(f"V1={v1_acc:.2f}% META={meta_acc:.2f}% -> {winner} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                task_bucket["errors"][str(seed)] = str(e)

            # Save after each seed (even on error)
            save_results(results_file, results)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY JAHS (META vs V1)")
    print("=" * 60)

    total_meta = 0
    total_v1 = 0

    for task, task_data in results["jahs"].items():
        meta_wins = 0
        v1_wins = 0
        ties = 0

        v1_by_seed = task_data.get("alba_v1", {})
        meta_by_seed = task_data.get("alba_meta", {})
        err_by_seed = task_data.get("errors", {})

        for seed in SEEDS:
            seed_key = str(seed)
            if seed_key in err_by_seed:
                continue

            v1_cp = v1_by_seed.get(seed_key, {})
            meta_cp = meta_by_seed.get(seed_key, {})

            v1_final = v1_cp.get(BUDGET)
            meta_final = meta_cp.get(BUDGET)
            if v1_final is None or meta_final is None:
                continue

            if meta_final < v1_final - 1e-9:
                meta_wins += 1
            elif v1_final < meta_final - 1e-9:
                v1_wins += 1
            else:
                ties += 1

        print(f"{task}: META {meta_wins} - {v1_wins} V1 (tie: {ties}, errors: {len(err_by_seed)})")
        total_meta += meta_wins
        total_v1 += v1_wins

    print(f"\nTOTAL: META {total_meta} - {total_v1} V1")
    denom = (total_meta + total_v1)
    if denom > 0:
        print(f"Win rate: {total_meta/denom*100:.1f}%")
    else:
        print("Win rate: n/a (nessun seed completato)")


if __name__ == "__main__":
    main()
