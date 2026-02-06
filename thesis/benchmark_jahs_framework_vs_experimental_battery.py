#!/usr/bin/env python3
"""JAHS-Bench-201 battery: alba_framework vs ALBA_V1_experimental.

- Minimizes error = 1 - valid_acc/100 (same as thesis/benchmark_jahs.py wrapper)
- Saves incremental checkpoints to JSON

Intended env: conda `py39`.

Usage:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_jahs_framework_vs_experimental_battery.py \
        --tasks fashion_mnist --budget 400 --seeds 70-79

Notes:
    If --tasks is omitted, the script runs all JAHS tasks:
    cifar10,fashion_mnist,colorectal_histology
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _make_logger(log_path: str | None):
    f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        print(msg, flush=True)
        if f is not None:
            try:
                f.write(msg + "\n")
            except Exception:
                pass

    return log, f


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tasks",
        default="cifar10,fashion_mnist,colorectal_histology",
        help=(
            "Comma-separated list of tasks (cifar10,fashion_mnist,colorectal_histology). "
            "Default: all three."
        ),
    )
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--checkpoints", default="100,200,400")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=70)
    p.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated explicit seeds (overrides seed-start/n-seeds). Supports ranges '70-79'.",
    )
    p.add_argument("--nepochs", type=int, default=200)
    p.add_argument("--grid-bins", type=int, default=8)
    p.add_argument("--grid-batch-size", type=int, default=512)
    p.add_argument("--grid-batches", type=int, default=4)
    p.add_argument(
        "--grid-sampling",
        default="grid_random",
        choices=["grid_random", "grid_halton", "halton", "heatmap_ucb"],
    )
    p.add_argument("--grid-jitter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grid-penalty-lambda", type=float, default=0.06)
    p.add_argument("--heatmap-ucb-beta", type=float, default=1.0)
    p.add_argument("--heatmap-ucb-explore-prob", type=float, default=0.25)
    p.add_argument("--heatmap-ucb-temperature", type=float, default=1.0)
    p.add_argument("--heatmap-blend-tau", type=float, default=1e9)
    p.add_argument("--heatmap-soft-assignment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--heatmap-multi-resolution", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--categorical-stage",
        default="auto",
        choices=["auto", "pre", "post"],
        help="How to apply categorical sampling relative to grid scoring.",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory to write per-run JSONL traces (framework only).",
    )
    p.add_argument("--trace-top-k", type=int, default=0, help="Store top-k candidates per ask() trace.")
    p.add_argument(
        "--log-path",
        default=None,
        help="Optional log file path. If omitted, uses the JSON output path with .log suffix.",
    )
    args = p.parse_args()

    allowed_tasks = {"cifar10", "fashion_mnist", "colorectal_histology"}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No tasks provided")
    bad = [t for t in tasks if t not in allowed_tasks]
    if bad:
        raise SystemExit(f"Invalid tasks: {bad}. Allowed: {sorted(allowed_tasks)}")

    checkpoints: List[int] = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    checkpoints = sorted(set([c for c in checkpoints if 1 <= c <= args.budget]))
    if args.budget not in checkpoints:
        checkpoints.append(args.budget)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_tag = "alltasks" if len(tasks) > 1 else tasks[0]
    out_path = f"{RESULTS_DIR}/jahs_framework_vs_experimental_{task_tag}_{timestamp}.json"

    log_path = args.log_path or (out_path.replace(".json", ".log"))
    log, log_fh = _make_logger(log_path)

    sys.path.insert(0, "/mnt/workspace/thesis")
    from benchmark_jahs import JAHSBenchWrapper
    # Use the grid-enabled framework variant.
    from alba_framework_grid import ALBA as ALBA_FW
    from ALBA_V1_experimental import ALBA as ALBA_EXP
    from alba_framework_grid.diagnostics import TraceJSONLWriter

    # Build seed list
    if args.seeds is not None:
        seeds: List[int] = []
        for part in args.seeds.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                lo = int(a)
                hi = int(b)
                if hi < lo:
                    lo, hi = hi, lo
                seeds.extend(list(range(lo, hi + 1)))
            else:
                seeds.append(int(part))
        seeds = sorted(set(seeds))
        if not seeds:
            raise SystemExit("--seeds was provided but empty")
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    # Indices based on wrapper.HP_ORDER:
    # ['LearningRate','WeightDecay','N','W','Resolution','Activation','TrivialAugment','Op1'..'Op6']
    # Experimental needs categorical dims as (dim_idx, n_choices)
    categorical_dims = [
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 2),
        (7, 5),
        (8, 5),
        (9, 5),
        (10, 5),
        (11, 5),
        (12, 5),
    ]

    results: Dict[str, Any] = {
        "config": {
            "tasks": tasks,
            "budget": args.budget,
            "checkpoints": checkpoints,
            "nepochs": args.nepochs,
            "seeds": seeds,
            "framework": {
                "grid_bins": args.grid_bins,
                "grid_batch_size": args.grid_batch_size,
                "grid_batches": args.grid_batches,
                "grid_sampling": args.grid_sampling,
                "grid_jitter": args.grid_jitter,
                "grid_penalty_lambda": args.grid_penalty_lambda,
                "heatmap_ucb": {
                    "beta": float(args.heatmap_ucb_beta),
                    "explore_prob": float(args.heatmap_ucb_explore_prob),
                    "temperature": float(args.heatmap_ucb_temperature),
                },
                "heatmap_blend_tau": float(args.heatmap_blend_tau),
                "heatmap_soft_assignment": bool(args.heatmap_soft_assignment),
                "heatmap_multi_resolution": bool(args.heatmap_multi_resolution),
                "categorical_stage": str(args.categorical_stage),
            },
            "timestamp": timestamp,
        },
        "tasks": {},
    }
    save_results(out_path, results)

    log(f"Benchmark tasks: {tasks} | budget={args.budget} | seeds={seeds}")
    log(f"Checkpoints: {checkpoints}")
    log(
        "Framework config: "
        f"grid_bins={args.grid_bins} "
        f"grid_batch_size={args.grid_batch_size} grid_batches={args.grid_batches} "
        f"grid_sampling={args.grid_sampling} grid_jitter={args.grid_jitter} "
        f"grid_penalty_lambda={args.grid_penalty_lambda} "
        f"heatmap_ucb_beta={args.heatmap_ucb_beta} "
        f"heatmap_ucb_explore_prob={args.heatmap_ucb_explore_prob} "
        f"heatmap_ucb_temperature={args.heatmap_ucb_temperature}"
    )
    log(f"Saving to: {out_path}")
    log(f"Logging to: {log_path}")

    for task in tasks:
        log(f"\n=== TASK {task} ===")
        wrapper = JAHSBenchWrapper(task=task)
        dim = wrapper.dim

        results["tasks"].setdefault(task, {"framework": {}, "experimental": {}, "errors": {}})
        save_results(out_path, results)

        for seed in seeds:
            log(f"\nTask {task} | Seed {seed}...")
            start = time.time()

            try:
                # ------------------------- alba_framework (array) -------------------------
                trace_writer = None
                trace_hook = None
                trace_hook_tell = None
                if args.trace_dir:
                    os.makedirs(args.trace_dir, exist_ok=True)
                    trace_path = os.path.join(
                        args.trace_dir,
                        f"trace_jahs_fw_{task}_seed{seed}_b{args.budget}_{timestamp}.jsonl",
                    )
                    trace_writer = TraceJSONLWriter(trace_path)

                    run_meta = {
                        "benchmark": "jahs",
                        "task": str(task),
                        "seed": int(seed),
                        "budget": int(args.budget),
                        "timestamp": str(timestamp),
                    }

                    def _trace_hook(ev: Dict[str, Any]) -> None:
                        ev["run"] = run_meta
                        trace_writer(ev)

                    trace_hook = _trace_hook
                    trace_hook_tell = _trace_hook

                opt_fw = ALBA_FW(
                    bounds=[(0.0, 1.0)] * dim,
                    categorical_dims=categorical_dims,
                    seed=seed,
                    maximize=False,  # minimize error
                    split_depth_max=8,
                    total_budget=args.budget,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                    grid_bins=args.grid_bins,
                    grid_batch_size=args.grid_batch_size,
                    grid_batches=args.grid_batches,
                    grid_sampling=args.grid_sampling,
                    grid_jitter=args.grid_jitter,
                    grid_penalty_lambda=args.grid_penalty_lambda,
                    heatmap_ucb_beta=float(args.heatmap_ucb_beta),
                    heatmap_ucb_explore_prob=float(args.heatmap_ucb_explore_prob),
                    heatmap_ucb_temperature=float(args.heatmap_ucb_temperature),
                    heatmap_blend_tau=float(args.heatmap_blend_tau),
                    heatmap_soft_assignment=bool(args.heatmap_soft_assignment),
                    heatmap_multi_resolution=bool(args.heatmap_multi_resolution),
                    categorical_stage=str(args.categorical_stage),
                    trace_top_k=int(args.trace_top_k),
                    trace_hook=trace_hook,
                    trace_hook_tell=trace_hook_tell,
                )

                wrapper.reset()
                best_err = float("inf")
                cp: Dict[str, float] = {}
                for it in range(args.budget):
                    x = opt_fw.ask()
                    y = float(wrapper.evaluate_array(x))
                    opt_fw.tell(x, y)
                    best_err = min(best_err, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best_err)

                results["tasks"][task]["framework"][str(seed)] = cp
                save_results(out_path, results)
                if trace_writer is not None:
                    try:
                        trace_writer.close()
                    except Exception:
                        pass

                # ------------------------- ALBA experimental (array) -------------------
                opt_exp = ALBA_EXP(
                    bounds=[(0.0, 1.0)] * dim,
                    categorical_dims=categorical_dims,
                    seed=seed,
                    maximize=False,  # minimize error
                    split_depth_max=8,
                    total_budget=args.budget,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                wrapper.reset()
                best_err = float("inf")
                cp = {}
                for it in range(args.budget):
                    x = opt_exp.ask()
                    y = float(wrapper.evaluate_array(x))
                    opt_exp.tell(x, y)
                    best_err = min(best_err, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best_err)
                results["tasks"][task]["experimental"][str(seed)] = cp
                save_results(out_path, results)

                elapsed = time.time() - start
                fw_final = results["tasks"][task]["framework"][str(seed)].get(str(args.budget))
                exp_final = results["tasks"][task]["experimental"][str(seed)].get(str(args.budget))
                if fw_final is not None and exp_final is not None:
                    fw_acc = (1.0 - fw_final) * 100.0
                    exp_acc = (1.0 - exp_final) * 100.0
                    winner = "FW" if fw_final < exp_final else "EXP"
                    log(f"  FW={fw_acc:.2f}%  EXP={exp_acc:.2f}%  -> {winner} ({elapsed:.1f}s)")
                else:
                    log(f"  Done ({elapsed:.1f}s)")

            except Exception as e:
                results["tasks"][task]["errors"][str(seed)] = {
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                save_results(out_path, results)
                log(f"  ERROR: {e}")

    # Summary per task
    def finals(block: Dict[str, Dict[str, float]]) -> List[float]:
        out: List[float] = []
        for seed in seeds:
            d = block.get(str(seed), {})
            v = d.get(str(args.budget))
            if v is not None:
                out.append(float(v))
        return out

    log("\n=== SUMMARY (final checkpoint) ===")
    for task in tasks:
        tdata = results["tasks"].get(task, {})
        fw_err = finals(tdata.get("framework", {}))
        exp_err = finals(tdata.get("experimental", {}))
        if not fw_err or not exp_err:
            log(f"{task}: n/a (missing finals)")
            continue
        fw_acc = (1.0 - np.array(fw_err)) * 100.0
        exp_acc = (1.0 - np.array(exp_err)) * 100.0
        wins = int((np.array(fw_err) < np.array(exp_err)).sum())
        ties = int((np.array(fw_err) == np.array(exp_err)).sum())
        log(
            f"{task}: FW mean={fw_acc.mean():.2f}% std={fw_acc.std():.2f} | "
            f"EXP mean={exp_acc.mean():.2f}% std={exp_acc.std():.2f} | wins FW={wins} ties={ties}"
        )

    log("\nSaved: " + out_path)
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
