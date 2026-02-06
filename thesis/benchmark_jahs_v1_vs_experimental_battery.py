#!/usr/bin/env python3
"""JAHS-Bench-201 battery: ALBA_V1 (param_space) vs ALBA_V1_experimental.

- Minimizes error = 1 - valid_acc/100 (same as thesis/benchmark_jahs.py wrapper)
- Budget default: 2000
- Saves incremental checkpoints to JSON

Usage:
  source /mnt/workspace/miniconda3/bin/activate py39
    python thesis/benchmark_jahs_v1_vs_experimental_battery.py --tasks cifar10,fashion_mnist,colorectal_histology --budget 2000 --seed-start 10 --n-seeds 10
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


def build_search_space(nepochs: int) -> Dict[str, Any]:
    # Keep consistent with thesis/benchmark_jahs.py decoding.
    return {
        "Optimizer": ["SGD"],
        "epoch": [nepochs],
        "LearningRate": (1e-3, 1.0, "log"),
        "WeightDecay": (1e-5, 1e-2, "log"),
        "Activation": ["ReLU", "Hardswish", "Mish"],
        "TrivialAugment": [True, False],
        "N": [1, 3, 5],
        "W": [4, 8, 16],
        # JAHS wrapper treats Resolution as ordinal with 3 choices
        "Resolution": [0.25, 0.5, 1.0],
        "Op1": [0, 1, 2, 3, 4],
        "Op2": [0, 1, 2, 3, 4],
        "Op3": [0, 1, 2, 3, 4],
        "Op4": [0, 1, 2, 3, 4],
        "Op5": [0, 1, 2, 3, 4],
        "Op6": [0, 1, 2, 3, 4],
    }


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
        default="cifar10",
        help="Comma-separated list of tasks (cifar10,fashion_mnist,colorectal_histology)",
    )
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--checkpoints", default="100,250,500,1000,1500,2000")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated explicit seeds (overrides seed-start/n-seeds)",
    )
    p.add_argument("--nepochs", type=int, default=200)
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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_tag = "alltasks" if len(tasks) > 1 else tasks[0]
    out_path = f"{RESULTS_DIR}/jahs_v1_vs_experimental_{task_tag}_{timestamp}.json"

    log_path = args.log_path or (out_path.replace(".json", ".log"))
    log, log_fh = _make_logger(log_path)

    sys.path.insert(0, "/mnt/workspace/thesis")
    from benchmark_jahs import JAHSBenchWrapper
    from ALBA_V1 import ALBA as ALBA_V1
    from ALBA_V1_experimental import ALBA as ALBA_EXP

    # Build seed list
    if args.seeds is not None:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        if not seeds:
            raise SystemExit("--seeds was provided but empty")
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    # Indices based on wrapper.HP_ORDER:
    # ['LearningRate','WeightDecay','N','W','Resolution','Activation','TrivialAugment','Op1'..'Op6']
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]

    results: Dict[str, Any] = {
        "config": {
            "tasks": tasks,
            "budget": args.budget,
            "checkpoints": checkpoints,
            "nepochs": args.nepochs,
            "seeds": seeds,
            "timestamp": timestamp,
        },
        "tasks": {},
    }
    save_results(out_path, results)

    search_space = build_search_space(args.nepochs)

    log(f"Benchmark tasks: {tasks} | budget={args.budget} | seeds={seeds}")
    log(f"Checkpoints: {checkpoints}")
    log(f"Saving to: {out_path}")
    log(f"Logging to: {log_path}")

    for task in tasks:
        log(f"\n=== TASK {task} ===")
        wrapper = JAHSBenchWrapper(task=task)
        dim = wrapper.dim

        results["tasks"].setdefault(task, {"v1_param_space": {}, "experimental": {}, "errors": {}})
        save_results(out_path, results)

        for seed in seeds:
            log(f"\nTask {task} | Seed {seed}...")
            start = time.time()

            try:
                # ------------------------- ALBA_V1 param_space -------------------------
                opt_v1 = ALBA_V1(
                    param_space=search_space,
                    seed=seed,
                    maximize=False,  # minimize error
                    split_depth_max=8,
                    total_budget=args.budget,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                wrapper.reset()
                best_err = float("inf")
                cp: Dict[str, float] = {}
                for it in range(args.budget):
                    cfg = opt_v1.ask()  # dict
                    y = float(wrapper.evaluate(cfg))
                    opt_v1.tell(cfg, y)
                    best_err = min(best_err, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best_err)
                results["tasks"][task]["v1_param_space"][str(seed)] = cp
                save_results(out_path, results)

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
                v1_final = results["tasks"][task]["v1_param_space"][str(seed)].get(str(args.budget))
                exp_final = results["tasks"][task]["experimental"][str(seed)].get(str(args.budget))
                if v1_final is not None and exp_final is not None:
                    v1_acc = (1.0 - v1_final) * 100.0
                    exp_acc = (1.0 - exp_final) * 100.0
                    winner = "V1" if v1_final < exp_final else "EXP"
                    log(f"  V1={v1_acc:.2f}%  EXP={exp_acc:.2f}%  -> {winner} ({elapsed:.1f}s)")
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
        v1_err = finals(tdata.get("v1_param_space", {}))
        exp_err = finals(tdata.get("experimental", {}))
        if not v1_err or not exp_err:
            log(f"{task}: n/a (missing finals)")
            continue
        v1_acc = (1.0 - np.array(v1_err)) * 100.0
        exp_acc = (1.0 - np.array(exp_err)) * 100.0
        wins = int((np.array(v1_err) < np.array(exp_err)).sum())
        log(
            f"{task}: V1 mean={v1_acc.mean():.3f} std={v1_acc.std():.3f} | "
            f"EXP mean={exp_acc.mean():.3f} std={exp_acc.std():.3f} | V1 wins {wins}/{len(v1_err)}"
        )

    log(f"\nSaved: {out_path}")
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
