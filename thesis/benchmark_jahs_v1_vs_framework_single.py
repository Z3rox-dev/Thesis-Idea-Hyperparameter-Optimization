#!/usr/bin/env python3
"""Single-task JAHS benchmark: ALBA_V1 vs alba_framework.

Runs both optimizers with the same seed on the same JAHS task and compares
best valid-acc over a fixed evaluation budget.

Note: In this environment, loading JAHS task='cifar10' may be killed by the OS.
The default task is 'fashion_mnist' because it loads reliably.

Usage:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_jahs_v1_vs_framework_single.py --task fashion_mnist --budget 200 --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
SAVE_DIR = "/mnt/workspace/jahs_bench_data"


def _parse_seeds(arg: Optional[str], fallback_seed: int) -> List[int]:
    if arg is None:
        return [fallback_seed]
    arg = arg.strip()
    if not arg:
        return [fallback_seed]
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


def _search_space() -> Dict[str, Any]:
    return {
        "Optimizer": ["SGD"],
        "LearningRate": (1e-3, 1.0, "log"),
        "WeightDecay": (1e-5, 1e-2, "log"),
        "Activation": ["ReLU", "Hardswish", "Mish"],
        "TrivialAugment": [False, True],
        "N": [1, 3, 5],
        "W": [4, 8, 16],
        "Resolution": (0.25, 1.0),
        "Op1": [0, 1, 2, 3, 4],
        "Op2": [0, 1, 2, 3, 4],
        "Op3": [0, 1, 2, 3, 4],
        "Op4": [0, 1, 2, 3, 4],
        "Op5": [0, 1, 2, 3, 4],
        "Op6": [0, 1, 2, 3, 4],
    }


def _run_optimizer(name: str, opt: Any, benchmark: Any, nepochs: int, budget: int, checkpoints: List[int]) -> Dict[str, Any]:
    best = -np.inf
    cp: Dict[str, float] = {}
    t0 = time.time()

    for i in range(budget):
        cfg = opt.ask()  # dict
        result = benchmark(cfg, nepochs=nepochs)
        y = float(result[nepochs]["valid-acc"])
        opt.tell(cfg, y)

        if y > best:
            best = y

        if (i + 1) in checkpoints:
            cp[str(i + 1)] = float(best)

        # Drop the big result dict to reduce memory pressure
        del result
        if (i + 1) % 10 == 0:
            gc.collect()

    elapsed = time.time() - t0
    return {"best": float(best), "checkpoints": cp, "elapsed_s": elapsed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fashion_mnist", choices=["cifar10", "fashion_mnist", "colorectal_histology"])
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seeds",
        default=None,
        help="Comma list and/or ranges, e.g. '0-19' or '1,2,3,10-20'. If set, overrides --seed.",
    )
    ap.add_argument("--nepochs", type=int, default=200)
    ap.add_argument("--checkpoints", default="50,100,150,200")
    args = ap.parse_args()

    checkpoints = sorted({int(x.strip()) for x in args.checkpoints.split(",") if x.strip()})
    checkpoints = [c for c in checkpoints if 1 <= c <= args.budget]
    if args.budget not in checkpoints:
        checkpoints.append(args.budget)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = _parse_seeds(args.seeds, args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/jahs_v1_vs_framework_{args.task}_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{stamp}.json"

    # Prefer the vendored workspace root for imports
    sys.path.insert(0, "/mnt/workspace")
    sys.path.insert(0, "/mnt/workspace/thesis")

    import jahs_bench

    print(f"Loading JAHS benchmark task={args.task} kind=surrogate...")
    benchmark = jahs_bench.Benchmark(task=args.task, kind="surrogate", download=False, save_dir=SAVE_DIR)

    space = _search_space()

    from ALBA_V1 import ALBA as ALBA_V1
    from alba_framework import ALBA as ALBA_FW

    results: Dict[str, Any] = {
        "config": {
            "task": args.task,
            "budget": args.budget,
            "seeds": seeds,
            "nepochs": args.nepochs,
            "checkpoints": checkpoints,
        },
        "seeds": {},
        "summary": {},
    }

    v1_bests: List[float] = []
    fw_bests: List[float] = []
    v1_wins = 0
    fw_wins = 0
    ties = 0

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")

        print("=== Running ALBA_V1 ===")
        opt_v1 = ALBA_V1(param_space=space, seed=seed, maximize=True, total_budget=args.budget)
        res_v1 = _run_optimizer("v1", opt_v1, benchmark, args.nepochs, args.budget, checkpoints)
        print(f"ALBA_V1 best={res_v1['best']:.4f}% elapsed={res_v1['elapsed_s']:.1f}s")

        print("=== Running alba_framework ===")
        opt_fw = ALBA_FW(param_space=space, seed=seed, maximize=True, total_budget=args.budget)
        res_fw = _run_optimizer("framework", opt_fw, benchmark, args.nepochs, args.budget, checkpoints)
        print(f"Framework best={res_fw['best']:.4f}% elapsed={res_fw['elapsed_s']:.1f}s")

        results["seeds"][str(seed)] = {"v1": res_v1, "framework": res_fw}

        v1_bests.append(float(res_v1["best"]))
        fw_bests.append(float(res_fw["best"]))
        if res_v1["best"] > res_fw["best"]:
            v1_wins += 1
        elif res_fw["best"] > res_v1["best"]:
            fw_wins += 1
        else:
            ties += 1

        # incremental save
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    def _mean_std(xs: List[float]) -> Dict[str, float]:
        arr = np.asarray(xs, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    results["summary"] = {
        "v1": _mean_std(v1_bests),
        "framework": _mean_std(fw_bests),
        "wins": {"v1": v1_wins, "framework": fw_wins, "ties": ties},
    }

    print("\n=== Summary ===")
    print(f"Task={args.task} budget={args.budget} seeds={seeds[0]}-{seeds[-1]} nepochs={args.nepochs}")
    print(f"ALBA_V1     mean(best)={results['summary']['v1']['mean']:.4f}% std={results['summary']['v1']['std']:.4f}")
    print(f"Framework   mean(best)={results['summary']['framework']['mean']:.4f}% std={results['summary']['framework']['std']:.4f}")
    print(f"Wins: V1={v1_wins} FW={fw_wins} ties={ties}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
