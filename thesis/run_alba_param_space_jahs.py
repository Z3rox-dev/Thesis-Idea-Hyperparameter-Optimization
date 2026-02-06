#!/usr/bin/env python3
"""Run ALBA_V1 in param_space mode on JAHS-Bench-201.

This script demonstrates the new API:
  opt = ALBA(param_space=SEARCH_SPACE, ...)
  cfg = opt.ask()          # dict config
  y = benchmark(cfg, ...)  # evaluate
  opt.tell(cfg, y)         # report

Default: maximize valid-acc for 500 iterations.

Usage:
  python thesis/run_alba_param_space_jahs.py --task cifar10 --budget 500
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import numpy as np


def build_search_space(nepochs: int) -> Dict[str, Any]:
    # Matches the conventions used in thesis/benchmark_jahs.py wrapper:
    # - Ops are integers 0..4
    # - Resolution is treated as ordinal with 3 choices
    return {
        "Optimizer": ["SGD"],
        "epoch": [nepochs],
        "LearningRate": (1e-3, 1.0, "log"),
        "WeightDecay": (1e-5, 1e-2, "log"),
        "N": [1, 3, 5],
        "W": [4, 8, 16],
        "Resolution": [0.25, 0.5, 1.0],
        "Activation": ["ReLU", "Hardswish", "Mish"],
        "TrivialAugment": [True, False],
        "Op1": [0, 1, 2, 3, 4],
        "Op2": [0, 1, 2, 3, 4],
        "Op3": [0, 1, 2, 3, 4],
        "Op4": [0, 1, 2, 3, 4],
        "Op5": [0, 1, 2, 3, 4],
        "Op6": [0, 1, 2, 3, 4],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cifar10", choices=["cifar10", "fashion_mnist", "colorectal_histology"])
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nepochs", type=int, default=200)
    p.add_argument("--save-dir", default="/mnt/workspace/jahs_bench_data")
    args = p.parse_args()

    # Import locally from thesis
    sys.path.insert(0, "/mnt/workspace/thesis")
    from ALBA_V1 import ALBA

    import jahs_bench

    print(f"Loading JAHS benchmark: task={args.task} kind=surrogate")
    bench = jahs_bench.Benchmark(
        task=args.task,
        kind="surrogate",
        download=False,
        save_dir=args.save_dir,
    )

    search_space = build_search_space(args.nepochs)

    opt = ALBA(
        param_space=search_space,
        seed=args.seed,
        maximize=True,
        total_budget=args.budget,
    )

    best_y = -np.inf
    for i in range(args.budget):
        cfg = opt.ask()  # dict
        result = bench(cfg, nepochs=args.nepochs)
        y = float(result[args.nepochs]["valid-acc"])
        opt.tell(cfg, y)

        if y > best_y:
            best_y = y

        if (i + 1) % 50 == 0:
            print(f"Iter {i+1}/{args.budget}: best valid-acc = {best_y:.4f}")

    print(f"Final best valid-acc = {best_y:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
