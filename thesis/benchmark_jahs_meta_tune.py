#!/usr/bin/env python3
"""Quick JAHS tuning: ALBA_V1 vs ALBA meta-objective variants.

Goal: find meta settings that at least match/beat V1 on JAHS with small budget,
then launch the full 2000-eval battery.

Environment: conda `py39`.

Example:
  python thesis/benchmark_jahs_meta_tune.py
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np


TASK = "cifar10"
BUDGET = 1000
SEEDS = [42, 43, 44, 45, 46]

RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_alba_from_path(module_name: str, file_path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses needs it during class processing
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ALBA"):
        raise RuntimeError(f"Module {file_path} does not export ALBA")
    return mod.ALBA


def run_one(wrapper, OptClass, dim: int, categorical_dims: List[Tuple[int, int]], seed: int, opt_kwargs: Dict[str, Any]):
    opt = OptClass(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=BUDGET,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
        **opt_kwargs,
    )

    wrapper.reset()
    best_err = float("inf")
    for _ in range(BUDGET):
        x = opt.ask()
        y = float(wrapper.evaluate_array(x))
        opt.tell(x, y)
        best_err = min(best_err, y)
    return float(best_err)


def main() -> None:
    sys.path.insert(0, "/mnt/workspace/thesis")
    from benchmark_jahs import JAHSBenchWrapper
    from ALBA_V1 import ALBA as ALBA_V1

    ALBA_META = _load_alba_from_path(
        module_name="alba_meta_copy2_tune",
        file_path="/mnt/workspace/thesis/ALBA_V1 copy 2.py",
    )

    wrapper = JAHSBenchWrapper(task=TASK)
    dim = wrapper.dim

    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]

    variants = [
        ("V1", ALBA_V1, {}),
        # Baseline meta (as currently defaulted in copy2)
        ("META_default", ALBA_META, {}),
        # Hierarchical (lexicographic): keep only Top-K by reward, then maximize knowledge.
        (
            "META_lexi_topk",
            ALBA_META,
            dict(
                leaf_selection_mode="lexi_topk",
                leaf_reward_topk=5,
                cat_selection_mode="lexi_topk",
                cat_reward_topk=3,
            ),
        ),
        # Same as default, but reduce categorical meta mixing a bit early.
        (
            "META_default_mix035",
            ALBA_META,
            dict(
                cat_meta_mix_start=0.35,
                cat_meta_mix_end=0.0,
            ),
        ),
        # Reduce meta further; only boost on stagnation
        (
            "META_weak_meta",
            ALBA_META,
            dict(
                meta_weight_start=0.25,
                meta_weight_end=0.0,
                meta_stagnation_boost=0.35,
                meta_beta=0.75,
            ),
        ),
        # Make meta_good stricter but TS mixing more reward-centric
        (
            "META_quantile_mix_low",
            ALBA_META,
            dict(
                meta_good_mode="quantile",
                gamma_m_quantile=0.50,
                cat_meta_mix_start=0.30,
                cat_meta_mix_end=0.0,
            ),
        ),
        # Meta-good = positive, but rely on reward for categoricals earlier
        (
            "META_rewardish_cats",
            ALBA_META,
            dict(
                cat_meta_mix_start=0.25,
                cat_meta_mix_end=0.0,
                meta_weight_start=0.30,
                meta_weight_end=0.0,
            ),
        ),
    ]

    results: Dict[str, Any] = {
        "task": TASK,
        "budget": BUDGET,
        "seeds": SEEDS,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "variants": {},
    }

    print("=" * 70)
    print(f"JAHS TUNE: task={TASK} budget={BUDGET} seeds={SEEDS}")
    print("=" * 70)

    for name, OptClass, kwargs in variants:
        errs = []
        t0 = time.time()
        for seed in SEEDS:
            err = run_one(wrapper, OptClass, dim, categorical_dims, seed, kwargs)
            errs.append(err)
            print(f"{name:22s} seed={seed} err={err:.6f} acc={(1-err)*100:.2f}%")
        elapsed = time.time() - t0
        mean = float(np.mean(errs))
        std = float(np.std(errs))
        results["variants"][name] = {
            "kwargs": kwargs,
            "errors": errs,
            "mean": mean,
            "std": std,
            "mean_acc": float((1.0 - mean) * 100.0),
            "time_s": float(elapsed),
        }
        print(f"{name:22s} MEAN err={mean:.6f} ± {std:.6f}  (acc={(1-mean)*100:.2f}%)  time={elapsed:.1f}s")
        print("-" * 70)

    # Rank by mean error
    ranked = sorted(results["variants"].items(), key=lambda kv: kv[1]["mean"])
    print("\nRANKING (lower error is better):")
    for i, (name, data) in enumerate(ranked, 1):
        print(f"  {i}. {name:22s} mean_err={data['mean']:.6f}  mean_acc={data['mean_acc']:.2f}%")

    out_path = os.path.join(RESULTS_DIR, f"jahs_meta_tune_{results['timestamp']}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
