#!/usr/bin/env python3
"""JAHS-Bench-201 long-horizon benchmark: ALBA_V1 vs meta variants.

This benchmark is meant to better reflect the "learn well, then money comes" story:
we compare learning curves at a larger budget, not just short-horizon final score.

Intended environment: conda `py39`.

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_jahs_long_horizon.py --task cifar10 --budget 5000 --seeds 42 43 44 45 46 --meta_variant lexi_topk

Resume:
  python thesis/benchmark_jahs_long_horizon.py --results_file <existing.json>
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_alba_from_path(module_name: str, file_path: str):
    import importlib.util

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


def _default_checkpoints(budget: int) -> List[int]:
    # Keep it sparse for long runs.
    cands = [100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000]
    return [c for c in cands if c <= budget] + ([budget] if budget not in cands else [])


def _is_seed_done(task_data: Dict[str, Any], algo_key: str, seed: int, budget: int) -> bool:
    seed_key = str(seed)
    cp = task_data.get(algo_key, {}).get(seed_key, {})
    if not cp:
        return False
    return str(budget) in cp or budget in cp


def save_results(results_file: str, obj: Dict[str, Any]) -> None:
    tmp = results_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, results_file)


def load_results_if_any(results_file: str) -> Dict[str, Any]:
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}


def _meta_kwargs(variant: str) -> Dict[str, Any]:
    if variant == "lexi_topk":
        return dict(
            leaf_selection_mode="lexi_topk",
            leaf_reward_topk=5,
            cat_selection_mode="lexi_topk",
            cat_reward_topk=3,
        )
    if variant == "default":
        return {}
    raise ValueError(f"Unknown meta_variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="JAHS long-horizon benchmark")
    parser.add_argument("--task", type=str, default="cifar10", choices=["cifar10", "fashion_mnist", "colorectal_histology"]) 
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--meta_variant", type=str, default="lexi_topk", choices=["default", "lexi_topk"])

    # Optional overrides for lexi_topk
    parser.add_argument("--leaf_reward_topk", type=int, default=None)
    parser.add_argument("--cat_reward_topk", type=int, default=None)
    parser.add_argument("--passion_during_study", action="store_true", help="During studying, ignore reward filtering and follow knowledge/interest")

    # Self-adapting controller (online)
    parser.add_argument("--controller_enabled", action="store_true", help="Enable online study/exploit controller")
    parser.add_argument("--controller_check_every", type=int, default=25)
    parser.add_argument("--controller_min_history", type=int, default=4)
    parser.add_argument("--controller_quantile_low", type=float, default=0.40)
    parser.add_argument("--controller_quantile_high", type=float, default=0.60)
    parser.add_argument("--controller_guardrail_init", type=float, default=0.50)
    parser.add_argument("--controller_guardrail_step", type=float, default=0.10)
    parser.add_argument("--controller_guardrail_min", type=float, default=0.20)
    parser.add_argument("--controller_guardrail_max", type=float, default=0.90)

    # Adaptive study->exploit switch (budget-aware)
    parser.add_argument("--adaptive_budget", action="store_true", help="Enable budget-adaptive study->exploit switch")
    parser.add_argument("--study_budget_fraction", type=float, default=0.25)
    parser.add_argument("--study_budget_min", type=int, default=50)
    parser.add_argument("--study_budget_max_fraction", type=float, default=0.60)

    # Adaptive study->exploit switch (performance-aware)
    parser.add_argument("--adaptive_performance", action="store_true", help="Stop studying early if no improvement")
    parser.add_argument("--performance_patience", type=int, default=75)

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable periodic optimizer debug prints")
    parser.add_argument("--debug_every", type=int, default=100)
    args = parser.parse_args()

    task = args.task
    budget = int(args.budget)
    checkpoints = args.checkpoints if args.checkpoints is not None else _default_checkpoints(budget)
    seeds = list(args.seeds)
    meta_variant = args.meta_variant
    leaf_reward_topk = args.leaf_reward_topk
    cat_reward_topk = args.cat_reward_topk
    passion_during_study = bool(args.passion_during_study)
    controller_enabled = bool(args.controller_enabled)
    controller_check_every = int(args.controller_check_every)
    controller_min_history = int(args.controller_min_history)
    controller_quantile_low = float(args.controller_quantile_low)
    controller_quantile_high = float(args.controller_quantile_high)
    controller_guardrail_init = float(args.controller_guardrail_init)
    controller_guardrail_step = float(args.controller_guardrail_step)
    controller_guardrail_min = float(args.controller_guardrail_min)
    controller_guardrail_max = float(args.controller_guardrail_max)
    adaptive_budget = bool(args.adaptive_budget)
    study_budget_fraction = float(args.study_budget_fraction)
    study_budget_min = int(args.study_budget_min)
    study_budget_max_fraction = float(args.study_budget_max_fraction)

    adaptive_performance = bool(args.adaptive_performance)
    performance_patience = int(args.performance_patience)

    debug = bool(args.debug)
    debug_every = int(args.debug_every)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.results_file or os.path.join(
        RESULTS_DIR,
        f"jahs_long_{task}_{meta_variant}_b{budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json",
    )

    print("JAHS LONG-HORIZON")
    print(f"Task: {task}")
    print(f"Budget: {budget}")
    print(f"Seeds: {seeds}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Meta variant: {meta_variant}")
    if meta_variant == "lexi_topk":
        print(f"Top-K: leaf_reward_topk={leaf_reward_topk if leaf_reward_topk is not None else 5}  cat_reward_topk={cat_reward_topk if cat_reward_topk is not None else 3}")
        print(f"Passion during study: {passion_during_study}")
    print(f"Adaptive budget: {adaptive_budget} (study_frac={study_budget_fraction}, min={study_budget_min}, max_frac={study_budget_max_fraction})")
    print(f"Adaptive performance: {adaptive_performance} (patience={performance_patience})")
    print(
        "Controller: "
        f"{controller_enabled} (every={controller_check_every}, min_hist={controller_min_history}, "
        f"q_low={controller_quantile_low}, q_high={controller_quantile_high}, "
        f"guard_init={controller_guardrail_init}, step={controller_guardrail_step}, "
        f"guard_min={controller_guardrail_min}, guard_max={controller_guardrail_max})"
    )
    print(f"Debug: {debug} (every={debug_every})")
    print(f"Results: {results_file}")
    print("=" * 70)

    sys.path.insert(0, "/mnt/workspace/thesis")
    from benchmark_jahs import JAHSBenchWrapper
    from ALBA_V1 import ALBA as ALBA_V1

    ALBA_META = _load_alba_from_path(
        module_name="alba_meta_copy2_long",
        file_path="/mnt/workspace/thesis/ALBA_V1 copy 2.py",
    )

    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]

    base_meta_kwargs = _meta_kwargs(meta_variant)
    if meta_variant == "lexi_topk":
        if leaf_reward_topk is not None:
            base_meta_kwargs["leaf_reward_topk"] = int(leaf_reward_topk)
        if cat_reward_topk is not None:
            base_meta_kwargs["cat_reward_topk"] = int(cat_reward_topk)
        base_meta_kwargs["passion_during_study"] = passion_during_study

    existing = load_results_if_any(results_file)
    if existing:
        results = existing
        results.setdefault("config", {})
        results["config"].update(
            {
                "task": task,
                "budget": budget,
                "checkpoints": checkpoints,
                "seeds": seeds,
                "meta_variant": meta_variant,
                "meta_kwargs": {
                    **base_meta_kwargs,
                    "adaptive_budget": adaptive_budget,
                    "study_budget_fraction": study_budget_fraction,
                    "study_budget_min": study_budget_min,
                    "study_budget_max_fraction": study_budget_max_fraction,
                    "adaptive_performance": adaptive_performance,
                    "performance_patience": performance_patience,
                    "controller_enabled": controller_enabled,
                    "controller_check_every": controller_check_every,
                    "controller_min_history": controller_min_history,
                    "controller_quantile_low": controller_quantile_low,
                    "controller_quantile_high": controller_quantile_high,
                    "controller_guardrail_init": controller_guardrail_init,
                    "controller_guardrail_step": controller_guardrail_step,
                    "controller_guardrail_min": controller_guardrail_min,
                    "controller_guardrail_max": controller_guardrail_max,
                    "debug": debug,
                    "debug_every": debug_every,
                },
            }
        )
        results.setdefault("jahs", {})
        results["jahs"].setdefault(task, {})
        print("Resuming existing results file")
    else:
        results = {
            "config": {
                "task": task,
                "budget": budget,
                "checkpoints": checkpoints,
                "seeds": seeds,
                "meta_variant": meta_variant,
                "meta_kwargs": {
                    **base_meta_kwargs,
                    "adaptive_budget": adaptive_budget,
                    "study_budget_fraction": study_budget_fraction,
                    "study_budget_min": study_budget_min,
                    "study_budget_max_fraction": study_budget_max_fraction,
                    "adaptive_performance": adaptive_performance,
                    "performance_patience": performance_patience,
                    "controller_enabled": controller_enabled,
                    "controller_check_every": controller_check_every,
                    "controller_min_history": controller_min_history,
                    "controller_quantile_low": controller_quantile_low,
                    "controller_quantile_high": controller_quantile_high,
                    "controller_guardrail_init": controller_guardrail_init,
                    "controller_guardrail_step": controller_guardrail_step,
                    "controller_guardrail_min": controller_guardrail_min,
                    "controller_guardrail_max": controller_guardrail_max,
                    "debug": debug,
                    "debug_every": debug_every,
                },
                "timestamp": timestamp,
            },
            "jahs": {task: {}},
        }
        save_results(results_file, results)

    wrapper = JAHSBenchWrapper(task=task)
    dim = wrapper.dim

    task_bucket = results["jahs"].setdefault(task, {})
    task_bucket.setdefault("dim", dim)
    task_bucket.setdefault("alba_v1", {})
    meta_key = "alba_meta" if meta_variant == "default" else f"alba_meta_{meta_variant}"
    task_bucket.setdefault(meta_key, {})

    for seed in seeds:
        if _is_seed_done(task_bucket, "alba_v1", seed, budget) and _is_seed_done(task_bucket, meta_key, seed, budget):
            print(f"Seed {seed}: SKIP (done)")
            continue

        print(f"\nSeed {seed}")
        start = time.time()

        # --- V1 ---
        opt_v1 = ALBA_V1(
            bounds=[(0.0, 1.0)] * dim,
            maximize=False,
            seed=seed,
            split_depth_max=8,
            total_budget=budget,
            global_random_prob=0.05,
            stagnation_threshold=50,
            categorical_dims=categorical_dims,
        )

        wrapper.reset()
        v1_best = float("inf")
        v1_cp: Dict[int, float] = {}
        for it in range(budget):
            x = opt_v1.ask()
            y = float(wrapper.evaluate_array(x))
            opt_v1.tell(x, y)
            v1_best = min(v1_best, y)
            if (it + 1) in checkpoints:
                v1_cp[it + 1] = float(v1_best)
        task_bucket["alba_v1"][str(seed)] = v1_cp
        save_results(results_file, results)

        # --- META ---
        opt_meta = ALBA_META(
            bounds=[(0.0, 1.0)] * dim,
            maximize=False,
            seed=seed,
            split_depth_max=8,
            total_budget=budget,
            global_random_prob=0.05,
            stagnation_threshold=50,
            categorical_dims=categorical_dims,
            **{
                **base_meta_kwargs,
                "adaptive_budget": adaptive_budget,
                "study_budget_fraction": study_budget_fraction,
                "study_budget_min": study_budget_min,
                "study_budget_max_fraction": study_budget_max_fraction,
                "adaptive_performance": adaptive_performance,
                "performance_patience": performance_patience,
                "controller_enabled": controller_enabled,
                "controller_check_every": controller_check_every,
                "controller_min_history": controller_min_history,
                "controller_quantile_low": controller_quantile_low,
                "controller_quantile_high": controller_quantile_high,
                "controller_guardrail_init": controller_guardrail_init,
                "controller_guardrail_step": controller_guardrail_step,
                "controller_guardrail_min": controller_guardrail_min,
                "controller_guardrail_max": controller_guardrail_max,
                "debug": debug,
                "debug_every": debug_every,
            },
        )

        wrapper.reset()
        meta_best = float("inf")
        meta_cp: Dict[int, float] = {}
        for it in range(budget):
            x = opt_meta.ask()
            y = float(wrapper.evaluate_array(x))
            opt_meta.tell(x, y)
            meta_best = min(meta_best, y)
            if (it + 1) in checkpoints:
                meta_cp[it + 1] = float(meta_best)
        task_bucket[meta_key][str(seed)] = meta_cp
        save_results(results_file, results)

        elapsed = time.time() - start
        v1_final = v1_cp.get(budget, v1_best)
        meta_final = meta_cp.get(budget, meta_best)
        print(
            f"Done seed {seed}: V1={(1.0 - v1_final) * 100:.2f}%  META={(1.0 - meta_final) * 100:.2f}%  time={elapsed:.1f}s"
        )

    # Summary
    v1_finals = []
    meta_finals = []
    for seed in seeds:
        v1_cp = task_bucket.get("alba_v1", {}).get(str(seed), {})
        m_cp = task_bucket.get(meta_key, {}).get(str(seed), {})
        if str(budget) in v1_cp and str(budget) in m_cp:
            v1_finals.append(float(v1_cp[str(budget)]))
            meta_finals.append(float(m_cp[str(budget)]))
        elif budget in v1_cp and budget in m_cp:
            v1_finals.append(float(v1_cp[budget]))
            meta_finals.append(float(m_cp[budget]))

    if v1_finals and meta_finals and len(v1_finals) == len(meta_finals):
        v1_mean = float(np.mean(v1_finals))
        m_mean = float(np.mean(meta_finals))
        print("\nSUMMARY")
        print(f"V1   mean_err={v1_mean:.6f}  mean_acc={(1.0 - v1_mean) * 100:.2f}%")
        print(f"META mean_err={m_mean:.6f}  mean_acc={(1.0 - m_mean) * 100:.2f}%")
        print(f"Saved: {results_file}")


if __name__ == "__main__":
    main()
