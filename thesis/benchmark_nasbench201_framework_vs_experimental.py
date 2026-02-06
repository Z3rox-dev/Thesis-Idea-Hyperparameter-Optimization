#!/usr/bin/env python3
"""NASBench-201 benchmark: alba_framework vs alba_experimental.

Goal: stress-test discrete-heavy optimization on Neural Architecture Search.
Runs alba_framework vs alba_experimental on NASBench-201 (tabular NAS).

- Datasets: cifar10-valid, cifar100, ImageNet16-120
- Seeds: configurable (default 70-73)
- Budget: configurable (default 400)

Outputs a JSON file in thesis/benchmark_results/.

Usage:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_nasbench201_framework_vs_experimental.py --budget 400 --seeds 70-73
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

# Reduce noisy warning spam from dependencies.
warnings.filterwarnings("ignore")


# Make HPOBench importable without installing.
sys.path.insert(0, "/mnt/workspace/HPOBench")
sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework import ALBA as ALBA_FRAMEWORK  # noqa: E402
from ALBA_V1_experimental import ALBA as ALBA_EXP  # noqa: E402


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def seed_to_nasbench_data_seed(seed: int) -> int:
    # NASBench-201 supports only {777, 888, 999}. Map deterministically.
    options = [777, 888, 999]
    return options[seed % 3]


def detect_categorical_dims(hps: List[Any]) -> List[Tuple[int, int]]:
    categorical_dims: List[Tuple[int, int]] = []
    for i, hp in enumerate(hps):
        # ConfigSpace categoricals have .choices; ordinals have .sequence.
        if hasattr(hp, "choices") and hp.choices is not None:
            categorical_dims.append((i, len(list(hp.choices))))
        elif hasattr(hp, "sequence") and hp.sequence is not None:
            categorical_dims.append((i, len(list(hp.sequence))))
    return categorical_dims


def config_from_x(cs, hps: List[Any], x: np.ndarray) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    for i, hp in enumerate(hps):
        name = hp.name

        if hasattr(hp, "choices") and hp.choices is not None:
            choices = list(hp.choices)
            idx = int(np.round(float(x[i]) * (len(choices) - 1)))
            idx = max(0, min(len(choices) - 1, idx))
            cfg[name] = choices[idx]
        elif hasattr(hp, "sequence") and hp.sequence is not None:
            seq = list(hp.sequence)
            idx = int(np.round(float(x[i]) * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            cfg[name] = seq[idx]
        elif hasattr(hp, "lower") and hasattr(hp, "upper"):
            lo, hi = float(hp.lower), float(hp.upper)
            val = lo + float(x[i]) * (hi - lo)
            # Try to keep ints as ints when appropriate.
            if "int" in str(type(hp)).lower():
                val = int(np.round(val))
            cfg[name] = val
        else:
            # Constant or unsupported type: sample default via ConfigSpace.
            pass

    # Validate via ConfigSpace
    import ConfigSpace as CS
    return CS.Configuration(cs, values=cfg).get_dictionary()


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description="NASBench-201: alba_framework vs alba_experimental")
    p.add_argument("--datasets", default="cifar10-valid,cifar100,ImageNet16-120")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--seeds", default="70-73", help="Comma list and/or ranges, e.g. '70-73' or '70,71,72'")
    p.add_argument("--epoch", type=int, default=200, help="NASBench-201 epoch fidelity")
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = _parse_seeds(args.seeds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/nasbench201_framework_exp_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json"

    from hpobench.benchmarks.nas.nasbench_201 import NasBench201BaseBenchmark

    print("=" * 78)
    print("NASBench-201 benchmark: alba_framework vs alba_experimental")
    print(f"Budget={args.budget} | Seeds={seeds} | Epoch={args.epoch}")
    print(f"Datasets={datasets}")
    print(f"Results: {out_path}")
    print("=" * 78)
    print()

    results: Dict[str, Any] = {
        "config": {
            "budget": args.budget,
            "seeds": seeds,
            "datasets": datasets,
            "epoch": args.epoch,
            "timestamp": timestamp,
        },
        "datasets": {},
    }
    save_results(out_path, results)

    for dataset in datasets:
        print(f"== {dataset} ==")

        bench = NasBench201BaseBenchmark(dataset=dataset, rng=0)
        cs = bench.get_configuration_space()
        hps = list(cs.get_hyperparameters())
        dim = len(hps)
        categorical_dims = detect_categorical_dims(hps)

        results["datasets"][dataset] = {
            "dim": dim,
            "n_cat": len(categorical_dims),
            "categorical_dims": categorical_dims,
            "framework": {},
            "experimental": {},
        }

        for seed in seeds:
            t0 = time.time()
            data_seed = seed_to_nasbench_data_seed(seed)
            fidelity = {"epoch": args.epoch}

            # ALBA framework
            opt_framework = ALBA_FRAMEWORK(
                bounds=[(0.0, 1.0)] * dim,
                maximize=False,
                seed=seed,
                total_budget=args.budget,
                categorical_dims=categorical_dims,
                split_depth_max=8,
                global_random_prob=0.05,
                stagnation_threshold=50,
            )

            best_framework = float("inf")
            for it in range(args.budget):
                x = opt_framework.ask()
                cfg_dict = config_from_x(cs, hps, x)
                result = bench.objective_function(
                    configuration=cfg_dict,
                    fidelity=fidelity,
                    rng=seed,
                    data_seed=data_seed,
                )
                score = float(result["function_value"])
                opt_framework.tell(x, score)
                best_framework = min(best_framework, score)

            results["datasets"][dataset]["framework"][str(seed)] = float(best_framework)

            # ALBA experimental
            opt_exp = ALBA_EXP(
                bounds=[(0.0, 1.0)] * dim,
                maximize=False,
                seed=seed,
                total_budget=args.budget,
                categorical_dims=categorical_dims,
                split_depth_max=8,
                global_random_prob=0.05,
                stagnation_threshold=50,
            )

            best_exp = float("inf")
            for it in range(args.budget):
                x = opt_exp.ask()
                cfg_dict = config_from_x(cs, hps, x)
                result = bench.objective_function(
                    configuration=cfg_dict,
                    fidelity=fidelity,
                    rng=seed,
                    data_seed=data_seed,
                )
                score = float(result["function_value"])
                opt_exp.tell(x, score)
                best_exp = min(best_exp, score)

            results["datasets"][dataset]["experimental"][str(seed)] = float(best_exp)
            save_results(out_path, results)

            print(f"seed {seed}")
            pair = [
                ("alba_framework", float(best_framework)),
                ("alba_experimental", float(best_exp)),
            ]
            pair.sort(key=lambda t: t[1])
            for name, score in pair:
                print(f"  {name} {score:.6f}")
        print()

    print(f"✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
