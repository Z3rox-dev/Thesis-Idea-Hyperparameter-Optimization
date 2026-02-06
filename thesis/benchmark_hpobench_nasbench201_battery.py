#!/usr/bin/env python3
"""Battery benchmark on HPOBench NASBench-201.

Goal: stress-test discrete-heavy optimization.
Runs ALBA_V1 vs Optuna TPE on NASBench-201 (tabular NAS) across multiple datasets.

- Datasets: cifar10-valid, cifar100, ImageNet16-120
- Seeds: 10 (42..51) mapped to NASBench data_seed in {777, 888, 999}
- Budget: 2000 evals
- Checkpoints: [100, 250, 500, 1000, 1500, 2000]

Outputs a JSON file in thesis/benchmark_results/.

Note: this script expects HPOBench to be importable via sys.path.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np


# Make HPOBench importable without installing.
sys.path.insert(0, "/mnt/workspace/HPOBench")
sys.path.insert(0, "/mnt/workspace/thesis")


BUDGET = 2000
CHECKPOINTS = [100, 250, 500, 1000, 1500, 2000]
N_SEEDS = 10
SEEDS = list(range(42, 42 + N_SEEDS))

DATASETS = ["cifar10-valid", "cifar100", "ImageNet16-120"]

RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = f"{RESULTS_DIR}/hpobench_nasbench201_battery_{TIMESTAMP}.json"


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
            # Safer fallback: do not set it and let ConfigSpace fill defaults.
            pass

    # Validate via ConfigSpace (deterministic; avoids injecting random defaults)
    import ConfigSpace as CS
    return CS.Configuration(cs, values=cfg).get_dictionary()


def save_results(all_results: dict) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def main() -> None:
    from ALBA_V1 import ALBA as ALBA_V1
    import optuna
    from hpobench.benchmarks.nas.nasbench_201 import NasBench201BaseBenchmark

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    print("=" * 80)
    print("HPOBench NASBench-201 battery: ALBA_V1 vs Optuna TPE")
    print(f"Budget={BUDGET} | Seeds={SEEDS} | Checkpoints={CHECKPOINTS}")
    print(f"Datasets={DATASETS}")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 80)

    all_results: Dict[str, Any] = {
        "config": {
            "budget": BUDGET,
            "checkpoints": CHECKPOINTS,
            "seeds": SEEDS,
            "datasets": DATASETS,
            "timestamp": TIMESTAMP,
        },
        "nasbench201": {},
    }
    save_results(all_results)

    for dataset in DATASETS:
        print(f"\n>>> Dataset: {dataset}")

        bench = NasBench201BaseBenchmark(dataset=dataset, rng=0)
        cs = bench.get_configuration_space()
        hps = list(cs.get_hyperparameters())
        dim = len(hps)
        categorical_dims = detect_categorical_dims(hps)

        ds_results: Dict[str, Any] = {
            "dim": dim,
            "n_cat": len(categorical_dims),
            "categorical_dims": categorical_dims,
            "objective": "function_value (valid_error = 100 - valid_acc)",
            "maximize": False,
            "seeds": {},
        }

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)

            data_seed = seed_to_nasbench_data_seed(seed)
            fidelity = {"epoch": 200}

            # ALBA
            alba = ALBA_V1(
                bounds=[(0.0, 1.0)] * dim,
                maximize=False,
                seed=seed,
                total_budget=BUDGET,
                categorical_dims=categorical_dims,
            )

            alba_best = float("inf")
            alba_checkpoints: Dict[int, float] = {}

            for it in range(BUDGET):
                x = alba.ask()
                cfg_dict = config_from_x(cs, hps, x)
                result = bench.objective_function(
                    configuration=cfg_dict,
                    fidelity=fidelity,
                    rng=seed,
                    data_seed=data_seed,
                )
                # HPOBench NASBench-201 returns valid error: (100 - valid_accuracy). Lower is better.
                score = float(result["function_value"])

                alba.tell(x, score)
                alba_best = min(alba_best, score)

                if (it + 1) in CHECKPOINTS:
                    alba_checkpoints[it + 1] = float(alba_best)

            # Optuna
            optuna_best = float("inf")
            optuna_checkpoints: Dict[int, float] = {}
            optuna_iter = [0]

            def optuna_objective(trial):
                x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)], dtype=float)
                cfg_dict = config_from_x(cs, hps, x)
                result = bench.objective_function(
                    configuration=cfg_dict,
                    fidelity=fidelity,
                    rng=seed,
                    data_seed=data_seed,
                )
                val = float(result["function_value"])

                nonlocal optuna_best
                optuna_best = min(optuna_best, val)

                optuna_iter[0] += 1
                if optuna_iter[0] in CHECKPOINTS:
                    optuna_checkpoints[optuna_iter[0]] = float(optuna_best)

                return val

            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(optuna_objective, n_trials=BUDGET, show_progress_bar=False)

            alba_final = alba_checkpoints.get(BUDGET, alba_best)
            optuna_final = optuna_checkpoints.get(BUDGET, optuna_best)
            winner = "ALBA" if alba_final < optuna_final else "Optuna"
            print(f"ALBA(err)={alba_final:.4f} Optuna(err)={optuna_final:.4f} -> {winner}")

            ds_results["seeds"][str(seed)] = {
                "data_seed": data_seed,
                "alba": {"checkpoints": alba_checkpoints, "final": float(alba_final)},
                "optuna": {"checkpoints": optuna_checkpoints, "final": float(optuna_final)},
            }

            all_results["nasbench201"][dataset] = ds_results
            save_results(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
