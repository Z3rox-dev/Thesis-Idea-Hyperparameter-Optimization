"""JAHS-Bench-201 battery: ALBA_V1 vs Optuna TPE (budget 2000, 10 seeds)."""

import json
import os
import sys
import time
from datetime import datetime

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
RESULTS_FILE = f"{RESULTS_DIR}/jahs_benchmark_{timestamp}.json"


def save_results(obj: dict) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(obj, f, indent=2)



def main():
    print(f"Starting JAHS Benchmark Battery")
    print(f"Tasks: {JAHS_TASKS}")
    print(f"Budget: {BUDGET}, Seeds: {N_SEEDS}")
    print(f"Results file: {RESULTS_FILE}")
    print("=" * 60)

    # Use the existing wrapper already used across the repo.
    sys.path.insert(0, "/mnt/workspace/thesis")
    from benchmark_jahs import JAHSBenchWrapper
    from ALBA_V1 import ALBA as ALBA_V1
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Categorical dims for JAHS in wrapper space.
    # (dim_idx, n_choices)
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]

    results = {
        "config": {
            "budget": BUDGET,
            "checkpoints": CHECKPOINTS,
            "n_seeds": N_SEEDS,
            "seeds": SEEDS,
            "timestamp": timestamp,
        },
        "jahs": {},
    }
    save_results(results)
    
    for task in JAHS_TASKS:
        print(f"\n>>> Running {task.upper()}")

        wrapper = JAHSBenchWrapper(task=task)
        dim = wrapper.dim

        results["jahs"][task] = {
            "dim": dim,
            "alba": {},
            "optuna": {},
            "errors": {},
        }
        
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            start = time.time()
            
            try:
                # ALBA run (minimize error)
                opt = ALBA_V1(
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
                alba_best = float("inf")
                alba_checkpoints = {}

                for it in range(BUDGET):
                    x = opt.ask()
                    y = float(wrapper.evaluate_array(x))
                    opt.tell(x, y)
                    alba_best = min(alba_best, y)

                    if (it + 1) in CHECKPOINTS:
                        alba_checkpoints[it + 1] = float(alba_best)

                results["jahs"][task]["alba"][str(seed)] = alba_checkpoints

                # Optuna run (minimize error)
                wrapper.reset()
                optuna_best = float("inf")
                optuna_checkpoints = {}
                optuna_iter = [0]

                def optuna_objective(trial):
                    x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)], dtype=float)
                    y = float(wrapper.evaluate_array(x))

                    nonlocal optuna_best
                    optuna_best = min(optuna_best, y)

                    optuna_iter[0] += 1
                    if optuna_iter[0] in CHECKPOINTS:
                        optuna_checkpoints[optuna_iter[0]] = float(optuna_best)

                    return y

                sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                study.optimize(optuna_objective, n_trials=BUDGET, show_progress_bar=False)

                results["jahs"][task]["optuna"][str(seed)] = optuna_checkpoints

                alba_final = alba_checkpoints.get(BUDGET, alba_best)
                optuna_final = optuna_checkpoints.get(BUDGET, optuna_best)
                alba_acc = (1.0 - alba_final) * 100
                optuna_acc = (1.0 - optuna_final) * 100
                winner = "ALBA" if alba_final < optuna_final else "Optuna"

                elapsed = time.time() - start
                print(f"ALBA={alba_acc:.2f}% Optuna={optuna_acc:.2f}% -> {winner} ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results["jahs"][task]["errors"][str(seed)] = str(e)
            
            # Save after each seed (even on error)
            save_results(results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY JAHS")
    print("=" * 60)
    
    total_alba = 0
    total_optuna = 0

    for task, task_data in results["jahs"].items():
        alba_wins = 0
        optuna_wins = 0
        ties = 0

        alba_by_seed = task_data.get("alba", {})
        optuna_by_seed = task_data.get("optuna", {})
        err_by_seed = task_data.get("errors", {})

        for seed in SEEDS:
            seed_key = str(seed)
            if seed_key in err_by_seed:
                continue

            alba_cp = alba_by_seed.get(seed_key, {})
            optuna_cp = optuna_by_seed.get(seed_key, {})

            alba_final = alba_cp.get(BUDGET)
            optuna_final = optuna_cp.get(BUDGET)
            if alba_final is None or optuna_final is None:
                continue

            if alba_final < optuna_final - 1e-9:
                alba_wins += 1
            elif optuna_final < alba_final - 1e-9:
                optuna_wins += 1
            else:
                ties += 1

        print(f"{task}: ALBA {alba_wins} - {optuna_wins} Optuna (tie: {ties}, errors: {len(err_by_seed)})")
        total_alba += alba_wins
        total_optuna += optuna_wins
    
    print(f"\nTOTAL: ALBA {total_alba} - {total_optuna} Optuna")
    denom = (total_alba + total_optuna)
    if denom > 0:
        print(f"Win rate: {total_alba/denom*100:.1f}%")
    else:
        print("Win rate: n/a (nessun seed completato)")


if __name__ == "__main__":
    main()
