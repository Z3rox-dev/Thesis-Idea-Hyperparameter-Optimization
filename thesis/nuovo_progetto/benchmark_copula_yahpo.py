#!/usr/bin/env python3
"""
Benchmark CopulaHPO (elite) vs Optuna vs Random on YAHPO Gym.

YAHPO scenarios with mixed hyperparameter spaces:
- rbv2_xgboost: 12 cont + 2 cat, 119 instances
- iaml_xgboost: 12 cont + 1 cat, 4 instances  
- rbv2_ranger: 5 cont + 3 cat, 119 instances
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# YAHPO Gym
import yahpo_gym
from yahpo_gym import BenchmarkSet, local_config

local_config.set_data_path("/root/.yahpo_gym_data/")

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# CopulaHPO
sys.path.insert(0, "/mnt/workspace/thesis/nuovo_progetto")
from copula_hpo_v2 import CopulaHPO, HyperparameterSpec


def build_copula_specs_from_yahpo(configspace, skip_params=None) -> list[HyperparameterSpec]:
    """Convert YAHPO ConfigSpace to CopulaHPO specs."""
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    if skip_params is None:
        skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]

    specs = []
    for hp in configspace.values():
        if hp.name in skip_params:
            continue

        if isinstance(hp, CategoricalHyperparameter):
            specs.append(
                HyperparameterSpec(
                    name=hp.name,
                    type="categorical",
                    bounds=list(hp.choices),
                )
            )
        elif isinstance(hp, OrdinalHyperparameter):
            specs.append(
                HyperparameterSpec(
                    name=hp.name,
                    type="categorical",
                    bounds=list(hp.sequence),
                )
            )
        elif isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                # Log-transform bounds
                specs.append(
                    HyperparameterSpec(
                        name=hp.name,
                        type="continuous",
                        bounds=(float(np.log(hp.lower)), float(np.log(hp.upper))),
                    )
                )
            else:
                specs.append(
                    HyperparameterSpec(
                        name=hp.name,
                        type="continuous",
                        bounds=(float(hp.lower), float(hp.upper)),
                    )
                )
        elif isinstance(hp, UniformIntegerHyperparameter):
            specs.append(
                HyperparameterSpec(
                    name=hp.name,
                    type="integer",
                    bounds=(int(hp.lower), int(hp.upper)),
                )
            )
        else:
            print(f"  [WARN] Skipping unsupported HP type: {hp.name} ({type(hp).__name__})")

    return specs


def copula_config_to_yahpo(
    x: dict,
    configspace,
    skip_params: list[str],
) -> dict:
    """Convert CopulaHPO config dict back to YAHPO format (handle log transforms)."""
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter

    config = {}
    for hp in configspace.values():
        if hp.name in skip_params:
            continue
        if hp.name not in x:
            continue

        val = x[hp.name]

        # Inverse log transform for log-scale floats
        if isinstance(hp, UniformFloatHyperparameter) and hp.log:
            val = float(np.exp(val))

        config[hp.name] = val

    return config


def run_copula(
    bench: BenchmarkSet,
    instance: Any,
    n_trials: int,
    seed: int,
    mode: str = "elite",
) -> float:
    """Run CopulaHPO on YAHPO benchmark."""
    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]

    specs = build_copula_specs_from_yahpo(cs, skip_params)
    if not specs:
        return 0.0

    opt = CopulaHPO(specs, mode=mode, seed=seed, budget=n_trials)
    best_y = float("-inf")

    for _ in range(n_trials):
        x = opt.ask()
        config = copula_config_to_yahpo(x, cs, skip_params)

        # Add instance info
        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance

        # Add fidelity params at max
        if hasattr(bench.config, "fidelity_params"):
            for fp in bench.config.fidelity_params:
                if fp == "epoch":
                    config["epoch"] = 52
                elif fp == "trainsize":
                    config["trainsize"] = 1.0
                elif fp == "repl":
                    config["repl"] = 10

        try:
            result = bench.objective_function(config)
            y = result[0][obj_name]
        except Exception:
            y = 0.0

        # CopulaHPO minimizes; YAHPO maximizes accuracy/AUC
        opt.tell(x, -y)
        best_y = max(best_y, y)

    return best_y


def run_optuna(
    bench: BenchmarkSet,
    instance: Any,
    n_trials: int,
    seed: int,
) -> float:
    """Run Optuna TPE on YAHPO benchmark."""
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    conditions = cs.get_conditions()

    def objective(trial: optuna.Trial) -> float:
        config: dict[str, Any] = {}

        # First: sample non-conditional
        for hp in cs.values():
            if hp.name in skip_params:
                continue
            is_conditional = any(
                hasattr(c, "child") and c.child.name == hp.name for c in conditions
            )
            if is_conditional:
                continue

            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                choices = list(hp.choices) if hasattr(hp, "choices") else list(hp.sequence)
                config[hp.name] = trial.suggest_categorical(hp.name, choices)
            elif isinstance(hp, UniformFloatHyperparameter):
                config[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper, log=hp.log)
            elif isinstance(hp, UniformIntegerHyperparameter):
                log = hp.log if hasattr(hp, "log") else False
                config[hp.name] = trial.suggest_int(hp.name, hp.lower, hp.upper, log=log)

        # Second: conditional
        for cond in conditions:
            if hasattr(cond, "child") and hasattr(cond, "parent"):
                child_hp = cond.child
                parent_name = cond.parent.name
                if parent_name not in config:
                    continue
                is_active = False
                if hasattr(cond, "value"):
                    is_active = config[parent_name] == cond.value
                elif hasattr(cond, "values"):
                    is_active = config[parent_name] in cond.values
                if is_active:
                    if isinstance(child_hp, UniformFloatHyperparameter):
                        config[child_hp.name] = trial.suggest_float(
                            child_hp.name, child_hp.lower, child_hp.upper, log=child_hp.log
                        )
                    elif isinstance(child_hp, UniformIntegerHyperparameter):
                        log = child_hp.log if hasattr(child_hp, "log") else False
                        config[child_hp.name] = trial.suggest_int(
                            child_hp.name, child_hp.lower, child_hp.upper, log=log
                        )
                    elif isinstance(child_hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                        choices = (
                            list(child_hp.choices)
                            if hasattr(child_hp, "choices")
                            else list(child_hp.sequence)
                        )
                        config[child_hp.name] = trial.suggest_categorical(child_hp.name, choices)

        # Instance
        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance

        # Fidelity
        if hasattr(bench.config, "fidelity_params"):
            for fp in bench.config.fidelity_params:
                if fp == "epoch":
                    config["epoch"] = 52
                elif fp == "trainsize":
                    config["trainsize"] = 1.0
                elif fp == "repl":
                    config["repl"] = 10

        try:
            result = bench.objective_function(config)
            return float(result[0][obj_name])
        except Exception:
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return float(study.best_value)


def run_random(
    bench: BenchmarkSet,
    instance: Any,
    n_trials: int,
    seed: int,
) -> float:
    """Run Random Search on YAHPO benchmark."""
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    rng = np.random.default_rng(seed)

    best_y = float("-inf")

    for _ in range(n_trials):
        config: dict[str, Any] = {}
        for hp in cs.values():
            if hp.name in skip_params:
                continue
            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                choices = list(hp.choices) if hasattr(hp, "choices") else list(hp.sequence)
                config[hp.name] = rng.choice(choices)
            elif isinstance(hp, UniformFloatHyperparameter):
                if hp.log:
                    log_val = rng.uniform(np.log(hp.lower), np.log(hp.upper))
                    config[hp.name] = float(np.exp(log_val))
                else:
                    config[hp.name] = float(rng.uniform(hp.lower, hp.upper))
            elif isinstance(hp, UniformIntegerHyperparameter):
                config[hp.name] = int(rng.integers(hp.lower, hp.upper + 1))

        # Instance
        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance

        # Fidelity
        if hasattr(bench.config, "fidelity_params"):
            for fp in bench.config.fidelity_params:
                if fp == "epoch":
                    config["epoch"] = 52
                elif fp == "trainsize":
                    config["trainsize"] = 1.0
                elif fp == "repl":
                    config["repl"] = 10

        try:
            result = bench.objective_function(config)
            y = float(result[0][obj_name])
        except Exception:
            y = 0.0

        best_y = max(best_y, y)

    return best_y


def benchmark_scenario(
    scenario_name: str,
    n_trials: int = 100,
    n_seeds: int = 3,
    n_instances: int = 3,
    mode: str = "elite",
) -> dict[str, Any]:
    """Run benchmark on a YAHPO scenario."""
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name} | Trials: {n_trials} | Seeds: {n_seeds}")
    print(f"{'='*70}")

    bench = BenchmarkSet(scenario_name)
    cs = bench.get_opt_space()

    # Info
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    params = [hp for hp in cs.values() if hp.name not in skip_params]
    n_cat = sum(1 for hp in params if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)))
    n_cont = len(params) - n_cat
    print(f"  Params: {len(params)} ({n_cont} cont, {n_cat} cat)")
    print(f"  Instances: {len(bench.instances)} (testing {n_instances})")

    instances = bench.instances[:n_instances]
    results: dict[str, list[float]] = {"Copula": [], "Optuna": [], "Random": []}
    wins = {"Copula": 0, "Optuna": 0, "Random": 0}

    for inst in instances:
        for seed in range(n_seeds):
            print(f"  [{inst}] seed={seed} ... ", end="", flush=True)

            copula_y = run_copula(bench, inst, n_trials, seed, mode=mode)
            optuna_y = run_optuna(bench, inst, n_trials, seed)
            random_y = run_random(bench, inst, n_trials, seed)

            results["Copula"].append(copula_y)
            results["Optuna"].append(optuna_y)
            results["Random"].append(random_y)

            # Determine winner
            scores = {"Copula": copula_y, "Optuna": optuna_y, "Random": random_y}
            winner = max(scores, key=lambda k: scores[k])
            wins[winner] += 1

            marker = "✓" if winner == "Copula" else ("≈" if copula_y >= optuna_y else "✗")
            print(f"Copula={copula_y:.4f} Optuna={optuna_y:.4f} Random={random_y:.4f} {marker}")

    print(f"\n  Summary for {scenario_name}:")
    for method in ["Copula", "Optuna", "Random"]:
        mean = np.mean(results[method])
        std = np.std(results[method])
        print(f"    {method:8s}: {mean:.4f} ± {std:.4f} (wins: {wins[method]})")

    return {"results": results, "wins": wins}


def main():
    parser = argparse.ArgumentParser(description="Benchmark CopulaHPO vs Optuna vs Random on YAHPO")
    parser.add_argument("--scenarios", type=str, default="rbv2_xgboost,iaml_xgboost",
                        help="Comma-separated YAHPO scenarios")
    parser.add_argument("--budget", type=int, default=200, help="Trials per run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--instances", type=int, default=3, help="Instances per scenario")
    parser.add_argument("--mode", type=str, default="elite", choices=["elite", "latent_cma"],
                        help="CopulaHPO mode")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]

    print("=" * 70)
    print("YAHPO Gym Benchmark: CopulaHPO (elite) vs Optuna vs Random")
    print("=" * 70)

    all_wins: dict[str, int] = {"Copula": 0, "Optuna": 0, "Random": 0}

    for scenario in scenarios:
        try:
            result = benchmark_scenario(
                scenario,
                n_trials=args.budget,
                n_seeds=args.seeds,
                n_instances=args.instances,
                mode=args.mode,
            )
            for k, v in result["wins"].items():
                all_wins[k] += v
        except Exception as e:
            print(f"  [ERROR] {scenario}: {e}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    total = sum(all_wins.values())
    for method in ["Copula", "Optuna", "Random"]:
        pct = 100 * all_wins[method] / total if total > 0 else 0
        print(f"  {method:8s}: {all_wins[method]:3d} wins ({pct:.1f}%)")

    winner = max(all_wins, key=lambda k: all_wins[k])
    print(f"\n  Winner: {winner} 🏆")


if __name__ == "__main__":
    main()
