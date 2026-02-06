#!/usr/bin/env python3
"""
Benchmark CopulaHPO vs Optuna TPE vs CMA-ES vs TuRBO vs Random on YAHPO Gym.

YAHPO scenarios with mixed hyperparameter spaces:
- rbv2_xgboost: 12 cont + 2 cat, 119 instances
- iaml_xgboost: 12 cont + 1 cat, 4 instances  
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

# CMA-ES
try:
    from cmaes import CMA
    HAS_CMAES = True
except ImportError:
    HAS_CMAES = False
    print("[WARN] cmaes not installed, skipping CMA-ES")

# TuRBO (BoTorch)
try:
    import torch
    from botorch.models import SingleTaskGP
    # botorch 0.7.x uses fit_gpytorch_model, newer versions use fit_gpytorch_mll
    try:
        from botorch.fit import fit_gpytorch_mll as fit_gp
    except ImportError:
        from botorch.fit import fit_gpytorch_model as fit_gp
    from botorch.acquisition import ExpectedImprovement
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_TURBO = True
except ImportError as e:
    HAS_TURBO = False
    print(f"[WARN] botorch not available: {e}")

# CopulaHPO
sys.path.insert(0, "/mnt/workspace/thesis/nuovo_progetto")
from copula_hpo_v2 import CopulaHPO, HyperparameterSpec

# ALBA Potential (hybrid drilling)
try:
    sys.path.insert(0, "/mnt/workspace/thesis")
    from alba_framework_potential.optimizer import ALBA as ALBA_Potential
    HAS_ALBA = True
except ImportError as e:
    HAS_ALBA = False
    print(f"[WARN] ALBA Potential not available: {e}")


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

    return specs


def copula_config_to_yahpo(x: dict, configspace, skip_params: list[str]) -> dict:
    """Convert CopulaHPO config dict back to YAHPO format."""
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter

    config = {}
    for hp in configspace.values():
        if hp.name in skip_params:
            continue
        if hp.name not in x:
            continue

        val = x[hp.name]
        if isinstance(hp, UniformFloatHyperparameter) and hp.log:
            val = float(np.exp(val))
        config[hp.name] = val

    return config


def get_bounds_info(configspace, skip_params):
    """Get bounds info for continuous optimization methods (CMA-ES, TuRBO)."""
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    
    param_info = []
    for hp in configspace.values():
        if hp.name in skip_params:
            continue
            
        if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
            choices = list(hp.choices) if hasattr(hp, "choices") else list(hp.sequence)
            param_info.append({
                "name": hp.name,
                "type": "categorical",
                "choices": choices,
                "lower": 0,
                "upper": len(choices) - 1,
            })
        elif isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                param_info.append({
                    "name": hp.name,
                    "type": "continuous",
                    "lower": float(np.log(hp.lower)),
                    "upper": float(np.log(hp.upper)),
                    "log": True,
                })
            else:
                param_info.append({
                    "name": hp.name,
                    "type": "continuous",
                    "lower": float(hp.lower),
                    "upper": float(hp.upper),
                    "log": False,
                })
        elif isinstance(hp, UniformIntegerHyperparameter):
            param_info.append({
                "name": hp.name,
                "type": "integer",
                "lower": int(hp.lower),
                "upper": int(hp.upper),
            })
    
    return param_info


def vec_to_config(vec: np.ndarray, param_info: list, configspace, skip_params) -> dict:
    """Convert continuous vector to config dict."""
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter
    
    config = {}
    for i, pi in enumerate(param_info):
        val = vec[i]
        
        if pi["type"] == "categorical":
            idx = int(np.clip(np.round(val), 0, len(pi["choices"]) - 1))
            config[pi["name"]] = pi["choices"][idx]
        elif pi["type"] == "integer":
            config[pi["name"]] = int(np.clip(np.round(val), pi["lower"], pi["upper"]))
        else:  # continuous
            val = float(np.clip(val, pi["lower"], pi["upper"]))
            if pi.get("log", False):
                config[pi["name"]] = float(np.exp(val))
            else:
                config[pi["name"]] = val
    
    return config


def run_copula(bench: BenchmarkSet, instance: Any, n_trials: int, seed: int, mode: str = "elite") -> float:
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

        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance

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

        opt.tell(x, -y)  # CopulaHPO minimizes
        best_y = max(best_y, y)

    return best_y


def run_optuna(bench: BenchmarkSet, instance: Any, n_trials: int, seed: int) -> float:
    """Run Optuna TPE on YAHPO benchmark with improved settings."""
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

        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance

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

    # IMPROVED TPE settings:
    # - multivariate=True: model correlations between hyperparameters
    # - constant_liar=True: better parallel handling
    # - n_startup_trials=10: standard warmup
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        constant_liar=True,
        n_startup_trials=10,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return float(study.best_value)


def run_random(bench: BenchmarkSet, instance: Any, n_trials: int, seed: int) -> float:
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

        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance

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


def run_cmaes(bench: BenchmarkSet, instance: Any, n_trials: int, seed: int) -> float:
    """Run CMA-ES on YAHPO benchmark (relaxed to continuous)."""
    if not HAS_CMAES:
        return 0.0
    
    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    
    param_info = get_bounds_info(cs, skip_params)
    if not param_info:
        return 0.0
    
    d = len(param_info)
    
    # Bounds for CMA-ES
    lower = np.array([pi["lower"] for pi in param_info])
    upper = np.array([pi["upper"] for pi in param_info])
    
    # Initialize at center
    x0 = (lower + upper) / 2
    sigma0 = np.mean(upper - lower) / 4
    
    np.random.seed(seed)
    
    # Population size (default CMA)
    popsize = 4 + int(3 * np.log(d))
    
    cma = CMA(
        mean=x0,
        sigma=sigma0,
        bounds=np.column_stack([lower, upper]),
        seed=seed,
        population_size=popsize,
    )
    
    best_y = float("-inf")
    evals = 0
    
    while evals < n_trials:
        # CMA-ES requires FULL population to be evaluated before tell()
        # Only start a generation if we can complete it
        remaining = n_trials - evals
        if remaining < popsize:
            # Not enough budget for full generation - do random for rest
            for _ in range(remaining):
                x = np.random.uniform(lower, upper)
                config = vec_to_config(x, param_info, cs, skip_params)
                
                if "OpenML_task_id" in [hp.name for hp in cs.values()]:
                    config["OpenML_task_id"] = instance
                elif "task_id" in [hp.name for hp in cs.values()]:
                    config["task_id"] = instance
                
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
                evals += 1
            break
        
        # Sample FULL population
        solutions = []
        for _ in range(popsize):
            x = cma.ask()
            config = vec_to_config(x, param_info, cs, skip_params)
            
            if "OpenML_task_id" in [hp.name for hp in cs.values()]:
                config["OpenML_task_id"] = instance
            elif "task_id" in [hp.name for hp in cs.values()]:
                config["task_id"] = instance
            
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
            
            solutions.append((x, -y))  # CMA minimizes
            best_y = max(best_y, y)
            evals += 1
        
        cma.tell(solutions)
    
    return best_y


def run_turbo(bench: BenchmarkSet, instance: Any, n_trials: int, seed: int) -> float:
    """Run TuRBO-like BO on YAHPO benchmark (simplified version)."""
    if not HAS_TURBO:
        return 0.0
    
    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    
    param_info = get_bounds_info(cs, skip_params)
    if not param_info:
        return 0.0
    
    d = len(param_info)
    
    lower = np.array([pi["lower"] for pi in param_info])
    upper = np.array([pi["upper"] for pi in param_info])
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Start with random samples
    n_init = min(2 * d, 20, n_trials // 2)
    
    X_train = []
    Y_train = []
    best_y = float("-inf")
    
    def eval_config(vec):
        config = vec_to_config(vec, param_info, cs, skip_params)
        
        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance
        
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
    
    # Initial random samples
    for _ in range(n_init):
        x = np.random.uniform(lower, upper)
        y = eval_config(x)
        X_train.append(x)
        Y_train.append(y)
        best_y = max(best_y, y)
    
    # BO loop
    for _ in range(n_trials - n_init):
        X = torch.tensor(np.array(X_train), dtype=torch.double)
        Y = torch.tensor(np.array(Y_train), dtype=torch.double).unsqueeze(-1)
        
        # Normalize to [0,1]
        lb = torch.tensor(lower, dtype=torch.double)
        ub = torch.tensor(upper, dtype=torch.double)
        X_norm = (X - lb) / (ub - lb)
        
        # Standardize Y
        Y_mean = Y.mean()
        Y_std = Y.std() + 1e-6
        Y_norm = (Y - Y_mean) / Y_std
        
        try:
            # Fit GP
            gp = SingleTaskGP(X_norm, Y_norm)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gp(mll)
            
            # EI acquisition
            ei = ExpectedImprovement(gp, best_f=(Y.max() - Y_mean) / Y_std)
            
            # Optimize acquisition (random + local)
            best_acq = float("-inf")
            best_x = None
            
            # Random candidates
            n_cand = 1000
            X_cand = torch.rand(n_cand, d, dtype=torch.double)
            
            with torch.no_grad():
                acq_vals = ei(X_cand.unsqueeze(1))
            
            idx = acq_vals.argmax()
            best_x = X_cand[idx]
            
            # De-normalize
            x_new = (best_x * (ub - lb) + lb).numpy()
            
        except Exception:
            # Fallback to random
            x_new = np.random.uniform(lower, upper)
        
        y_new = eval_config(x_new)
        X_train.append(x_new)
        Y_train.append(y_new)
        best_y = max(best_y, y_new)
    
    return best_y


def run_alba(bench: BenchmarkSet, instance: Any, n_trials: int, seed: int) -> float:
    """Run ALBA Potential (hybrid drilling) on YAHPO benchmark."""
    if not HAS_ALBA:
        return 0.0
    
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    
    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    
    # Build param_space for ALBA
    param_space = {}
    for hp in cs.values():
        if hp.name in skip_params:
            continue
        
        if isinstance(hp, CategoricalHyperparameter):
            param_space[hp.name] = list(hp.choices)
        elif isinstance(hp, OrdinalHyperparameter):
            param_space[hp.name] = list(hp.sequence)
        elif isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                param_space[hp.name] = (float(hp.lower), float(hp.upper), 'log')
            else:
                param_space[hp.name] = (float(hp.lower), float(hp.upper))
        elif isinstance(hp, UniformIntegerHyperparameter):
            param_space[hp.name] = (int(hp.lower), int(hp.upper), 'int')
    
    if not param_space:
        return 0.0
    
    # Create ALBA optimizer
    opt = ALBA_Potential(
        param_space=param_space,
        maximize=True,  # YAHPO returns higher is better
        seed=seed,
        total_budget=n_trials,
        use_potential_field=True,
        use_drilling=True,
    )
    
    best_y = float("-inf")
    
    for _ in range(n_trials):
        config = opt.ask()
        
        # Add fixed params
        if "OpenML_task_id" in [hp.name for hp in cs.values()]:
            config["OpenML_task_id"] = instance
        elif "task_id" in [hp.name for hp in cs.values()]:
            config["task_id"] = instance
        
        # Add fidelity params
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
        
        opt.tell(config, y)
        best_y = max(best_y, y)
    
    return best_y


def benchmark_scenario(
    scenario_name: str,
    n_trials: int = 100,
    n_seeds: int = 3,
    n_instances: int = 3,
    mode: str = "elite",
    methods: list[str] = None,
) -> dict[str, Any]:
    """Run benchmark on a YAHPO scenario."""
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter

    if methods is None:
        methods = ["Copula", "Optuna", "Random"]
        if HAS_CMAES:
            methods.append("CMA-ES")
        if HAS_TURBO:
            methods.append("TuRBO")
        if HAS_ALBA:
            methods.append("ALBA")

    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name} | Trials: {n_trials} | Seeds: {n_seeds}")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'='*70}")

    bench = BenchmarkSet(scenario_name)
    cs = bench.get_opt_space()

    skip_params = ["task_id", "OpenML_task_id", "trainsize", "repl", "epoch"]
    params = [hp for hp in cs.values() if hp.name not in skip_params]
    n_cat = sum(1 for hp in params if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)))
    n_cont = len(params) - n_cat
    print(f"  Params: {len(params)} ({n_cont} cont, {n_cat} cat)")
    print(f"  Instances: {len(bench.instances)} (testing {n_instances})")

    instances = bench.instances[:n_instances]
    results: dict[str, list[float]] = {m: [] for m in methods}
    wins = {m: 0 for m in methods}

    for inst in instances:
        for seed in range(n_seeds):
            print(f"  [{inst}] seed={seed} ... ", end="", flush=True)

            scores = {}
            
            if "Copula" in methods:
                scores["Copula"] = run_copula(bench, inst, n_trials, seed, mode=mode)
            if "Optuna" in methods:
                scores["Optuna"] = run_optuna(bench, inst, n_trials, seed)
            if "Random" in methods:
                scores["Random"] = run_random(bench, inst, n_trials, seed)
            if "CMA-ES" in methods and HAS_CMAES:
                scores["CMA-ES"] = run_cmaes(bench, inst, n_trials, seed)
            if "TuRBO" in methods and HAS_TURBO:
                scores["TuRBO"] = run_turbo(bench, inst, n_trials, seed)
            if "ALBA" in methods and HAS_ALBA:
                scores["ALBA"] = run_alba(bench, inst, n_trials, seed)

            for m in methods:
                if m in scores:
                    results[m].append(scores[m])

            winner = max(scores, key=lambda k: scores[k])
            wins[winner] += 1

            parts = [f"{m}={scores[m]:.4f}" for m in methods if m in scores]
            marker = "✓" if winner == "Copula" else "✗"
            print(f"{' '.join(parts)} [{winner}] {marker}")

    print(f"\n  Summary for {scenario_name}:")
    for method in methods:
        if results[method]:
            mean = np.mean(results[method])
            std = np.std(results[method])
            print(f"    {method:8s}: {mean:.4f} ± {std:.4f} (wins: {wins[method]})")

    return {"results": results, "wins": wins}


def main():
    parser = argparse.ArgumentParser(description="Benchmark CopulaHPO vs others on YAHPO")
    parser.add_argument("--scenarios", type=str, default="rbv2_xgboost,iaml_xgboost",
                        help="Comma-separated YAHPO scenarios")
    parser.add_argument("--budget", type=int, default=200, help="Trials per run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--instances", type=int, default=3, help="Instances per scenario")
    parser.add_argument("--mode", type=str, default="elite", choices=["elite", "latent_cma"],
                        help="CopulaHPO mode")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods (Copula,Optuna,Random,CMA-ES,TuRBO)")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = ["Copula", "Optuna", "Random"]
        if HAS_CMAES:
            methods.append("CMA-ES")
        if HAS_TURBO:
            methods.append("TuRBO")

    print("=" * 70)
    print("YAHPO Gym Benchmark: CopulaHPO vs Optuna vs CMA-ES vs TuRBO vs Random")
    print(f"Methods: {', '.join(methods)}")
    print("=" * 70)

    all_wins: dict[str, int] = {m: 0 for m in methods}

    for scenario in scenarios:
        try:
            result = benchmark_scenario(
                scenario,
                n_trials=args.budget,
                n_seeds=args.seeds,
                n_instances=args.instances,
                mode=args.mode,
                methods=methods,
            )
            for k, v in result["wins"].items():
                all_wins[k] += v
        except Exception as e:
            print(f"  [ERROR] {scenario}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    total = sum(all_wins.values())
    for method in methods:
        pct = 100 * all_wins[method] / total if total > 0 else 0
        print(f"  {method:8s}: {all_wins[method]:3d} wins ({pct:.1f}%)")

    winner = max(all_wins, key=lambda k: all_wins[k])
    print(f"\n  Winner: {winner} 🏆")


if __name__ == "__main__":
    main()
