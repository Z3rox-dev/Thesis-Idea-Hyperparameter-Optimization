#!/usr/bin/env python3
"""
Benchmark HPOBench (ParamNet + NN Tabular) con 5 metodi:
- Copula (CopulaHPO v2)
- Optuna (TPE multivariate)
- Random
- CMA-ES
- ALBA Potential

Uso: python benchmark_hpobench_full.py --model paramnet --budget 400 --seeds 3
     python benchmark_hpobench_full.py --model nn --budget 400 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

# Numpy compatibility
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# sklearn compatibility fix for HPOBench (sklearn.ensemble.forest was removed in sklearn 1.0)
import sklearn.ensemble
import sklearn.tree
import types

if not hasattr(sklearn.ensemble, 'forest'):
    # Create a mock module with the needed classes
    sklearn.ensemble.forest = types.ModuleType('sklearn.ensemble.forest')
    sklearn.ensemble.forest.RandomForestRegressor = sklearn.ensemble.RandomForestRegressor
    sklearn.ensemble.forest.RandomForestClassifier = sklearn.ensemble.RandomForestClassifier
    sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble.forest

if not hasattr(sklearn.tree, 'tree'):
    sklearn.tree.tree = types.ModuleType('sklearn.tree.tree')
    sklearn.tree.tree.DecisionTreeRegressor = sklearn.tree.DecisionTreeRegressor
    sklearn.tree.tree.DecisionTreeClassifier = sklearn.tree.DecisionTreeClassifier
    sklearn.tree.tree.BaseDecisionTree = sklearn.tree.DecisionTreeClassifier  # Fallback
    sys.modules['sklearn.tree.tree'] = sklearn.tree.tree

import ConfigSpace as CS

# HPOBench imports
sys.path.insert(0, "/mnt/workspace/HPOBench")

try:
    from hpobench.benchmarks.surrogates.paramnet_benchmark import (
        ParamNetAdultOnStepsBenchmark,
        ParamNetHiggsOnStepsBenchmark,
        ParamNetLetterOnStepsBenchmark,
        ParamNetMnistOnStepsBenchmark,
        ParamNetOptdigitsOnStepsBenchmark,
        ParamNetPokerOnStepsBenchmark,
    )
    PARAMNET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ParamNet not available: {e}")
    PARAMNET_AVAILABLE = False

try:
    from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
    NN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TabularBenchmark (NN) not available: {e}")
    NN_AVAILABLE = False

# Our optimizers
from copula_hpo_v2 import CopulaHPO, HyperparameterSpec

# Import ALBA as a package
sys.path.insert(0, "/mnt/workspace/thesis")
from alba_framework_potential.optimizer import ALBA as ALBAOptimizer

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False


# =============================================================================
# ParamNet datasets
# =============================================================================

_PARAMNET_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark if PARAMNET_AVAILABLE else None,
    "higgs": ParamNetHiggsOnStepsBenchmark if PARAMNET_AVAILABLE else None,
    "letter": ParamNetLetterOnStepsBenchmark if PARAMNET_AVAILABLE else None,
    "mnist": ParamNetMnistOnStepsBenchmark if PARAMNET_AVAILABLE else None,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark if PARAMNET_AVAILABLE else None,
    "poker": ParamNetPokerOnStepsBenchmark if PARAMNET_AVAILABLE else None,
}

# NN task IDs (OpenML)
_NN_TASKS = {
    "credit-g": 31,      # German Credit
    "vehicle": 53,       # Vehicle
    "kc1": 3917,         # KC1 Software defect
    "phoneme": 9952,     # Phoneme
    "blood": 10101,      # Blood Transfusion
}


def build_paramnet_benchmark(dataset: str) -> Tuple[Any, List[HyperparameterSpec], int]:
    """Build ParamNet benchmark and return (bench, param_specs, dim)."""
    if not PARAMNET_AVAILABLE:
        raise RuntimeError("ParamNet not available")
    
    key = dataset.lower()
    if key not in _PARAMNET_MAP or _PARAMNET_MAP[key] is None:
        raise ValueError(f"Unknown ParamNet dataset: {dataset}")
    
    bench = _PARAMNET_MAP[key]()
    cs = bench.get_configuration_space()
    hps = cs.get_hyperparameters()
    
    # Build param specs
    specs = []
    for hp in hps:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="continuous",
                bounds=(float(hp.lower), float(hp.upper))
            ))
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="integer",
                bounds=(int(hp.lower), int(hp.upper))
            ))
        elif isinstance(hp, CS.CategoricalHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="categorical",
                bounds=list(hp.choices)
            ))
        elif isinstance(hp, CS.OrdinalHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="categorical",
                bounds=list(hp.sequence)
            ))
        else:
            raise ValueError(f"Unsupported HP type: {type(hp)}")
    
    return bench, cs, specs


def build_nn_benchmark(task_id: int) -> Tuple[Any, Any, List[HyperparameterSpec]]:
    """Build NN TabularBenchmark and return (bench, cs, param_specs)."""
    if not NN_AVAILABLE:
        raise RuntimeError("TabularBenchmark (NN) not available")
    
    bench = TabularBenchmark(model="nn", task_id=task_id)
    cs = bench.get_configuration_space()
    hps = cs.get_hyperparameters()
    
    specs = []
    for hp in hps:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="continuous",
                bounds=(float(hp.lower), float(hp.upper))
            ))
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="integer",
                bounds=(int(hp.lower), int(hp.upper))
            ))
        elif isinstance(hp, CS.CategoricalHyperparameter):
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="categorical",
                bounds=list(hp.choices)
            ))
        elif isinstance(hp, CS.OrdinalHyperparameter):
            # Treat ordinal as categorical
            specs.append(HyperparameterSpec(
                name=hp.name,
                type="categorical",
                bounds=list(hp.sequence)
            ))
        else:
            raise ValueError(f"Unsupported HP type: {type(hp)}")
    
    return bench, cs, specs


def spec_to_config(x: dict, cs: CS.ConfigurationSpace) -> CS.Configuration:
    """Convert our dict to ConfigSpace Configuration."""
    values = {}
    for hp in cs.get_hyperparameters():
        if hp.name in x:
            val = x[hp.name]
            if isinstance(hp, CS.UniformIntegerHyperparameter):
                val = int(round(val))
                val = max(hp.lower, min(hp.upper, val))
            elif isinstance(hp, CS.OrdinalHyperparameter):
                # Ordinal: value must be in the sequence
                if val not in hp.sequence:
                    # Find closest
                    val = hp.sequence[0]
            values[hp.name] = val
    return CS.Configuration(cs, values=values)


# =============================================================================
# Optimizers
# =============================================================================

def run_copula(objective, specs: List[HyperparameterSpec], budget: int, seed: int) -> float:
    """Run CopulaHPO."""
    opt = CopulaHPO(specs, seed=seed)
    for _ in range(budget):
        x = opt.ask()
        y = objective(x)
        opt.tell(x, y)
    return opt.best_y


def run_optuna(objective, specs: List[HyperparameterSpec], budget: int, seed: int) -> float:
    """Run Optuna TPE."""
    def optuna_objective(trial):
        x = {}
        for spec in specs:
            if spec.type == "continuous":
                x[spec.name] = trial.suggest_float(spec.name, spec.bounds[0], spec.bounds[1])
            elif spec.type == "integer":
                x[spec.name] = trial.suggest_int(spec.name, spec.bounds[0], spec.bounds[1])
            elif spec.type == "categorical":
                x[spec.name] = trial.suggest_categorical(spec.name, spec.bounds)
        return objective(x)
    
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
    return study.best_value


def run_random(objective, specs: List[HyperparameterSpec], budget: int, seed: int) -> float:
    """Run Random Search."""
    rng = np.random.default_rng(seed)
    best_y = float("inf")
    
    for _ in range(budget):
        x = {}
        for spec in specs:
            if spec.type == "continuous":
                x[spec.name] = rng.uniform(spec.bounds[0], spec.bounds[1])
            elif spec.type == "integer":
                x[spec.name] = rng.integers(spec.bounds[0], spec.bounds[1] + 1)
            elif spec.type == "categorical":
                x[spec.name] = rng.choice(spec.bounds)
        y = objective(x)
        if y < best_y:
            best_y = y
    return best_y


def run_cmaes(objective, specs: List[HyperparameterSpec], budget: int, seed: int) -> float:
    """Run CMA-ES (continuous only, maps categoricals to integers)."""
    if not CMA_AVAILABLE:
        return float("nan")
    
    # Build bounds for CMA-ES
    dim = len(specs)
    lower = []
    upper = []
    
    for spec in specs:
        if spec.type == "continuous":
            lower.append(spec.bounds[0])
            upper.append(spec.bounds[1])
        elif spec.type == "integer":
            lower.append(float(spec.bounds[0]))
            upper.append(float(spec.bounds[1]))
        elif spec.type == "categorical":
            lower.append(0.0)
            upper.append(float(len(spec.bounds) - 1))
    
    lower = np.array(lower)
    upper = np.array(upper)
    
    def cma_objective(x_arr):
        x = {}
        for i, spec in enumerate(specs):
            val = x_arr[i]
            if spec.type == "continuous":
                x[spec.name] = float(np.clip(val, spec.bounds[0], spec.bounds[1]))
            elif spec.type == "integer":
                x[spec.name] = int(np.clip(round(val), spec.bounds[0], spec.bounds[1]))
            elif spec.type == "categorical":
                idx = int(np.clip(round(val), 0, len(spec.bounds) - 1))
                x[spec.name] = spec.bounds[idx]
        return objective(x)
    
    x0 = (lower + upper) / 2
    sigma0 = np.mean(upper - lower) / 4
    
    opts = {
        "bounds": [lower.tolist(), upper.tolist()],
        "seed": seed,
        "maxfevals": budget,
        "verbose": -9,
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    while not es.stop() and es.result.evaluations < budget:
        solutions = es.ask()
        fitnesses = [cma_objective(s) for s in solutions]
        es.tell(solutions, fitnesses)
    
    return es.result.fbest


def run_alba(objective, specs: List[HyperparameterSpec], budget: int, seed: int) -> float:
    """Run ALBA Potential with proper param_space for categoricals."""
    # Build param_space for ALBA (proper typed space)
    param_space = {}
    
    for spec in specs:
        if spec.type == "continuous":
            param_space[spec.name] = (spec.bounds[0], spec.bounds[1])
        elif spec.type == "integer":
            # Use list of integers for proper handling
            param_space[spec.name] = list(range(int(spec.bounds[0]), int(spec.bounds[1]) + 1))
        elif spec.type == "categorical":
            param_space[spec.name] = list(spec.bounds)
    
    opt = ALBAOptimizer(
        param_space=param_space,
        seed=seed,
        maximize=False,
        total_budget=budget,
    )
    
    best_y = float('inf')
    for _ in range(budget):
        x = opt.ask()  # Returns dict with proper types
        y = objective(x)
        opt.tell(x, y)
        best_y = min(best_y, y)
    
    return best_y


# =============================================================================
# Main benchmark
# =============================================================================

def run_benchmark(
    model: str,
    datasets: List[str],
    budget: int,
    seeds: List[int],
    methods: List[str],
) -> Dict[str, Dict[str, List[float]]]:
    """Run benchmark across datasets, seeds, methods."""
    
    results = {}  # dataset -> method -> list of best_y
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset} | Budget: {budget} | Seeds: {len(seeds)}")
        print(f"Methods: {', '.join(methods)}")
        print("="*70)
        
        # Build benchmark
        try:
            if model == "paramnet":
                bench, cs, specs = build_paramnet_benchmark(dataset)
                fidelity = {"step": 50}
            else:  # nn
                if dataset in _NN_TASKS:
                    task_id = _NN_TASKS[dataset]
                else:
                    task_id = int(dataset)
                bench, cs, specs = build_nn_benchmark(task_id)
                fidelity = {}  # NN uses default fidelity
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        dim = len(specs)
        n_cat = sum(1 for s in specs if s.type == "categorical")
        n_int = sum(1 for s in specs if s.type == "integer")
        n_cont = dim - n_cat - n_int
        print(f"  Params: {dim} ({n_cont} cont, {n_int} int, {n_cat} cat)")
        
        results[dataset] = {m: [] for m in methods}
        
        def objective(x: dict) -> float:
            cfg = spec_to_config(x, cs)
            if model == "paramnet":
                res = bench.objective_function(cfg, fidelity=fidelity)
            else:
                res = bench.objective_function(cfg)
            return float(res["function_value"])
        
        for seed in seeds:
            method_results = {}
            
            for method in methods:
                try:
                    if method == "Copula":
                        best_y = run_copula(objective, specs, budget, seed)
                    elif method == "Optuna":
                        best_y = run_optuna(objective, specs, budget, seed)
                    elif method == "Random":
                        best_y = run_random(objective, specs, budget, seed)
                    elif method == "CMA-ES":
                        best_y = run_cmaes(objective, specs, budget, seed)
                    elif method == "ALBA":
                        best_y = run_alba(objective, specs, budget, seed)
                    else:
                        best_y = float("nan")
                except Exception as e:
                    print(f"    ERROR {method}: {e}")
                    best_y = float("nan")
                
                method_results[method] = best_y
                results[dataset][method].append(best_y)
            
            # Print results for this seed
            result_str = " ".join([f"{m}={method_results[m]:.4f}" for m in methods])
            
            # Find winner
            valid = [(m, method_results[m]) for m in methods if np.isfinite(method_results[m])]
            if valid:
                winner = min(valid, key=lambda x: x[1])[0]
            else:
                winner = "?"
            
            print(f"  [seed={seed}] {result_str} [{winner}]")
        
        # Print summary for this dataset
        print(f"\n  Summary for {dataset}:")
        method_stats = []
        for method in methods:
            vals = np.array(results[dataset][method])
            valid_vals = vals[np.isfinite(vals)]
            if len(valid_vals) > 0:
                mean_val = valid_vals.mean()
                std_val = valid_vals.std()
                
                # Count wins
                wins = 0
                for i in range(len(seeds)):
                    scores = {m: results[dataset][m][i] for m in methods if np.isfinite(results[dataset][m][i])}
                    if scores and method in scores:
                        if scores[method] == min(scores.values()):
                            wins += 1
                
                method_stats.append((method, mean_val, std_val, wins))
                print(f"    {method:8s}: {mean_val:.4f} ± {std_val:.4f} (wins: {wins})")
            else:
                print(f"    {method:8s}: N/A")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="HPOBench benchmark with 5 methods")
    parser.add_argument("--model", type=str, default="paramnet", 
                        choices=["paramnet", "nn"],
                        help="Benchmark type: paramnet or nn")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated datasets or 'all'")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--methods", type=str, default="Copula,Optuna,Random,CMA-ES,ALBA")
    args = parser.parse_args()
    
    # Parse datasets
    if args.model == "paramnet":
        all_datasets = ["adult", "higgs", "letter", "mnist", "optdigits", "poker"]
    else:
        all_datasets = list(_NN_TASKS.keys())
    
    if args.datasets == "all":
        datasets = all_datasets
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
    
    seeds = list(range(args.seeds))
    methods = [m.strip() for m in args.methods.split(",")]
    
    print("="*70)
    print(f"HPOBench Benchmark: {args.model.upper()}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Budget: {args.budget} | Seeds: {args.seeds}")
    print(f"Methods: {', '.join(methods)}")
    print("="*70)
    
    results = run_benchmark(args.model, datasets, args.budget, seeds, methods)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    total_wins = {m: 0 for m in methods}
    total_runs = 0
    
    for dataset in results:
        for i in range(len(seeds)):
            scores = {}
            for m in methods:
                if i < len(results[dataset][m]) and np.isfinite(results[dataset][m][i]):
                    scores[m] = results[dataset][m][i]
            if scores:
                winner = min(scores, key=scores.get)
                total_wins[winner] += 1
                total_runs += 1
    
    for method in methods:
        pct = 100 * total_wins[method] / total_runs if total_runs > 0 else 0
        marker = " 🏆" if total_wins[method] == max(total_wins.values()) else ""
        print(f"  {method:8s}: {total_wins[method]:3d} wins ({pct:5.1f}%){marker}")
    
    if total_wins:
        winner = max(total_wins, key=total_wins.get)
        print(f"\n  Winner: {winner} 🏆")


if __name__ == "__main__":
    main()
