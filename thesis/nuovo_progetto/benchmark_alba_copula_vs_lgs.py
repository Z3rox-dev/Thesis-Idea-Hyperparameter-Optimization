#!/usr/bin/env python3
"""
Benchmark: ALBA-Copula vs ALBA-LGS vs Optuna su ParamNet surrogates.

Testa se sostituire LGS con Copula migliora le performance su surrogati RF.
"""

from __future__ import annotations

import sys
import numpy as np

# Numpy compatibility
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# sklearn compatibility fix for HPOBench
import sklearn.ensemble
import sklearn.tree
import types

if not hasattr(sklearn.ensemble, 'forest'):
    sklearn.ensemble.forest = types.ModuleType('sklearn.ensemble.forest')
    sklearn.ensemble.forest.RandomForestRegressor = sklearn.ensemble.RandomForestRegressor
    sklearn.ensemble.forest.RandomForestClassifier = sklearn.ensemble.RandomForestClassifier
    sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble.forest

if not hasattr(sklearn.tree, 'tree'):
    sklearn.tree.tree = types.ModuleType('sklearn.tree.tree')
    sklearn.tree.tree.DecisionTreeRegressor = sklearn.tree.DecisionTreeRegressor
    sklearn.tree.tree.DecisionTreeClassifier = sklearn.tree.DecisionTreeClassifier
    sklearn.tree.tree.BaseDecisionTree = sklearn.tree.DecisionTreeClassifier
    sys.modules['sklearn.tree.tree'] = sklearn.tree.tree

sys.path.insert(0, "/mnt/workspace/HPOBench")
sys.path.insert(0, "/mnt/workspace/thesis")

from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
)

from alba_framework_copula.optimizer import ALBA as ALBACopula
from alba_framework_potential.optimizer import ALBA as ALBAOriginal

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)


def get_paramnet_spec():
    """Return ParamNet hyperparameter specification (correct bounds from HPOBench)."""
    return [
        ("initial_lr_log10", "cont", (-6.0, -2.0)),
        ("batch_size_log2", "cont", (3.0, 8.0)),
        ("average_units_per_layer_log2", "cont", (4.0, 8.0)),
        ("final_lr_fraction_log2", "cont", (-4.0, 0.0)),
        ("shape_parameter_1", "cont", (0.0, 1.0)),
        ("num_layers", "int", (1, 5)),
        ("dropout_0", "cont", (0.0, 0.5)),
        ("dropout_1", "cont", (0.0, 0.5)),
    ]


def build_param_space(hp_spec):
    """Build param_space dict for ALBA."""
    param_space = {}
    for name, ptype, bounds in hp_spec:
        if ptype == "cont":
            param_space[name] = (bounds[0], bounds[1])
        elif ptype == "int":
            param_space[name] = list(range(bounds[0], bounds[1] + 1))
    return param_space


def run_benchmark(dataset_name, benchmark_cls, budget=100, seeds=3):
    """Run ALBA-Copula vs ALBA-LGS vs Optuna on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Budget: {budget} | Seeds: {seeds}")
    print(f"{'='*60}")
    
    bench = benchmark_cls(rng=42)
    hp_spec = get_paramnet_spec()
    param_space = build_param_space(hp_spec)
    
    def objective(x_dict):
        config = {}
        for name, ptype, _ in hp_spec:
            if ptype == "int":
                config[name] = int(round(x_dict[name]))
            else:
                config[name] = float(x_dict[name])
        # step is the fidelity parameter (not epoch)
        res = bench.objective_function(config, fidelity={"step": 50})
        return 1.0 - res["function_value"]  # Return error
    
    results = {"ALBA-LGS": [], "ALBA-Copula": [], "Optuna": []}
    
    for seed in range(seeds):
        print(f"  [seed={seed}]", end=" ", flush=True)
        
        # ALBA LGS (original)
        opt = ALBAOriginal(param_space=param_space, seed=seed, maximize=False)
        best_y = float('inf')
        for _ in range(budget):
            x = opt.ask()
            y = objective(x)
            opt.tell(x, y)
            best_y = min(best_y, y)
        results["ALBA-LGS"].append(best_y)
        
        # ALBA Copula (new)
        opt = ALBACopula(param_space=param_space, seed=seed, maximize=False)
        best_y = float('inf')
        for _ in range(budget):
            x = opt.ask()
            y = objective(x)
            opt.tell(x, y)
            best_y = min(best_y, y)
        results["ALBA-Copula"].append(best_y)
        
        # Optuna
        def optuna_obj(trial):
            x = {}
            for name, ptype, bounds in hp_spec:
                if ptype == "cont":
                    x[name] = trial.suggest_float(name, bounds[0], bounds[1])
                elif ptype == "int":
                    x[name] = trial.suggest_int(name, bounds[0], bounds[1])
            return objective(x)
        
        study = optuna.create_study(direction='minimize', 
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(optuna_obj, n_trials=budget, show_progress_bar=False)
        results["Optuna"].append(study.best_value)
        
        print(f"LGS={results['ALBA-LGS'][-1]:.4f} "
              f"Copula={results['ALBA-Copula'][-1]:.4f} "
              f"Optuna={results['Optuna'][-1]:.4f}")
    
    # Summary
    print(f"\n  Summary for {dataset_name}:")
    for method in ["ALBA-LGS", "ALBA-Copula", "Optuna"]:
        vals = results[method]
        print(f"    {method:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    # Count wins
    n = len(results["ALBA-LGS"])
    wins = {m: 0 for m in results.keys()}
    for i in range(n):
        best = min(results.keys(), key=lambda m: results[m][i])
        wins[best] += 1
    
    print(f"  Wins: LGS={wins['ALBA-LGS']}, Copula={wins['ALBA-Copula']}, Optuna={wins['Optuna']}")
    
    return results, wins


def main():
    BUDGET = 100
    SEEDS = 3
    
    print("="*60)
    print("ALBA-Copula vs ALBA-LGS vs Optuna on ParamNet (RF surrogates)")
    print("="*60)
    
    datasets = [
        ("adult", ParamNetAdultOnStepsBenchmark),
        ("higgs", ParamNetHiggsOnStepsBenchmark),
        ("letter", ParamNetLetterOnStepsBenchmark),
    ]
    
    total_wins = {"ALBA-LGS": 0, "ALBA-Copula": 0, "Optuna": 0}
    
    for name, cls in datasets:
        _, wins = run_benchmark(name, cls, BUDGET, SEEDS)
        for m in wins:
            total_wins[m] += wins[m]
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    total = sum(total_wins.values())
    for method in ["ALBA-LGS", "ALBA-Copula", "Optuna"]:
        pct = 100 * total_wins[method] / total
        trophy = "🏆" if total_wins[method] == max(total_wins.values()) else ""
        print(f"  {method:12s}: {total_wins[method]:2d} wins ({pct:5.1f}%) {trophy}")


if __name__ == "__main__":
    main()
