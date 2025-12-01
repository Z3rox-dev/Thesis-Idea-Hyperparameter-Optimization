"""
Benchmark: HPO-LGS vs Main vs Optuna su ParamNet

Confronto tra:
- LGS: Local Geometry Score (surrogato geometrico)
- Main: Surrogato quadratico tradizionale
- Optuna TPE: Baseline stato dell'arte
"""
from __future__ import annotations

import argparse
import sys
import types
from typing import Any, Dict, List, Tuple

import numpy as np

# Compatibilità con vecchie dipendenze che usano np.float / np.int / np.bool
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import ConfigSpace as CS

sys.path.insert(0, '/mnt/workspace/HPOBench')

from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetMnistOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)
from hpo_lgs import HPOptimizer as LGSOptimizer
from hpo_main import HPOptimizer as MainOptimizer

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  Optuna non installato")


_PARAMNET_STEPS_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
    "mnist": ParamNetMnistOnStepsBenchmark,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark,
    "poker": ParamNetPokerOnStepsBenchmark,
}


def build_paramnet_adapter(
    dataset: str,
) -> Tuple[Any, CS.ConfigurationSpace, List[CS.Hyperparameter], List[Tuple[float, float]], List[str]]:
    """Costruisce benchmark ParamNet(dataset) + mapping da [0,1]^d a ConfigSpace."""
    key = dataset.lower()
    if key not in _PARAMNET_STEPS_MAP:
        raise ValueError(f"Dataset ParamNet non supportato: {dataset}")
    bench_cls = _PARAMNET_STEPS_MAP[key]
    bench = bench_cls()
    cs = bench.get_configuration_space()
    hps = cs.get_hyperparameters()

    bounds: List[Tuple[float, float]] = []
    types: List[str] = []

    for hp in hps:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("float")
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("int")
        else:
            raise ValueError(f"Unsupported hyperparameter type: {hp} ({type(hp)})")

    return bench, cs, hps, bounds, types


def xnorm_to_config(
    x_norm: np.ndarray,
    cs: CS.ConfigurationSpace,
    hps: List[CS.Hyperparameter],
    bounds: List[Tuple[float, float]],
    types: List[str],
) -> CS.Configuration:
    """Mappa un vettore x_norm in [0,1]^d a una Configuration di ConfigSpace."""
    x_norm = np.asarray(x_norm, dtype=float)
    values: Dict[str, Any] = {}

    for val, hp, (lo, hi), t in zip(x_norm, hps, bounds, types):
        v = lo + float(val) * (hi - lo)
        if t == "int":
            v = int(round(v))
            v = max(int(hp.lower), min(int(hp.upper), int(v)))
        values[hp.name] = v

    return CS.Configuration(cs, values=values)


def optuna_search(objective: Any, dim: int, budget: int, seed: int) -> float:
    """Optuna TPE su [0,1]^dim per minimizzare l'obiettivo."""
    if not HAS_OPTUNA:
        return float("nan")

    def _objective(trial: "optuna.trial.Trial") -> float:
        x_norm = np.array(
            [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)],
            dtype=float,
        )
        return float(objective(x_norm))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=int(budget), show_progress_bar=False)
    return float(study.best_value)


def benchmark_paramnet(seeds, budget=200):
    """Benchmark LGS vs Main vs Optuna su tutti i dataset ParamNet."""
    datasets = ["adult", "higgs", "letter", "mnist", "optdigits", "poker"]
    
    print("=" * 80)
    print("  Benchmark: LGS vs Main vs Optuna su ParamNet")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Budget per seed: {budget}")
    print(f"Datasets: {', '.join(datasets)}")
    print("=" * 80)
    
    all_results: Dict[str, Dict[str, List[float]]] = {}
    
    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset}")
        print("=" * 60)
        
        try:
            bench, cs, hps, bounds, types = build_paramnet_adapter(dataset)
        except Exception as exc:
            print(f"  ERROR loading dataset: {exc}")
            continue
        
        dim = len(bounds)
        print(f"  Dimensions: {dim}")
        
        def make_objective(bench, cs, hps, bounds, types):
            def objective(x_norm: np.ndarray) -> float:
                cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
                res = bench.objective_function(cfg, fidelity={"step": 50})
                return float(res["function_value"])
            return objective
        
        objective = make_objective(bench, cs, hps, bounds, types)
        
        results = {'LGS': [], 'Main': [], 'Optuna': []}
        
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            
            # LGS
            opt_lgs = LGSOptimizer(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed)
            _, best_lgs = opt_lgs.optimize(objective, budget=budget)
            results['LGS'].append(best_lgs)
            print(f"    LGS:    {best_lgs:.6f}")
            
            # Main
            opt_main = MainOptimizer(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed)
            _, best_main = opt_main.optimize(objective, budget=budget)
            results['Main'].append(best_main)
            print(f"    Main:   {best_main:.6f}")
            
            # Optuna
            best_optuna = optuna_search(objective, dim=dim, budget=budget, seed=seed)
            results['Optuna'].append(best_optuna)
            print(f"    Optuna: {best_optuna:.6f}")
        
        # Summary per dataset
        print(f"\n  {dataset} Summary:")
        for name, vals in results.items():
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"    {name}: {mean:.6f} +/- {std:.6f}")
        
        # Winner
        means = {k: np.mean(v) for k, v in results.items()}
        winner = min(means, key=means.get)
        print(f"    → Winner: {winner}")
        
        all_results[dataset] = results
    
    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    
    wins = {'LGS': 0, 'Main': 0, 'Optuna': 0}
    
    print(f"\n{'Dataset':<12} {'LGS':<20} {'Main':<20} {'Optuna':<20} {'Winner':<10}")
    print("-" * 85)
    
    for ds_name, results in all_results.items():
        means = {k: np.mean(v) for k, v in results.items()}
        stds = {k: np.std(v) for k, v in results.items()}
        winner = min(means, key=means.get)
        wins[winner] += 1
        
        lgs_str = f"{means['LGS']:.4f}+/-{stds['LGS']:.4f}"
        main_str = f"{means['Main']:.4f}+/-{stds['Main']:.4f}"
        optuna_str = f"{means['Optuna']:.4f}+/-{stds['Optuna']:.4f}"
        
        print(f"{ds_name:<12} {lgs_str:<20} {main_str:<20} {optuna_str:<20} {winner:<10}")
    
    print("-" * 85)
    print(f"\nWins: LGS={wins['LGS']}, Main={wins['Main']}, Optuna={wins['Optuna']}")
    
    return all_results


if __name__ == "__main__":
    # Patch per scikit-learn < 0.24
    try:
        import sklearn.ensemble
        import sklearn.tree

        if "sklearn.ensemble.forest" not in sys.modules:
            forest = types.ModuleType("sklearn.ensemble.forest")
            for attr in dir(sklearn.ensemble):
                setattr(forest, attr, getattr(sklearn.ensemble, attr))
            sys.modules["sklearn.ensemble.forest"] = forest

        if "sklearn.tree.tree" not in sys.modules:
            tree_mod = types.ModuleType("sklearn.tree.tree")
            for attr in dir(sklearn.tree):
                setattr(tree_mod, attr, getattr(sklearn.tree, attr))
            sys.modules["sklearn.tree.tree"] = tree_mod
    except ImportError:
        pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default="42,123,456")
    parser.add_argument('--budget', type=int, default=200)
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    benchmark_paramnet(seeds, args.budget)
