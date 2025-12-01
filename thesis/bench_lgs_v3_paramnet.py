"""
Benchmark: LGS v3 (non-parametrico) vs LGS v1 vs Main vs Optuna su ParamNet
"""
from __future__ import annotations

import argparse
import sys
import types
from typing import Any, Dict, List, Tuple

import numpy as np

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
from hpo_lgs_v3 import HPOptimizer as LGSv3Optimizer
from hpo_lgs import HPOptimizer as LGSv1Optimizer
from hpo_main import HPOptimizer as MainOptimizer

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


_PARAMNET_STEPS_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
    "mnist": ParamNetMnistOnStepsBenchmark,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark,
    "poker": ParamNetPokerOnStepsBenchmark,
}


def build_paramnet_adapter(dataset: str):
    key = dataset.lower()
    bench_cls = _PARAMNET_STEPS_MAP[key]
    bench = bench_cls()
    cs = bench.get_configuration_space()
    hps = list(cs.values())

    bounds = []
    types = []

    for hp in hps:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("float")
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("int")

    return bench, cs, hps, bounds, types


def xnorm_to_config(x_norm, cs, hps, bounds, types):
    values = {}
    for val, hp, (lo, hi), t in zip(x_norm, hps, bounds, types):
        v = lo + float(val) * (hi - lo)
        if t == "int":
            v = int(round(v))
            v = max(int(hp.lower), min(int(hp.upper), int(v)))
        values[hp.name] = v
    return CS.Configuration(cs, values=values)


def optuna_search(objective, dim, budget, seed):
    if not HAS_OPTUNA:
        return float("nan")

    def _objective(trial):
        x_norm = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
        return float(objective(x_norm))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=int(budget), show_progress_bar=False)
    return float(study.best_value)


def benchmark_paramnet(seeds, budget=200):
    datasets = ["adult", "higgs", "letter", "mnist", "optdigits", "poker"]
    
    print("=" * 90)
    print("  Benchmark: LGS v3 (NON-PARAMETRICO) vs LGS v1 vs Main vs Optuna")
    print("=" * 90)
    print(f"Seeds: {seeds}")
    print(f"Budget: {budget}")
    print("=" * 90)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset}")
        print("=" * 70)
        
        try:
            bench, cs, hps, bounds, types = build_paramnet_adapter(dataset)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        
        dim = len(bounds)
        
        def make_objective(bench, cs, hps, bounds, types):
            def objective(x_norm):
                cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
                res = bench.objective_function(cfg, fidelity={"step": 50})
                return float(res["function_value"])
            return objective
        
        objective = make_objective(bench, cs, hps, bounds, types)
        
        results = {'LGS_v3': [], 'LGS_v1': [], 'Main': [], 'Optuna': []}
        
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            
            # LGS v3 (non-parametrico)
            opt = LGSv3Optimizer(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed)
            _, best = opt.optimize(objective, budget=budget)
            results['LGS_v3'].append(best)
            print(f"    LGS v3: {best:.6f}")
            
            # LGS v1
            opt = LGSv1Optimizer(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed)
            _, best = opt.optimize(objective, budget=budget)
            results['LGS_v1'].append(best)
            print(f"    LGS v1: {best:.6f}")
            
            # Main
            opt = MainOptimizer(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed)
            _, best = opt.optimize(objective, budget=budget)
            results['Main'].append(best)
            print(f"    Main:   {best:.6f}")
            
            # Optuna
            best = optuna_search(objective, dim=dim, budget=budget, seed=seed)
            results['Optuna'].append(best)
            print(f"    Optuna: {best:.6f}")
        
        means = {k: np.mean(v) for k, v in results.items()}
        winner = min(means, key=means.get)
        print(f"\n  → Winner: {winner}")
        
        all_results[dataset] = results
    
    # Summary
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    
    wins = {'LGS_v3': 0, 'LGS_v1': 0, 'Main': 0, 'Optuna': 0}
    
    print(f"\n{'Dataset':<12} {'LGS_v3':<18} {'LGS_v1':<18} {'Main':<18} {'Optuna':<18} {'Winner':<10}")
    print("-" * 100)
    
    for ds_name, results in all_results.items():
        means = {k: np.mean(v) for k, v in results.items()}
        stds = {k: np.std(v) for k, v in results.items()}
        winner = min(means, key=means.get)
        wins[winner] += 1
        
        v3_str = f"{means['LGS_v3']:.4f}±{stds['LGS_v3']:.4f}"
        v1_str = f"{means['LGS_v1']:.4f}±{stds['LGS_v1']:.4f}"
        main_str = f"{means['Main']:.4f}±{stds['Main']:.4f}"
        opt_str = f"{means['Optuna']:.4f}±{stds['Optuna']:.4f}"
        
        print(f"{ds_name:<12} {v3_str:<18} {v1_str:<18} {main_str:<18} {opt_str:<18} {winner:<10}")
    
    print("-" * 100)
    print(f"\n🏆 WINS: LGS_v3={wins['LGS_v3']}, LGS_v1={wins['LGS_v1']}, Main={wins['Main']}, Optuna={wins['Optuna']}")
    
    # Head to head
    print("\n📊 Head-to-head (per seed):")
    h2h = {'v3_vs_opt': [0, 0], 'v3_vs_v1': [0, 0], 'v3_vs_main': [0, 0]}
    for ds_name, results in all_results.items():
        for v3, v1, main, opt in zip(results['LGS_v3'], results['LGS_v1'], results['Main'], results['Optuna']):
            if v3 < opt: h2h['v3_vs_opt'][0] += 1
            elif opt < v3: h2h['v3_vs_opt'][1] += 1
            if v3 < v1: h2h['v3_vs_v1'][0] += 1
            elif v1 < v3: h2h['v3_vs_v1'][1] += 1
            if v3 < main: h2h['v3_vs_main'][0] += 1
            elif main < v3: h2h['v3_vs_main'][1] += 1
    
    print(f"   LGS v3 vs Optuna: {h2h['v3_vs_opt'][0]}-{h2h['v3_vs_opt'][1]}")
    print(f"   LGS v3 vs LGS v1: {h2h['v3_vs_v1'][0]}-{h2h['v3_vs_v1'][1]}")
    print(f"   LGS v3 vs Main:   {h2h['v3_vs_main'][0]}-{h2h['v3_vs_main'][1]}")
    
    return all_results


if __name__ == "__main__":
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
    parser.add_argument('--budget', type=int, default=150)
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    benchmark_paramnet(seeds, args.budget)
