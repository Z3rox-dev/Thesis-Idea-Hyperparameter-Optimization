"""
Benchmark completo V3 vs Optuna su tutti i dataset ParamNet.
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnTimeBenchmark,
    ParamNetHiggsOnTimeBenchmark,
    ParamNetLetterOnTimeBenchmark,
    ParamNetMnistOnTimeBenchmark,
    ParamNetOptdigitsOnTimeBenchmark,
    ParamNetPokerOnTimeBenchmark,
)

from hpo_density_v3 import HPOptimizer

# Configurazione
BENCHMARKS = [
    ('adult', ParamNetAdultOnTimeBenchmark),
    ('higgs', ParamNetHiggsOnTimeBenchmark),
    ('letter', ParamNetLetterOnTimeBenchmark),
    ('mnist', ParamNetMnistOnTimeBenchmark),
    ('optdigits', ParamNetOptdigitsOnTimeBenchmark),
    ('poker', ParamNetPokerOnTimeBenchmark),
]

SEEDS = [0, 1, 2, 3, 4]
N_TRIALS = 100

def setup_benchmark(bench, seed):
    """Prepara bounds e objective per un benchmark."""
    cs = bench.get_configuration_space(seed=seed)
    
    param_names = list(cs.keys())
    bounds = []
    hp_types = []
    hp_log = []
    
    for name in param_names:
        hp = cs[name]
        hp_type = 'float'
        is_log = False
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            lo, hi = hp.lower, hp.upper
            if hasattr(hp, 'log') and hp.log:
                lo, hi = np.log10(lo), np.log10(hi)
                is_log = True
            if 'Integer' in str(type(hp)):
                hp_type = 'int'
            bounds.append((lo, hi))
            hp_types.append(hp_type)
            hp_log.append(is_log)
    
    def objective(x):
        config = {}
        for i, name in enumerate(param_names):
            val = x[i]
            if hp_log[i]:
                val = 10**val
            if hp_types[i] == 'int':
                val = int(round(val))
            config[name] = val
        result = bench.objective_function(config, rng=seed)
        return float(result['function_value'])
    
    return bounds, param_names, hp_types, hp_log, objective


def run_v3(bounds, objective, seed, n_trials):
    """Esegue V3."""
    opt = HPOptimizer(bounds, maximize=False, seed=seed)
    best_x, _ = opt.optimize(objective, budget=n_trials)
    return objective(best_x)


def run_optuna(bounds, param_names, hp_types, objective, seed, n_trials):
    """Esegue Optuna TPE."""
    def optuna_objective(trial):
        x = []
        for i, name in enumerate(param_names):
            lo, hi = bounds[i]
            if hp_types[i] == 'int':
                val = trial.suggest_int(name, int(lo), int(hi))
            else:
                val = trial.suggest_float(name, lo, hi)
            x.append(val)
        return objective(np.array(x))
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_value


def main():
    print("=" * 80)
    print(f"BENCHMARK V3 vs OPTUNA")
    print(f"Trials: {N_TRIALS}, Seeds: {SEEDS}")
    print("=" * 80)
    
    results = {}
    
    for bench_name, BenchClass in BENCHMARKS:
        print(f"\n{bench_name.upper()}:")
        results[bench_name] = {'v3': [], 'optuna': []}
        
        for seed in SEEDS:
            bench = BenchClass(rng=42)
            bounds, param_names, hp_types, hp_log, objective = setup_benchmark(bench, seed)
            
            # V3
            v3_result = run_v3(bounds, objective, seed, N_TRIALS)
            results[bench_name]['v3'].append(v3_result)
            
            # Optuna
            optuna_result = run_optuna(bounds, param_names, hp_types, objective, seed, N_TRIALS)
            results[bench_name]['optuna'].append(optuna_result)
            
            winner = "V3" if v3_result < optuna_result else ("TIE" if v3_result == optuna_result else "Optuna")
            print(f"  Seed {seed}: V3={v3_result:.5f}, Optuna={optuna_result:.5f} -> {winner}")
    
    # Riepilogo
    print("\n" + "=" * 80)
    print("RIEPILOGO")
    print("=" * 80)
    print(f"{'Dataset':<12} | {'V3 Mean':>12} | {'Optuna Mean':>12} | {'V3 Wins':>8} | {'Winner':>8}")
    print("-" * 70)
    
    total_v3_wins = 0
    total_tests = 0
    
    for bench_name, _ in BENCHMARKS:
        v3_mean = np.mean(results[bench_name]['v3'])
        v3_std = np.std(results[bench_name]['v3'])
        optuna_mean = np.mean(results[bench_name]['optuna'])
        optuna_std = np.std(results[bench_name]['optuna'])
        
        v3_wins = sum(1 for v, o in zip(results[bench_name]['v3'], results[bench_name]['optuna']) if v < o)
        total_v3_wins += v3_wins
        total_tests += len(SEEDS)
        
        winner = "V3" if v3_mean < optuna_mean else "Optuna"
        print(f"{bench_name:<12} | {v3_mean:>12.5f} | {optuna_mean:>12.5f} | {v3_wins:>4}/{len(SEEDS):<3} | {winner:>8}")
    
    print("-" * 70)
    print(f"{'TOTALE':<12} | {'':>12} | {'':>12} | {total_v3_wins:>4}/{total_tests:<3} |")
    print("=" * 80)


if __name__ == "__main__":
    main()
