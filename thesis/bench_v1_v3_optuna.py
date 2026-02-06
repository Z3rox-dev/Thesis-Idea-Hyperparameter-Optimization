#!/usr/bin/env python3
"""
Benchmark ALBA_V1 vs ALBA_V3 vs Optuna on JAHS-Bench-201
Seeds: 100-109, Budget: 2000
"""

import sys
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_jahs import JAHSBenchWrapper, run_optuna_tpe
from ALBA_V1 import ALBA as ALBA_V1
from ALBA_V3 import ALBA as ALBA_V3


def run_alba_v1(wrapper, n_evals, seed):
    dim = wrapper.dim
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]
    opt = ALBA_V1(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
    )
    wrapper.reset()
    best_error = float('inf')
    history = []
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_error = min(best_error, y)
        history.append(best_error)
    return history, best_error


def run_alba_v3(wrapper, n_evals, seed):
    dim = wrapper.dim
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
    ]
    opt = ALBA_V3(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=categorical_dims,
    )
    wrapper.reset()
    best_error = float('inf')
    history = []
    for _ in range(n_evals):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        best_error = min(best_error, y)
        history.append(best_error)
    return history, best_error


if __name__ == '__main__':
    # Config
    TASKS = ['cifar10', 'fashion_mnist', 'colorectal_histology']
    BUDGET = 2000
    SEEDS = list(range(100, 110))  # 10 seeds: 100-109
    OPTIMIZERS = {
        'Optuna': run_optuna_tpe,
        'ALBA_V1': run_alba_v1,
        'ALBA_V3': run_alba_v3,
    }

    print('='*80)
    print('JAHS-BENCH-201 BENCHMARK: ALBA_V1 vs ALBA_V3 vs Optuna')
    print('='*80)
    print(f'Budget: {BUDGET}')
    print(f'Seeds: {SEEDS}')
    print(f'Tasks: {TASKS}')
    print('='*80)

    all_results = {}

    for task in TASKS:
        print(f'\n>>> TASK: {task} <<<')
        wrapper = JAHSBenchWrapper(task=task)
        
        all_results[task] = {}
        
        for opt_name, opt_func in OPTIMIZERS.items():
            print(f'\n  Running {opt_name}...')
            errors = []
            times = []
            
            for seed in SEEDS:
                start = time.time()
                try:
                    history, final_error = opt_func(wrapper, BUDGET, seed)
                    elapsed = time.time() - start
                    errors.append(final_error)
                    times.append(elapsed)
                    acc = (1 - final_error) * 100
                    print(f'    Seed {seed}: acc={acc:.2f}%, time={elapsed:.1f}s')
                except Exception as e:
                    print(f'    Seed {seed}: FAILED - {e}')
                    errors.append(float('nan'))
            
            valid_errors = [e for e in errors if not np.isnan(e)]
            if valid_errors:
                mean_err = np.mean(valid_errors)
                std_err = np.std(valid_errors)
                mean_acc = (1 - mean_err) * 100
                std_acc = std_err * 100
                all_results[task][opt_name] = {
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'errors': errors
                }
                print(f'  {opt_name}: {mean_acc:.2f}% +/- {std_acc:.2f}%')

    # Summary
    print('\n' + '='*80)
    print('FINAL SUMMARY')
    print('='*80)

    # Per-task results
    for task in TASKS:
        print(f'\n{task}:')
        ranked = sorted(all_results[task].items(), key=lambda x: -x[1]['mean_acc'])
        for rank, (name, data) in enumerate(ranked, 1):
            print(f'  {rank}. {name:12s}: {data["mean_acc"]:.2f}% +/- {data["std_acc"]:.2f}%')

    # Overall
    print('\nOVERALL (average across tasks):')
    overall = {}
    for opt_name in OPTIMIZERS.keys():
        accs = [all_results[task][opt_name]['mean_acc'] for task in TASKS if opt_name in all_results[task]]
        stds = [all_results[task][opt_name]['std_acc'] for task in TASKS if opt_name in all_results[task]]
        if accs:
            overall[opt_name] = (np.mean(accs), np.mean(stds))

    ranked = sorted(overall.items(), key=lambda x: -x[1][0])
    for rank, (name, (mean_acc, mean_std)) in enumerate(ranked, 1):
        print(f'  {rank}. {name:12s}: {mean_acc:.2f}% +/- {mean_std:.2f}%')

    # Head-to-head per seed
    print('\n' + '-'*60)
    print('HEAD-TO-HEAD WINS (across all tasks and seeds):')
    print('-'*60)

    for opt1 in list(OPTIMIZERS.keys()):
        for opt2 in list(OPTIMIZERS.keys()):
            if opt1 >= opt2:
                continue
            wins1 = 0
            wins2 = 0
            ties = 0
            for task in TASKS:
                e1 = all_results[task].get(opt1, {}).get('errors', [])
                e2 = all_results[task].get(opt2, {}).get('errors', [])
                for err1, err2 in zip(e1, e2):
                    if np.isnan(err1) or np.isnan(err2):
                        continue
                    if err1 < err2 - 1e-6:
                        wins1 += 1
                    elif err2 < err1 - 1e-6:
                        wins2 += 1
                    else:
                        ties += 1
            print(f'  {opt1} vs {opt2}: {wins1}-{wins2}-{ties}')

    print('\n' + '='*80)
    print('DONE!')
