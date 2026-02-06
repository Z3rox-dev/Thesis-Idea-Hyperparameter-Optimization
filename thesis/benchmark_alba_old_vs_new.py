#!/usr/bin/env python3
"""
Benchmark: ALBA_V1 (old TPE-like categoricals) vs ALBA_V1 (Quantum Categorical Collapse)
on JAHS-Bench-201 surrogate.
"""

import sys
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')

# Import benchmark wrapper
sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_jahs import JAHSBenchWrapper

# Import OLD ALBA (TPE-like categorical handling)
from ALBA_V1_old import ALBA as ALBA_OLD

# Import NEW ALBA (cubic regularization)
from ALBA_V1 import ALBA as ALBA_NEW


# JAHS categorical dims mapping:
# dim 0: LearningRate (continuous)
# dim 1: WeightDecay (continuous)  
# dim 2: N (3 choices: 1,3,5)
# dim 3: W (3 choices: 4,8,16)
# dim 4: Resolution (3 choices: 0.25,0.5,1.0)
# dim 5: Activation (3 choices)
# dim 6: TrivialAugment (2 choices)
# dim 7-12: Op1-Op6 (5 choices each)

CATEGORICAL_DIMS = [
    (2, 3),   # N
    (3, 3),   # W
    (4, 3),   # Resolution
    (5, 3),   # Activation
    (6, 2),   # TrivialAugment
    (7, 5),   # Op1
    (8, 5),   # Op2
    (9, 5),   # Op3
    (10, 5),  # Op4
    (11, 5),  # Op5
    (12, 5),  # Op6
]


def run_alba_old(wrapper, n_evals, seed):
    """Run old ALBA with TPE-like categorical handling."""
    dim = wrapper.dim
    opt = ALBA_OLD(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=CATEGORICAL_DIMS,
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


def run_alba_new(wrapper, n_evals, seed):
    """Run new ALBA with Quantum Categorical Collapse."""
    dim = wrapper.dim
    opt = ALBA_NEW(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=n_evals,
        global_random_prob=0.05,
        stagnation_threshold=50,
        categorical_dims=CATEGORICAL_DIMS,
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


def main():
    # Config
    TASKS = ['cifar10', 'fashion_mnist']
    BUDGET = 500
    SEEDS = [42, 123, 456, 789, 1000, 1234, 5678]
    
    OPTIMIZERS = {
        'ALBA_OLD (TPE)': run_alba_old,
        'ALBA_NEW (Quantum)': run_alba_new,
    }
    
    print('=' * 70)
    print('BENCHMARK: ALBA OLD (TPE-like) vs ALBA NEW (Quantum Categorical)')
    print('=' * 70)
    print(f'Budget: {BUDGET}')
    print(f'Seeds: {SEEDS}')
    print(f'Tasks: {TASKS}')
    print('=' * 70)
    
    all_results = {}
    
    for task in TASKS:
        print(f'\n>>> TASK: {task} <<<')
        wrapper = JAHSBenchWrapper(task=task)
        
        all_results[task] = {}
        
        for opt_name, opt_func in OPTIMIZERS.items():
            print(f'\n  Running {opt_name}...')
            errors = []
            times = []
            histories = []
            
            for seed in SEEDS:
                start = time.time()
                try:
                    history, final_error = opt_func(wrapper, BUDGET, seed)
                    elapsed = time.time() - start
                    errors.append(final_error)
                    times.append(elapsed)
                    histories.append(history)
                    acc = (1 - final_error) * 100
                    print(f'    Seed {seed}: acc={acc:.2f}%, time={elapsed:.1f}s')
                except Exception as e:
                    print(f'    Seed {seed}: FAILED - {e}')
                    import traceback
                    traceback.print_exc()
                    errors.append(float('nan'))
            
            valid_errors = [e for e in errors if not np.isnan(e)]
            if valid_errors:
                mean_err = np.mean(valid_errors)
                std_err = np.std(valid_errors)
                mean_acc = (1 - mean_err) * 100
                std_acc = std_err * 100
                mean_time = np.mean(times) if times else 0
                all_results[task][opt_name] = {
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'errors': errors,
                    'mean_time': mean_time,
                    'histories': histories,
                }
                print(f'  {opt_name}: {mean_acc:.2f}% +/- {std_acc:.2f}% (avg time: {mean_time:.1f}s)')
    
    # Summary
    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)
    
    # Per-task results
    for task in TASKS:
        print(f'\n{task}:')
        ranked = sorted(all_results[task].items(), key=lambda x: -x[1]['mean_acc'])
        for rank, (name, data) in enumerate(ranked, 1):
            print(f'  {rank}. {name:20s}: {data["mean_acc"]:.2f}% +/- {data["std_acc"]:.2f}%')
    
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
        print(f'  {rank}. {name:20s}: {mean_acc:.2f}% +/- {mean_std:.2f}%')
    
    # Head-to-head per seed
    print('\n' + '-' * 60)
    print('HEAD-TO-HEAD WINS (across all tasks and seeds):')
    print('-' * 60)
    
    opt_names = list(OPTIMIZERS.keys())
    for i, opt1 in enumerate(opt_names):
        for opt2 in opt_names[i+1:]:
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
    
    # Convergence comparison at different budgets
    print('\n' + '-' * 60)
    print('CONVERGENCE AT DIFFERENT BUDGETS (mean accuracy %):')
    print('-' * 60)
    checkpoints = [50, 100, 200, 300, 400, 500]
    
    for task in TASKS:
        print(f'\n{task}:')
        print(f'  {"Budget":<10}', end='')
        for opt_name in OPTIMIZERS.keys():
            print(f'{opt_name:>22}', end='')
        print()
        
        for cp in checkpoints:
            if cp > BUDGET:
                continue
            print(f'  {cp:<10}', end='')
            for opt_name in OPTIMIZERS.keys():
                histories = all_results[task].get(opt_name, {}).get('histories', [])
                if histories:
                    accs_at_cp = [(1 - h[cp-1]) * 100 for h in histories if len(h) >= cp]
                    if accs_at_cp:
                        mean_acc = np.mean(accs_at_cp)
                        print(f'{mean_acc:>22.2f}', end='')
                    else:
                        print(f'{"N/A":>22}', end='')
                else:
                    print(f'{"N/A":>22}', end='')
            print()
    
    print('\n' + '=' * 70)
    print('DONE!')
    print('=' * 70)


if __name__ == '__main__':
    main()
