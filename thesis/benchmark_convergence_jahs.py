#!/usr/bin/env python3
"""
Benchmark: ALBA_NEW (Thompson) vs ALBA_OLD (TPE) vs Optuna
on JAHS-Bench-201 with convergence checkpoints.
"""

import sys
import warnings
import time
import numpy as np
warnings.filterwarnings('ignore')

# Import benchmark wrapper
sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_jahs import JAHSBenchWrapper, run_optuna_tpe

# Import ALBA versions
from ALBA_V1_old import ALBA as ALBA_OLD
from ALBA_V1 import ALBA as ALBA_NEW

# JAHS categorical dims
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
    """Run new ALBA with Thompson Sampling + Elite Crossover."""
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


def run_optuna(wrapper, n_evals, seed):
    """Run Optuna TPE."""
    history, final_error = run_optuna_tpe(wrapper, n_evals, seed)
    return history, final_error


def main():
    # Config
    TASKS = ['cifar10', 'fashion_mnist', 'colorectal_histology']
    BUDGET = 2000
    SEEDS = list(range(42, 52))  # 10 seeds
    CHECKPOINTS = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    
    OPTIMIZERS = {
        'NEW (Thompson)': run_alba_new,
        'OLD (TPE)': run_alba_old,
        'Optuna': run_optuna,
    }
    
    print('=' * 80)
    print('CONVERGENCE BENCHMARK: NEW (Thompson) vs OLD (TPE) vs Optuna')
    print('=' * 80)
    print(f'Budget: {BUDGET}')
    print(f'Seeds: {SEEDS}')
    print(f'Tasks: {TASKS}')
    print(f'Checkpoints: {CHECKPOINTS}')
    print('=' * 80)
    
    all_results = {}
    
    for task in TASKS:
        print(f'\n{"="*80}')
        print(f'>>> TASK: {task} <<<')
        print('='*80)
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
                    histories.append([])
            
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
                print(f'  >> {opt_name}: {mean_acc:.2f}% +/- {std_acc:.2f}%')
    
    # =====================================================================
    # CONVERGENCE ANALYSIS
    # =====================================================================
    print('\n' + '=' * 80)
    print('CONVERGENCE ANALYSIS (mean accuracy % at each checkpoint)')
    print('=' * 80)
    
    for task in TASKS:
        print(f'\n{task}:')
        print(f'  {"Budget":<8}', end='')
        for opt_name in OPTIMIZERS.keys():
            print(f'{opt_name:>18}', end='')
        print('  | Best')
        print('  ' + '-' * 70)
        
        for cp in CHECKPOINTS:
            if cp > BUDGET:
                continue
            print(f'  {cp:<8}', end='')
            
            accs = {}
            for opt_name in OPTIMIZERS.keys():
                histories = all_results[task].get(opt_name, {}).get('histories', [])
                if histories:
                    accs_at_cp = [(1 - h[cp-1]) * 100 for h in histories if len(h) >= cp]
                    if accs_at_cp:
                        mean_acc = np.mean(accs_at_cp)
                        accs[opt_name] = mean_acc
                        print(f'{mean_acc:>18.2f}', end='')
                    else:
                        print(f'{"N/A":>18}', end='')
                else:
                    print(f'{"N/A":>18}', end='')
            
            # Best at this checkpoint
            if accs:
                best_name = max(accs, key=accs.get)
                print(f'  | {best_name}')
            else:
                print()
    
    # =====================================================================
    # HEAD-TO-HEAD AT EACH CHECKPOINT
    # =====================================================================
    print('\n' + '=' * 80)
    print('HEAD-TO-HEAD WINS AT EACH CHECKPOINT')
    print('=' * 80)
    
    opt_names = list(OPTIMIZERS.keys())
    
    for cp in CHECKPOINTS:
        if cp > BUDGET:
            continue
        print(f'\nAt budget {cp}:')
        
        for i, opt1 in enumerate(opt_names):
            for opt2 in opt_names[i+1:]:
                wins1, wins2, ties = 0, 0, 0
                
                for task in TASKS:
                    h1_list = all_results[task].get(opt1, {}).get('histories', [])
                    h2_list = all_results[task].get(opt2, {}).get('histories', [])
                    
                    for h1, h2 in zip(h1_list, h2_list):
                        if len(h1) >= cp and len(h2) >= cp:
                            e1, e2 = h1[cp-1], h2[cp-1]
                            if e1 < e2 - 1e-6:
                                wins1 += 1
                            elif e2 < e1 - 1e-6:
                                wins2 += 1
                            else:
                                ties += 1
                
                emoji = '✅' if wins1 > wins2 else ('❌' if wins2 > wins1 else '➖')
                print(f'  {opt1} vs {opt2}: {wins1}-{wins2}-{ties} {emoji}')
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print('\n' + '=' * 80)
    print('FINAL SUMMARY (at budget 2000)')
    print('=' * 80)
    
    # Per-task results
    for task in TASKS:
        print(f'\n{task}:')
        ranked = sorted(all_results[task].items(), key=lambda x: -x[1]['mean_acc'])
        for rank, (name, data) in enumerate(ranked, 1):
            print(f'  {rank}. {name:18s}: {data["mean_acc"]:.2f}% +/- {data["std_acc"]:.2f}%')
    
    # Overall
    print('\nOVERALL (average across tasks):')
    overall = {}
    for opt_name in OPTIMIZERS.keys():
        accs = [all_results[task][opt_name]['mean_acc'] for task in TASKS if opt_name in all_results[task]]
        if accs:
            overall[opt_name] = np.mean(accs)
    
    ranked = sorted(overall.items(), key=lambda x: -x[1])
    for rank, (name, mean_acc) in enumerate(ranked, 1):
        print(f'  {rank}. {name:18s}: {mean_acc:.2f}%')
    
    # Final head-to-head
    print('\n' + '-' * 60)
    print('FINAL HEAD-TO-HEAD (all tasks, all seeds, budget 2000):')
    print('-' * 60)
    
    for i, opt1 in enumerate(opt_names):
        for opt2 in opt_names[i+1:]:
            wins1, wins2, ties = 0, 0, 0
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
            emoji = '✅' if wins1 > wins2 else ('❌' if wins2 > wins1 else '➖')
            print(f'  {opt1} vs {opt2}: {wins1}-{wins2}-{ties} {emoji}')
    
    print('\n' + '=' * 80)
    print('DONE!')
    print('=' * 80)


if __name__ == '__main__':
    main()
