#!/usr/bin/env python
"""Benchmark NN Tabular: Minimal vs Debug vs Random vs Optuna."""
import sys
sys.path.insert(0, '/mnt/workspace/HPOBench')
import numpy as np
import warnings
warnings.filterwarnings('ignore')
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

import ConfigSpace as CS
from hpobench.benchmarks.ml import TabularBenchmark
from hpo_minimal import HPOptimizer
from hpo_debug import QuadHPO as QuadHPO_Debug
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configurazione
MODEL = 'nn'
TASK_ID = 31
BUDGET = 150
SEEDS = [42, 123, 456, 789, 999, 1234, 2024]

print('='*80)
print(f'BENCHMARK: {MODEL.upper()} Tabular | Task ID: {TASK_ID}')
print(f'Budget: {BUDGET} | Seeds: {SEEDS}')
print('='*80)

bench = TabularBenchmark(model=MODEL, task_id=TASK_ID)
cs = bench.get_configuration_space()
hps = cs.get_hyperparameters()
max_fid = bench.get_max_fidelity()

print(f'Dimensioni: {len(hps)} iperparametri')
for hp in hps:
    print(f'  - {hp.name}')
print(flush=True)

# Build bounds e types
bounds = []
types = []
for hp in hps:
    if hasattr(hp, 'sequence') and hp.sequence:
        seq = list(hp.sequence)
        bounds.append((0.0, float(len(seq) - 1)))
        types.append('index')
    elif isinstance(hp, CS.UniformFloatHyperparameter):
        bounds.append((float(hp.lower), float(hp.upper)))
        types.append('float')
    elif isinstance(hp, CS.UniformIntegerHyperparameter):
        bounds.append((float(hp.lower), float(hp.upper)))
        types.append('int')
    else:
        raise ValueError(f'Unsupported HP: {hp}')

dim = len(bounds)

def xnorm_to_config(x_norm):
    values = {}
    for val, hp, (lo, hi), t in zip(x_norm, hps, bounds, types):
        if t == 'index':
            seq = list(hp.sequence)
            idx = int(np.clip(np.floor(val * len(seq)), 0, len(seq) - 1))
            values[hp.name] = seq[idx]
        else:
            v = lo + float(val) * (hi - lo)
            if t == 'int':
                v = int(round(v))
                v = max(int(hp.lower), min(int(hp.upper), int(v)))
            values[hp.name] = v
    return CS.Configuration(cs, values=values)

def objective(x_norm):
    cfg = xnorm_to_config(x_norm)
    res = bench.objective_function(cfg, fidelity=max_fid, metric='acc')
    return float(res['function_value'])

results = {'minimal': [], 'debug': [], 'random': [], 'optuna': []}

for seed in SEEDS:
    print(f'Seed {seed:4d}: ', end='', flush=True)
    
    # Minimal
    opt = HPOptimizer(bounds=[(0.0, 1.0)]*dim, maximize=False, seed=seed)
    _, val_m = opt.optimize(objective, budget=BUDGET)
    results['minimal'].append(val_m)
    print(f'M={val_m:.5f} ', end='', flush=True)
    
    # Debug
    opt2 = QuadHPO_Debug(bounds=[(0.0, 1.0)]*dim, maximize=False, rng_seed=seed, debug_log=False)
    opt2.optimize(objective, budget=BUDGET)
    val_d = -opt2.best_score_global
    results['debug'].append(val_d)
    print(f'D={val_d:.5f} ', end='', flush=True)
    
    # Random
    rng = np.random.default_rng(seed)
    best_r = float('inf')
    for _ in range(BUDGET):
        x = rng.random(dim)
        v = objective(x)
        if v < best_r:
            best_r = v
    results['random'].append(best_r)
    print(f'R={best_r:.5f} ', end='', flush=True)
    
    # Optuna
    def optuna_obj(trial):
        x = np.array([trial.suggest_float(f'x{i}', 0.0, 1.0) for i in range(dim)])
        return objective(x)
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(optuna_obj, n_trials=BUDGET, show_progress_bar=False)
    results['optuna'].append(study.best_value)
    print(f'O={study.best_value:.5f}', flush=True)

print()
print('='*80)
print('SUMMARY:')
print('='*80)
for method in ['minimal', 'debug', 'random', 'optuna']:
    arr = np.array(results[method])
    print(f'{method.upper():8s}: {arr.mean():.6f} +/- {arr.std():.6f}  (best={arr.min():.6f})')

means = {k: np.mean(v) for k, v in results.items()}
winner = min(means, key=means.get)
print(f'\n>>> WINNER: {winner.upper()} with mean loss = {means[winner]:.6f}')
