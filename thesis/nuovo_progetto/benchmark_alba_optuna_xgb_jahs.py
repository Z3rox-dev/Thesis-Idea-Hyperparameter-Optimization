#!/usr/bin/env python3
"""
Benchmark ALBA vs Optuna vs TuRBO su:
1. XGBoost (YAHPO Gym)
2. JAHS-Bench-201
"""

from __future__ import annotations
import sys
import warnings
import argparse
import numpy as np

warnings.filterwarnings('ignore')


# ==== TuRBO utilities ====

def run_turbo_continuous(objective_fn, dim: int, budget: int, seed: int):
    """
    Run TuRBO-M on a continuous [0,1]^d objective.
    Returns best_y found.
    """
    try:
        from turbo import TurboM
        import torch
    except ImportError:
        print("TuRBO not available, skipping")
        return None
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    best_y = [float('inf')]
    
    class Objective:
        def __init__(self):
            self.dim = dim
            self.lb = np.zeros(dim)
            self.ub = np.ones(dim)
        
        def __call__(self, x):
            y = objective_fn(x)
            best_y[0] = min(best_y[0], y)
            return y
    
    f = Objective()
    
    # TuRBO-M parameters
    n_trust_regions = min(5, max(2, budget // 50))
    n_init = min(2 * dim, budget // (n_trust_regions * 3))
    n_init = max(n_init, dim + 1)
    
    turbo_m = TurboM(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=budget,
        n_trust_regions=n_trust_regions,
        batch_size=1,
        verbose=False,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        device="cpu",
        dtype="float64",
    )
    
    turbo_m.optimize()
    
    return best_y[0]

# ==== YAHPO XGBoost Benchmark ====

def run_yahpo_xgboost(budget: int = 200, seeds: int = 3):
    """Run YAHPO XGBoost benchmark: ALBA vs Optuna vs TuRBO."""
    try:
        from yahpo_gym import benchmark_set, local_config
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("YAHPO or Optuna not available")
        return None
    
    # Import ALBA
    sys.path.insert(0, '/mnt/workspace/thesis')
    from alba_framework_potential.optimizer import ALBA as ALBAPotentialOptimizer
    
    local_config.init_config()
    local_config.set_data_path("/mnt/workspace/data")
    
    scenarios = ['rbv2_xgboost', 'iaml_xgboost']
    
    results = {'ALBA': [], 'Optuna': [], 'TuRBO': []}
    
    for scenario_name in scenarios:
        bench = benchmark_set.BenchmarkSet(scenario_name)
        instances = bench.instances[:3]  # First 3 instances
        fidelity_params = bench.config.fidelity_params if hasattr(bench.config, 'fidelity_params') else []
        hp_names = [p for p in bench.config_space.get_hyperparameter_names() 
                    if p not in fidelity_params]
        
        for instance in instances:
            bench.set_instance(instance)
            
            # Get bounds for optimization
            bounds = []
            hp_types = []
            for hp_name in hp_names:
                hp = bench.config_space.get_hyperparameter(hp_name)
                if hasattr(hp, 'lower'):
                    bounds.append((float(hp.lower), float(hp.upper)))
                    # Check if it's integer type
                    hp_type_str = str(type(hp).__name__)
                    if 'Integer' in hp_type_str:
                        hp_types.append('int')
                    else:
                        hp_types.append('cont')
                elif hasattr(hp, 'choices'):
                    bounds.append((0.0, float(len(hp.choices) - 1)))
                    hp_types.append('cat')
            
            dim = len(bounds)
            
            def make_objective(hp_names, hp_types, bounds, bench, fidelity_params):
                def objective(x):
                    cfg = {}
                    for j, (hp_name, hp_type) in enumerate(zip(hp_names, hp_types)):
                        hp = bench.config_space.get_hyperparameter(hp_name)
                        if hp_type == 'cont':
                            cfg[hp_name] = float(np.clip(x[j], bounds[j][0], bounds[j][1]))
                        elif hp_type == 'int':
                            cfg[hp_name] = int(np.clip(round(x[j]), int(bounds[j][0]), int(bounds[j][1])))
                        else:  # cat
                            idx = int(np.clip(round(x[j]), 0, len(hp.choices) - 1))
                            cfg[hp_name] = hp.choices[idx]
                    
                    # Add fidelity
                    for fp in fidelity_params:
                        fp_hp = bench.config_space.get_hyperparameter(fp)
                        if hasattr(fp_hp, 'upper'):
                            cfg[fp] = fp_hp.upper
                        elif hasattr(fp_hp, 'choices'):
                            cfg[fp] = fp_hp.choices[-1]
                    
                    try:
                        result = bench.objective_function(cfg)[0]
                        target_key = next((k for k in result if 'acc' in k.lower() or 'auc' in k.lower()), list(result.keys())[0])
                        val = result[target_key]
                        return -float(val) if 'acc' in target_key.lower() or 'auc' in target_key.lower() else float(val)
                    except Exception:
                        return 1.0  # Penalty for invalid configs
                return objective
            
            objective = make_objective(hp_names, hp_types, bounds, bench, fidelity_params)
            
            # Create normalized objective for TuRBO (maps [0,1]^d -> original space)
            def make_turbo_objective(objective, bounds):
                def turbo_obj(x_01):
                    x = np.array([lo + x_01[j] * (hi - lo) for j, (lo, hi) in enumerate(bounds)])
                    return objective(x)
                return turbo_obj
            
            turbo_objective = make_turbo_objective(objective, bounds)
            
            for seed in range(seeds):
                print(f"  [{scenario_name}][{instance}] seed={seed}", end=" ", flush=True)
                
                # --- ALBA ---
                alba_opt = ALBAPotentialOptimizer(
                    bounds=bounds,
                    seed=seed,
                    maximize=False,
                )
                for _ in range(budget):
                    x = alba_opt.ask()
                    y = objective(x)
                    alba_opt.tell(x, y)
                alba_best = alba_opt.best_y
                
                # --- Optuna ---
                def optuna_objective(trial):
                    x = []
                    for j, (hp_name, hp_type) in enumerate(zip(hp_names, hp_types)):
                        lo, hi = bounds[j]
                        if hp_type == 'cont':
                            x.append(trial.suggest_float(hp_name, lo, hi))
                        elif hp_type == 'int':
                            x.append(trial.suggest_int(hp_name, int(lo), int(hi)))
                        else:  # cat
                            x.append(trial.suggest_float(hp_name, lo, hi))
                    return objective(np.array(x))
                
                study = optuna.create_study(direction='minimize', 
                                           sampler=optuna.samplers.TPESampler(seed=seed))
                study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
                optuna_best = study.best_value
                
                # --- TuRBO ---
                turbo_best = run_turbo_continuous(turbo_objective, dim, budget, seed)
                if turbo_best is None:
                    turbo_best = float('inf')
                
                results['ALBA'].append(alba_best)
                results['Optuna'].append(optuna_best)
                results['TuRBO'].append(turbo_best)
                
                all_bests = [('ALBA', alba_best), ('Optuna', optuna_best), ('TuRBO', turbo_best)]
                winner = min(all_bests, key=lambda x: x[1])[0]
                print(f"ALBA={alba_best:.4f} Optuna={optuna_best:.4f} TuRBO={turbo_best:.4f} -> {winner}")
    
    # Summary
    n = len(results['ALBA'])
    alba_wins = sum(1 for i in range(n) if results['ALBA'][i] < results['Optuna'][i] and results['ALBA'][i] < results['TuRBO'][i])
    optuna_wins = sum(1 for i in range(n) if results['Optuna'][i] < results['ALBA'][i] and results['Optuna'][i] < results['TuRBO'][i])
    turbo_wins = sum(1 for i in range(n) if results['TuRBO'][i] < results['ALBA'][i] and results['TuRBO'][i] < results['Optuna'][i])
    
    print(f"\n=== YAHPO XGBoost Summary ===")
    print(f"ALBA wins: {alba_wins}/{n}")
    print(f"Optuna wins: {optuna_wins}/{n}")
    print(f"TuRBO wins: {turbo_wins}/{n}")
    
    return results


# ==== JAHS-Bench-201 Benchmark ====

def run_jahs_benchmark(task: str = 'cifar10', budget: int = 200, seeds: int = 3):
    """Run JAHS-Bench-201 benchmark: ALBA vs Optuna vs TuRBO."""
    try:
        from jahs_bench import Benchmark
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        print(f"JAHS or Optuna not available: {e}")
        return None
    
    # Import ALBA
    sys.path.insert(0, '/mnt/workspace/thesis')
    from alba_framework_potential.optimizer import ALBA as ALBAPotentialOptimizer
    
    # Build param_space for ALBA (proper typed space)
    param_space = {
        'LearningRate': (0.001, 1.0, 'log'),
        'WeightDecay': (1e-5, 0.01, 'log'),
        'N': [1, 3, 5],  # categorical
        'W': [4, 8, 16],  # categorical
        'Resolution': [0.25, 0.5, 1.0],  # categorical
        'Activation': ['ReLU', 'Hardswish', 'Mish'],  # categorical
        'TrivialAugment': [True, False],  # categorical
        'Op1': [0, 1, 2, 3, 4],  # categorical
        'Op2': [0, 1, 2, 3, 4],
        'Op3': [0, 1, 2, 3, 4],
        'Op4': [0, 1, 2, 3, 4],
        'Op5': [0, 1, 2, 3, 4],
        'Op6': [0, 1, 2, 3, 4],
    }
    
    # JAHS space definition for TuRBO (normalized [0,1]^13)
    # Dimensions: LR, WD, N, W, Res, Act, TA, Op1-6
    # We'll decode from [0,1] to actual values
    JAHS_DIM = 13
    
    print(f"\nInitializing JAHS-Bench-201 (task={task})...")
    bench = Benchmark(
        task=task,
        kind='surrogate',
        download=True,
        save_dir='/mnt/workspace/jahs_bench_data',
        metrics=['valid-acc']
    )
    
    def objective_from_config(config):
        """Evaluate config dict, return error (1 - acc/100)."""
        config = dict(config)  # Copy
        config['Optimizer'] = 'SGD'
        config['epoch'] = 200
        result = bench(config)
        last_epoch = max(result.keys())
        valid_acc = result[last_epoch]['valid-acc']
        return 1.0 - valid_acc / 100.0
    
    def decode_x01_to_config(x):
        """Decode [0,1]^13 to JAHS config dict."""
        config = {}
        # LR: log-scale [0.001, 1.0]
        config['LearningRate'] = 0.001 * (1.0 / 0.001) ** x[0]
        # WD: log-scale [1e-5, 0.01]
        config['WeightDecay'] = 1e-5 * (0.01 / 1e-5) ** x[1]
        # N: [1, 3, 5]
        config['N'] = [1, 3, 5][int(np.clip(round(x[2] * 2), 0, 2))]
        # W: [4, 8, 16]
        config['W'] = [4, 8, 16][int(np.clip(round(x[3] * 2), 0, 2))]
        # Resolution: [0.25, 0.5, 1.0]
        config['Resolution'] = [0.25, 0.5, 1.0][int(np.clip(round(x[4] * 2), 0, 2))]
        # Activation: ['ReLU', 'Hardswish', 'Mish']
        config['Activation'] = ['ReLU', 'Hardswish', 'Mish'][int(np.clip(round(x[5] * 2), 0, 2))]
        # TrivialAugment: [True, False]
        config['TrivialAugment'] = x[6] < 0.5
        # Op1-6: [0,1,2,3,4]
        for i, op_name in enumerate(['Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']):
            config[op_name] = int(np.clip(round(x[7 + i] * 4), 0, 4))
        return config
    
    def turbo_objective(x):
        """Objective for TuRBO: [0,1]^13 -> error."""
        config = decode_x01_to_config(x)
        return objective_from_config(config)
    
    # HP spec for Optuna
    HP_SPEC = {
        'LearningRate': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True},
        'WeightDecay': {'type': 'float', 'low': 1e-5, 'high': 0.01, 'log': True},
        'N': {'type': 'cat', 'choices': [1, 3, 5]},
        'W': {'type': 'cat', 'choices': [4, 8, 16]},
        'Resolution': {'type': 'cat', 'choices': [0.25, 0.5, 1.0]},
        'Activation': {'type': 'cat', 'choices': ['ReLU', 'Hardswish', 'Mish']},
        'TrivialAugment': {'type': 'cat', 'choices': [True, False]},
        'Op1': {'type': 'cat', 'choices': [0, 1, 2, 3, 4]},
        'Op2': {'type': 'cat', 'choices': [0, 1, 2, 3, 4]},
        'Op3': {'type': 'cat', 'choices': [0, 1, 2, 3, 4]},
        'Op4': {'type': 'cat', 'choices': [0, 1, 2, 3, 4]},
        'Op5': {'type': 'cat', 'choices': [0, 1, 2, 3, 4]},
        'Op6': {'type': 'cat', 'choices': [0, 1, 2, 3, 4]},
    }
    
    results = {'ALBA': [], 'Optuna': [], 'TuRBO': []}
    
    for seed in range(seeds):
        print(f"  [JAHS {task}] seed={seed}", end=" ", flush=True)
        
        # --- ALBA with param_space (proper categoricals) ---
        alba_opt = ALBAPotentialOptimizer(
            param_space=param_space,
            seed=seed,
            maximize=False,
            total_budget=budget,
        )
        alba_best = float('inf')
        for _ in range(budget):
            config = alba_opt.ask()  # Returns dict
            y = objective_from_config(config)
            alba_opt.tell(config, y)
            alba_best = min(alba_best, y)
        
        # --- Optuna ---
        def optuna_objective(trial):
            config = {}
            for hp_name, hp_spec in HP_SPEC.items():
                if hp_spec['type'] == 'float':
                    config[hp_name] = trial.suggest_float(
                        hp_name, hp_spec['low'], hp_spec['high'],
                        log=hp_spec.get('log', False)
                    )
                else:
                    config[hp_name] = trial.suggest_categorical(hp_name, hp_spec['choices'])
            return objective_from_config(config)
        
        study = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
        optuna_best = study.best_value
        
        # --- TuRBO ---
        turbo_best = run_turbo_continuous(turbo_objective, JAHS_DIM, budget, seed)
        if turbo_best is None:
            turbo_best = float('inf')
        
        results['ALBA'].append(alba_best)
        results['Optuna'].append(optuna_best)
        results['TuRBO'].append(turbo_best)
        
        all_bests = [('ALBA', alba_best), ('Optuna', optuna_best), ('TuRBO', turbo_best)]
        winner = min(all_bests, key=lambda x: x[1])[0]
        print(f"ALBA={alba_best:.4f} Optuna={optuna_best:.4f} TuRBO={turbo_best:.4f} -> {winner}")
    
    # Summary
    n = len(results['ALBA'])
    alba_wins = sum(1 for i in range(n) if results['ALBA'][i] < results['Optuna'][i] and results['ALBA'][i] < results['TuRBO'][i])
    optuna_wins = sum(1 for i in range(n) if results['Optuna'][i] < results['ALBA'][i] and results['Optuna'][i] < results['TuRBO'][i])
    turbo_wins = sum(1 for i in range(n) if results['TuRBO'][i] < results['ALBA'][i] and results['TuRBO'][i] < results['Optuna'][i])
    
    print(f"\n=== JAHS {task} Summary ===")
    print(f"ALBA wins: {alba_wins}/{n}")
    print(f"Optuna wins: {optuna_wins}/{n}")
    print(f"TuRBO wins: {turbo_wins}/{n}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--only', type=str, default=None, help='xgb or jahs')
    parser.add_argument('--jahs-task', type=str, default='cifar10',
                       choices=['cifar10', 'colorectal_histology', 'fashion_mnist'])
    args = parser.parse_args()
    
    print("=" * 70)
    print("BENCHMARK: ALBA vs Optuna vs TuRBO")
    print("=" * 70)
    print(f"Budget: {args.budget} evaluations")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)
    
    all_alba_wins = 0
    all_optuna_wins = 0
    all_turbo_wins = 0
    all_total = 0
    
    # XGBoost (YAHPO)
    if args.only is None or args.only == 'xgb':
        print("\n" + "=" * 70)
        print("PART 1: YAHPO XGBoost")
        print("=" * 70)
        xgb_results = run_yahpo_xgboost(budget=args.budget, seeds=args.seeds)
        if xgb_results:
            n = len(xgb_results['ALBA'])
            alba_w = sum(1 for i in range(n) if xgb_results['ALBA'][i] < xgb_results['Optuna'][i] and xgb_results['ALBA'][i] < xgb_results['TuRBO'][i])
            optuna_w = sum(1 for i in range(n) if xgb_results['Optuna'][i] < xgb_results['ALBA'][i] and xgb_results['Optuna'][i] < xgb_results['TuRBO'][i])
            turbo_w = sum(1 for i in range(n) if xgb_results['TuRBO'][i] < xgb_results['ALBA'][i] and xgb_results['TuRBO'][i] < xgb_results['Optuna'][i])
            all_alba_wins += alba_w
            all_optuna_wins += optuna_w
            all_turbo_wins += turbo_w
            all_total += n
    
    # JAHS-Bench-201
    if args.only is None or args.only == 'jahs':
        print("\n" + "=" * 70)
        print(f"PART 2: JAHS-Bench-201 ({args.jahs_task})")
        print("=" * 70)
        jahs_results = run_jahs_benchmark(task=args.jahs_task, budget=args.budget, seeds=args.seeds)
        if jahs_results:
            n = len(jahs_results['ALBA'])
            alba_w = sum(1 for i in range(n) if jahs_results['ALBA'][i] < jahs_results['Optuna'][i] and jahs_results['ALBA'][i] < jahs_results['TuRBO'][i])
            optuna_w = sum(1 for i in range(n) if jahs_results['Optuna'][i] < jahs_results['ALBA'][i] and jahs_results['Optuna'][i] < jahs_results['TuRBO'][i])
            turbo_w = sum(1 for i in range(n) if jahs_results['TuRBO'][i] < jahs_results['ALBA'][i] and jahs_results['TuRBO'][i] < jahs_results['Optuna'][i])
            all_alba_wins += alba_w
            all_optuna_wins += optuna_w
            all_turbo_wins += turbo_w
            all_total += n
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total runs: {all_total}")
    if all_total > 0:
        print(f"ALBA wins: {all_alba_wins} ({100*all_alba_wins/all_total:.1f}%)")
        print(f"Optuna wins: {all_optuna_wins} ({100*all_optuna_wins/all_total:.1f}%)")
        print(f"TuRBO wins: {all_turbo_wins} ({100*all_turbo_wins/all_total:.1f}%)")
        ties = all_total - all_alba_wins - all_optuna_wins - all_turbo_wins
        print(f"Ties: {ties}")
    print("=" * 70)


if __name__ == '__main__':
    main()
