"""
Benchmark ALBA vs Optuna on YAHPO Gym benchmarks
Focus on scenarios with categorical hyperparameters
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from collections import defaultdict

import yahpo_gym
from yahpo_gym import BenchmarkSet, local_config
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ALBA_V1 import ALBA as ALBACore

# Setup YAHPO
local_config.set_data_path('/root/.yahpo_gym_data/')


class ALBAWrapper:
    """Wrapper for ALBA that handles YAHPO configuration spaces with conditional params."""
    
    def __init__(self, configspace, skip_params=None, seed=42, n_trials=100):
        from ConfigSpace.hyperparameters import (
            CategoricalHyperparameter, OrdinalHyperparameter,
            UniformFloatHyperparameter, UniformIntegerHyperparameter
        )
        
        if skip_params is None:
            skip_params = ['task_id', 'OpenML_task_id', 'trainsize', 'repl']
        
        self.configspace = configspace
        self.skip_params = skip_params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Identify all HPs except skip
        self.hp_list = []
        categorical_dims = []
        dim_idx = 0
        
        for hp in configspace.values():
            if hp.name in skip_params:
                continue
            
            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                self.hp_list.append({
                    'name': hp.name, 
                    'type': 'categorical', 
                    'choices': choices
                })
                categorical_dims.append((dim_idx, len(choices)))
            elif isinstance(hp, UniformFloatHyperparameter):
                self.hp_list.append({
                    'name': hp.name, 
                    'type': 'float', 
                    'low': hp.lower, 
                    'high': hp.upper,
                    'log': hp.log
                })
            elif isinstance(hp, UniformIntegerHyperparameter):
                self.hp_list.append({
                    'name': hp.name, 
                    'type': 'int', 
                    'low': hp.lower, 
                    'high': hp.upper,
                    'log': hp.log if hasattr(hp, 'log') else False
                })
            dim_idx += 1
        
        self.dim = len(self.hp_list)
        
        # Create ALBA optimizer in [0,1]^d
        self.opt = ALBACore(
            bounds=[(0.0, 1.0)] * self.dim,
            maximize=False,
            seed=seed,
            total_budget=n_trials,
            categorical_dims=categorical_dims,
        )
    
    def _array_to_config(self, x):
        """Convert [0,1]^d array to config dict, handling conditional params."""
        config = {}
        for i, hp in enumerate(self.hp_list):
            val = np.clip(x[i], 0, 1)
            
            if hp['type'] == 'categorical':
                choices = hp['choices']
                idx = int(round(val * (len(choices) - 1)))
                idx = np.clip(idx, 0, len(choices) - 1)
                config[hp['name']] = choices[idx]
            elif hp['type'] == 'float':
                if hp.get('log', False):
                    low_log = np.log(hp['low'])
                    high_log = np.log(hp['high'])
                    config[hp['name']] = np.exp(low_log + val * (high_log - low_log))
                else:
                    config[hp['name']] = hp['low'] + val * (hp['high'] - hp['low'])
            elif hp['type'] == 'int':
                if hp.get('log', False):
                    low_log = np.log(hp['low'])
                    high_log = np.log(hp['high'])
                    config[hp['name']] = int(round(np.exp(low_log + val * (high_log - low_log))))
                else:
                    config[hp['name']] = int(round(hp['low'] + val * (hp['high'] - hp['low'])))
        
        # Remove inactive conditional params
        conditions = self.configspace.get_conditions()
        params_to_remove = []
        
        for cond in conditions:
            if hasattr(cond, 'child') and hasattr(cond, 'parent'):
                child_name = cond.child.name
                parent_name = cond.parent.name
                
                if parent_name in config and child_name in config:
                    # Check if child should be active
                    is_active = False
                    if hasattr(cond, 'value'):
                        is_active = (config[parent_name] == cond.value)
                    elif hasattr(cond, 'values'):
                        is_active = (config[parent_name] in cond.values)
                    
                    if not is_active:
                        params_to_remove.append(child_name)
        
        # Remove inactive params
        for name in params_to_remove:
            if name in config:
                del config[name]
        
        return config
    
    def suggest(self):
        """Get next configuration to evaluate."""
        x = self.opt.ask()
        return self._array_to_config(x), x
    
    def observe(self, x, y):
        """Report objective value."""
        self.opt.tell(x, y)


def run_alba(bench, instance, n_trials, seed):
    """Run ALBA on YAHPO benchmark"""
    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]  # Usually 'acc', 'auc', etc.
    
    wrapper = ALBAWrapper(cs, seed=seed, n_trials=n_trials)
    best_y = float('-inf')
    
    for _ in range(n_trials):
        config, x = wrapper.suggest()
        
        # Add instance info
        if 'OpenML_task_id' in [hp.name for hp in cs.values()]:
            config['OpenML_task_id'] = instance
        elif 'task_id' in [hp.name for hp in cs.values()]:
            config['task_id'] = instance
        
        # Add fidelity params at max
        if hasattr(bench.config, 'fidelity_params'):
            for fp in bench.config.fidelity_params:
                if fp == 'epoch':
                    config['epoch'] = 52
                elif fp == 'trainsize':
                    config['trainsize'] = 1.0
                elif fp == 'repl':
                    config['repl'] = 10
        
        try:
            result = bench.objective_function(config)
            y = result[0][obj_name]  # result is a list of dicts
        except Exception as e:
            y = 0.0  # Penalty for invalid configs
        
        # ALBA minimizes, so negate
        wrapper.observe(x, -y)
        best_y = max(best_y, y)
    
    return best_y

def run_optuna(bench, instance, n_trials, seed):
    """Run Optuna TPE on YAHPO benchmark with conditional params support"""
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter, OrdinalHyperparameter,
        UniformFloatHyperparameter, UniformIntegerHyperparameter
    )
    
    cs = bench.get_opt_space()
    obj_name = bench.config.y_names[0]
    skip_params = ['task_id', 'OpenML_task_id', 'trainsize', 'repl']
    conditions = cs.get_conditions()
    
    def objective(trial):
        config = {}
        
        # First pass: sample non-conditional params
        for hp in cs.values():
            if hp.name in skip_params:
                continue
            
            # Check if this param is conditional
            is_conditional = False
            for cond in conditions:
                if hasattr(cond, 'child') and cond.child.name == hp.name:
                    is_conditional = True
                    break
            
            if is_conditional:
                continue  # Handle later
            
            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                config[hp.name] = trial.suggest_categorical(hp.name, choices)
            elif isinstance(hp, UniformFloatHyperparameter):
                config[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper, log=hp.log)
            elif isinstance(hp, UniformIntegerHyperparameter):
                log = hp.log if hasattr(hp, 'log') else False
                config[hp.name] = trial.suggest_int(hp.name, hp.lower, hp.upper, log=log)
        
        # Second pass: handle conditional params
        for cond in conditions:
            if hasattr(cond, 'child') and hasattr(cond, 'parent'):
                child_hp = cond.child
                parent_name = cond.parent.name
                
                if parent_name not in config:
                    continue
                
                # Check if condition is satisfied
                is_active = False
                if hasattr(cond, 'value'):
                    is_active = (config[parent_name] == cond.value)
                elif hasattr(cond, 'values'):
                    is_active = (config[parent_name] in cond.values)
                
                if is_active:
                    if isinstance(child_hp, UniformFloatHyperparameter):
                        config[child_hp.name] = trial.suggest_float(child_hp.name, child_hp.lower, child_hp.upper, log=child_hp.log)
                    elif isinstance(child_hp, UniformIntegerHyperparameter):
                        log = child_hp.log if hasattr(child_hp, 'log') else False
                        config[child_hp.name] = trial.suggest_int(child_hp.name, child_hp.lower, child_hp.upper, log=log)
                    elif isinstance(child_hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                        choices = list(child_hp.choices) if hasattr(child_hp, 'choices') else list(child_hp.sequence)
                        config[child_hp.name] = trial.suggest_categorical(child_hp.name, choices)
        
        # Add instance info
        if 'OpenML_task_id' in [hp.name for hp in cs.values()]:
            config['OpenML_task_id'] = instance
        elif 'task_id' in [hp.name for hp in cs.values()]:
            config['task_id'] = instance
        
        # Add fidelity params
        if hasattr(bench.config, 'fidelity_params'):
            for fp in bench.config.fidelity_params:
                if fp == 'epoch':
                    config['epoch'] = 52
                elif fp == 'trainsize':
                    config['trainsize'] = 1.0
                elif fp == 'repl':
                    config['repl'] = 10
        
        try:
            result = bench.objective_function(config)
            return result[0][obj_name]
        except:
            return 0.0
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_value


def benchmark_scenario(scenario_name, n_trials=50, n_seeds=10, n_instances=5):
    """Run benchmark on a YAHPO scenario"""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    bench = BenchmarkSet(scenario_name)
    cs = bench.get_opt_space()
    
    # Info
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
    params = [hp for hp in cs.values() if hp.name not in ['task_id', 'OpenML_task_id', 'trainsize', 'repl']]
    n_cat = sum(1 for hp in params if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)))
    print(f"Parameters: {len(params)} ({n_cat} categorical)")
    print(f"Instances available: {len(bench.instances)}")
    
    # Select instances
    instances = bench.instances[:n_instances]
    print(f"Testing on {len(instances)} instances with {n_seeds} seeds each")
    
    results = defaultdict(list)
    alba_wins = 0
    optuna_wins = 0
    ties = 0
    
    for inst in instances:
        for seed in range(n_seeds):
            try:
                alba_score = run_alba(bench, inst, n_trials, seed)
                optuna_score = run_optuna(bench, inst, n_trials, seed)
                
                results['alba'].append(alba_score)
                results['optuna'].append(optuna_score)
                
                if alba_score > optuna_score + 1e-6:
                    alba_wins += 1
                elif optuna_score > alba_score + 1e-6:
                    optuna_wins += 1
                else:
                    ties += 1
                
                print(f"  {inst}/seed{seed}: ALBA={alba_score:.4f}, Optuna={optuna_score:.4f}", end='')
                if alba_score > optuna_score + 1e-6:
                    print(" ✓")
                elif optuna_score > alba_score + 1e-6:
                    print(" ✗")
                else:
                    print(" =")
            except Exception as e:
                print(f"  {inst}/seed{seed}: ERROR - {e}")
    
    print(f"\nResults for {scenario_name}:")
    print(f"  ALBA mean: {np.mean(results['alba']):.4f} ± {np.std(results['alba']):.4f}")
    print(f"  Optuna mean: {np.mean(results['optuna']):.4f} ± {np.std(results['optuna']):.4f}")
    print(f"  ALBA vs Optuna: {alba_wins} - {optuna_wins} (ties: {ties})")
    
    return alba_wins, optuna_wins, ties


if __name__ == '__main__':
    # Scenarios with categorical params (sorted by number of categoricals)
    scenarios = [
        'rbv2_xgboost',   # 12 cont + 2 cat, 119 instances
        'rbv2_ranger',    # 5 cont + 3 cat, 119 instances
        'rbv2_svm',       # 4 cont + 2 cat, 106 instances
        'iaml_xgboost',   # 12 cont + 1 cat, 4 instances
    ]
    
    print("="*60)
    print("YAHPO Gym Benchmark: ALBA vs Optuna TPE")
    print("="*60)
    
    total_alba = 0
    total_optuna = 0
    total_ties = 0
    
    for scenario in scenarios:
        a, o, t = benchmark_scenario(scenario, n_trials=2000, n_seeds=3, n_instances=3)
        total_alba += a
        total_optuna += o
        total_ties += t
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"ALBA vs Optuna: {total_alba} - {total_optuna} (ties: {total_ties})")
    
    if total_alba > total_optuna:
        print("Winner: ALBA! 🏆")
    elif total_optuna > total_alba:
        print("Winner: Optuna")
    else:
        print("Tie!")
