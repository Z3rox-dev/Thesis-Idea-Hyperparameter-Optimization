#!/usr/bin/env python3
"""
Test Leaf Selection su YAHPO rbv2_xgboost

rbv2_xgboost ha 14 hyperparameter (12 cont + 2 cat).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import ConfigSpace as CS
from yahpo_gym import benchmark_set, local_config

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube
from alba_framework_potential.optimizer import ALBA


# ============================================================================
# LEAF SELECTORS
# ============================================================================

@dataclass
class ThompsonSamplingLeafSelector(UCBSoftmaxLeafSelector):
    """Thompson Sampling."""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        samples = []
        for c in leaves:
            alpha = c.n_good + 1
            beta = (c.n_trials - c.n_good) + 1
            sample = rng.beta(alpha, beta)
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            samples.append(sample + exploration + model_bonus)
        
        return leaves[int(np.argmax(samples))]


@dataclass
class ThompsonOptimisticLeafSelector(UCBSoftmaxLeafSelector):
    """Thompson Optimistic: max of k samples."""
    k_samples: int = 5
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta = (c.n_trials - c.n_good) + 1
            samples = rng.beta(alpha, beta, size=self.k_samples)
            sample = np.max(samples)
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(sample + exploration + model_bonus)
        
        return leaves[int(np.argmax(scores))]


@dataclass
class ProbabilityOfImprovementLeafSelector(UCBSoftmaxLeafSelector):
    """PI: P(X > best)"""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        from scipy.stats import beta as beta_dist
        best_ratio = max(c.good_ratio() for c in leaves)
        
        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            pi = 1 - beta_dist.cdf(best_ratio, alpha, beta_param)
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(pi + exploration + model_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


# ============================================================================
# YAHPO WRAPPER for rbv2_xgboost
# ============================================================================

class RBV2XGBoostWrapper:
    """Wrapper per YAHPO rbv2_xgboost che gestisce config space con condizionali."""
    
    def __init__(self, task_id: str = "3"):
        local_config.init_config()
        local_config.set_data_path("/mnt/workspace/data/")
        
        self.bench = benchmark_set.BenchmarkSet("rbv2_xgboost")
        self.bench.set_instance(task_id)
        self.task_id = task_id
        
        # Get config space and extract hyperparameters
        self.cs = self.bench.get_opt_space()
        self._build_simple_bounds()
    
    def _build_simple_bounds(self):
        """Build simple [0,1] bounds, handling conditionals by using defaults."""
        self.hp_names = []
        self.hp_info = []
        
        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, CS.Constant):
                continue
            
            name = hp.name
            self.hp_names.append(name)
            
            if isinstance(hp, CS.UniformFloatHyperparameter):
                if hp.log:
                    self.hp_info.append(("float_log", hp.lower, hp.upper))
                else:
                    self.hp_info.append(("float", hp.lower, hp.upper))
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                if hp.log:
                    self.hp_info.append(("int_log", hp.lower, hp.upper))
                else:
                    self.hp_info.append(("int", hp.lower, hp.upper))
            elif isinstance(hp, CS.CategoricalHyperparameter):
                self.hp_info.append(("cat", hp.choices))
            elif isinstance(hp, CS.OrdinalHyperparameter):
                self.hp_info.append(("ord", hp.sequence))
            else:
                # Skip unknown types
                self.hp_names.pop()
        
        self.bounds = [(0.0, 1.0)] * len(self.hp_names)
        self.dim = len(self.bounds)
    
    def x_to_config(self, x_norm: np.ndarray) -> dict:
        """Map [0,1]^d to config dict."""
        config = {"task_id": self.task_id}
        
        for i, (name, info) in enumerate(zip(self.hp_names, self.hp_info)):
            val = x_norm[i]
            
            if info[0] == "float":
                lo, hi = info[1], info[2]
                config[name] = lo + val * (hi - lo)
            elif info[0] == "float_log":
                lo, hi = np.log(info[1]), np.log(info[2])
                config[name] = np.exp(lo + val * (hi - lo))
            elif info[0] == "int":
                lo, hi = info[1], info[2]
                config[name] = int(round(lo + val * (hi - lo)))
            elif info[0] == "int_log":
                lo, hi = np.log(info[1]), np.log(info[2])
                config[name] = int(round(np.exp(lo + val * (hi - lo))))
            elif info[0] == "cat":
                choices = info[1]
                idx = min(int(val * len(choices)), len(choices) - 1)
                config[name] = choices[idx]
            elif info[0] == "ord":
                sequence = info[1]
                idx = min(int(val * len(sequence)), len(sequence) - 1)
                config[name] = sequence[idx]
        
        return config
    
    def __call__(self, x_norm: np.ndarray) -> float:
        """Evaluate and return AUC (to maximize)."""
        config = self.x_to_config(x_norm)
        try:
            result = self.bench.objective_function(config)[0]
            return float(result.get("auc", 0.5))
        except Exception as e:
            return 0.5


# ============================================================================
# RUN
# ============================================================================

def run_yahpo(wrapper: RBV2XGBoostWrapper, selector_class, n_trials: int, seed: int) -> float:
    selector = selector_class()
    
    opt = ALBA(
        bounds=wrapper.bounds,
        seed=seed,
        leaf_selector=selector,
        total_budget=n_trials,
        maximize=True,
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        y = wrapper(x)
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("="*70)
    print("  Leaf Selection Battle on YAHPO rbv2_xgboost")
    print("="*70)
    
    N_TRIALS = 80
    N_SEEDS = 6
    
    # Multiple task_ids (OpenML dataset ids)
    task_ids = ["3", "31", "37", "44"]  # 4 different datasets
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "Thompson": ThompsonSamplingLeafSelector,
        "TS-Opt": ThompsonOptimisticLeafSelector,
        "PI": ProbabilityOfImprovementLeafSelector,
    }
    
    all_winners = {}
    all_results = {}
    
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"  rbv2_xgboost task_id={task_id}")
        print(f"{'='*60}")
        
        try:
            wrapper = RBV2XGBoostWrapper(task_id)
            print(f"  Dim: {wrapper.dim}, HPs: {wrapper.hp_names[:5]}...")
            
            # Test
            test_x = np.random.rand(wrapper.dim)
            test_y = wrapper(test_x)
            print(f"  Test AUC: {test_y:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        
        results = {name: [] for name in selectors}
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}:", end=" ", flush=True)
            for name, selector_class in selectors.items():
                try:
                    val = run_yahpo(wrapper, selector_class, N_TRIALS, seed)
                    results[name].append(val)
                    print(f"{name}={val:.4f}", end=" ", flush=True)
                except Exception as e:
                    print(f"{name}=ERR", end=" ", flush=True)
                    results[name].append(0.5)
            print()
        
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if v > 0.4]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                best_val = np.max(vals)
                means[name] = mean_val
                print(f"{name:<12} | {mean_val:>10.4f} | {std_val:>10.4f} | {best_val:>10.4f}")
            else:
                means[name] = 0.0
                print(f"{name:<12} | {'N/A':>10}")
        
        winner = max(means, key=means.get) if means else "N/A"
        all_winners[task_id] = winner
        all_results[task_id] = means
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {name: 0 for name in selectors}
    for tid, w in all_winners.items():
        print(f"  task_id={tid}: {w}")
        if w in counts:
            counts[w] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get) if counts else "N/A"
    print(f"\n  🏆 Overall Winner: {overall}")


if __name__ == "__main__":
    main()
