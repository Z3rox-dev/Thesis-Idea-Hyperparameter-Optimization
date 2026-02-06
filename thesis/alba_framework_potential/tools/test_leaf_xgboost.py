#!/usr/bin/env python3
"""
Test Leaf Selection su XGBoost Tabular Benchmark

Questo benchmark ha 20 dimensioni continue - ideale per testare alta dimensione.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube
from alba_framework_potential.optimizer import ALBA

# Import XGBoost benchmark
sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_xgboost_tabular import xgboost_tabular


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
# RUN
# ============================================================================

def run_xgb(selector_class, n_trials: int, seed: int) -> float:
    """Run ALBA on XGBoost tabular benchmark (20D)."""
    selector = selector_class()
    
    # 20D continuous in [0,1]
    bounds = [(0.0, 1.0)] * 20
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        leaf_selector=selector,
        total_budget=n_trials,
        maximize=True,  # Maximize F1 score
    )
    
    for i in range(n_trials):
        x = opt.ask()
        try:
            metrics = xgboost_tabular(x, use_gpu=False, trial_seed=seed + i)
            y = metrics["val_f1"]  # F1 score on validation
        except Exception as e:
            y = 0.0
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("="*70)
    print("  Leaf Selection Battle on XGBoost Tabular (20D)")
    print("="*70)
    
    N_TRIALS = 50  # XGBoost è più lento
    N_SEEDS = 5
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "Thompson": ThompsonSamplingLeafSelector,
        "TS-Opt": ThompsonOptimisticLeafSelector,
        "PI": ProbabilityOfImprovementLeafSelector,
    }
    
    results = {name: [] for name in selectors}
    
    for seed in range(N_SEEDS):
        print(f"\nSeed {seed}:", flush=True)
        for name, selector_class in selectors.items():
            try:
                val = run_xgb(selector_class, N_TRIALS, seed * 100)
                results[name].append(val)
                print(f"  {name}: {val:.4f}")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                results[name].append(0.0)
    
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
    print("-" * 50)
    
    means = {}
    for name, vals in results.items():
        vals = [v for v in vals if v > 0.1]
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
    print(f"\n  🏆 Winner: {winner}")
    
    # Head-to-head
    print("\n" + "-"*40)
    print("HEAD-TO-HEAD:")
    for name in selectors:
        if name == "Thompson":
            continue
        t_vals = results["Thompson"]
        o_vals = results[name]
        t_wins = sum(1 for t, o in zip(t_vals, o_vals) if t > o * 1.01)
        o_wins = sum(1 for t, o in zip(t_vals, o_vals) if o > t * 1.01)
        ties = N_SEEDS - t_wins - o_wins
        print(f"  Thompson vs {name}: T={t_wins} | {name[0]}={o_wins} | Tie={ties}")


if __name__ == "__main__":
    main()
