#!/usr/bin/env python3
"""
Test Leaf Selection su JAHS-Bench-201

Usa il venv py39.
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

# JAHS imports
import jahs_bench


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
# JAHS WRAPPER
# ============================================================================

class JAHSWrapper:
    """Wrapper per JAHS-Bench-201."""
    
    # JAHS hyperparameters
    HP_NAMES = [
        "N", "W", "Resolution", "TrivialAugment", "Activation", "Op1", "Op2", 
        "Op3", "Op4", "Op5", "Op6", "LearningRate", "WeightDecay"
    ]
    
    # Continuous: LearningRate, WeightDecay
    # Discrete: tutto il resto
    
    def __init__(self, dataset: str = "cifar10"):
        self.bench = jahs_bench.Benchmark(
            task=dataset,
            download=False,
            save_dir="/mnt/workspace/jahs_bench_data"
        )
        self.dataset = dataset
        
        # Define bounds for normalization to [0,1]
        # N: {1,3,5}, W: {4,8,16}, Resolution: 0.25-1.0
        # TrivialAugment: {0,1}, Activation: {ReLU, Hardswish, Mish}
        # Op1-Op6: {0,1,2,3,4} (5 operations)
        # LearningRate: log scale 1e-3 to 1e0
        # WeightDecay: log scale 1e-5 to 1e-2
        
        self.bounds = [(0.0, 1.0)] * 13  # All normalized to [0,1]
        self.dim = 13
    
    def x_to_config(self, x_norm: np.ndarray) -> dict:
        """Map [0,1]^13 to JAHS config."""
        x = x_norm.copy()
        
        # N: {1,3,5}
        n_vals = [1, 3, 5]
        n_idx = min(int(x[0] * 3), 2)
        
        # W: {4,8,16}
        w_vals = [4, 8, 16]
        w_idx = min(int(x[1] * 3), 2)
        
        # Resolution: [0.25, 1.0]
        resolution = 0.25 + x[2] * 0.75
        
        # TrivialAugment: {False, True}
        trivial_augment = x[3] > 0.5
        
        # Activation: {ReLU, Hardswish, Mish}
        act_vals = ["ReLU", "Hardswish", "Mish"]
        act_idx = min(int(x[4] * 3), 2)
        
        # Op1-Op6: {0,1,2,3,4}
        ops = []
        for i in range(5, 11):
            op_idx = min(int(x[i] * 5), 4)
            ops.append(op_idx)
        
        # LearningRate: log [1e-3, 1e0]
        lr = 10 ** (-3 + x[11] * 3)
        
        # WeightDecay: log [1e-5, 1e-2]
        wd = 10 ** (-5 + x[12] * 3)
        
        return {
            "N": n_vals[n_idx],
            "W": w_vals[w_idx],
            "Resolution": resolution,
            "TrivialAugment": trivial_augment,
            "Activation": act_vals[act_idx],
            "Op1": ops[0],
            "Op2": ops[1],
            "Op3": ops[2],
            "Op4": ops[3],
            "Op5": ops[4],
            "Op6": ops[5],
            "LearningRate": lr,
            "WeightDecay": wd,
            "epoch": 200,  # Max epochs
        }
    
    def __call__(self, x_norm: np.ndarray) -> float:
        """Return validation accuracy (to maximize)."""
        config = self.x_to_config(x_norm)
        result = self.bench(config)
        return float(result[200]["valid-acc"])  # Accuracy at epoch 200


# ============================================================================
# RUN
# ============================================================================

def run_jahs(wrapper: JAHSWrapper, selector_class, n_trials: int, seed: int) -> float:
    selector = selector_class()
    
    opt = ALBA(
        bounds=wrapper.bounds,
        seed=seed,
        leaf_selector=selector,
        total_budget=n_trials,
        maximize=True,  # Maximize accuracy
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        try:
            y = wrapper(x)
        except Exception as e:
            y = 0.0
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("="*70)
    print("  Leaf Selection Battle on JAHS-Bench-201")
    print("="*70)
    
    N_TRIALS = 50  # JAHS is slower
    N_SEEDS = 5
    
    datasets = ["cifar10", "fashion_mnist", "colorectal_histology"]
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "Thompson": ThompsonSamplingLeafSelector,
        "TS-Opt": ThompsonOptimisticLeafSelector,
        "PI": ProbabilityOfImprovementLeafSelector,
    }
    
    all_winners = {}
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  JAHS: {dataset}")
        print(f"{'='*60}")
        
        try:
            print("  Loading benchmark...", flush=True)
            wrapper = JAHSWrapper(dataset)
            print(f"  Dim: {wrapper.dim}")
            
            # Quick test
            test_x = np.random.rand(wrapper.dim)
            test_y = wrapper(test_x)
            print(f"  Test eval: {test_y:.4f}")
        except Exception as e:
            print(f"  Failed to load: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        results = {name: [] for name in selectors}
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed}:", end=" ", flush=True)
            for name, selector_class in selectors.items():
                try:
                    val = run_jahs(wrapper, selector_class, N_TRIALS, seed)
                    results[name].append(val)
                    print(f"{name}={val:.4f}", end=" ", flush=True)
                except Exception as e:
                    print(f"{name}=ERR", end=" ", flush=True)
                    results[name].append(0.0)
            print()
        
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if v > 0.1]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                best_val = np.max(vals)  # Higher is better
                means[name] = mean_val
                print(f"{name:<12} | {mean_val:>10.4f} | {std_val:>10.4f} | {best_val:>10.4f}")
            else:
                means[name] = 0.0
                print(f"{name:<12} | {'N/A':>10}")
        
        winner = max(means, key=means.get) if means else "N/A"
        all_winners[dataset] = winner
        all_results[dataset] = means
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {name: 0 for name in selectors}
    for ds, w in all_winners.items():
        print(f"  {ds:<25}: {w}")
        if w in counts:
            counts[w] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get) if counts else "N/A"
    print(f"\n  🏆 Overall Winner: {overall}")


if __name__ == "__main__":
    main()
