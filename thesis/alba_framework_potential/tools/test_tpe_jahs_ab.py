#!/usr/bin/env python3
"""
Test A/B: TPE-inspired Leaf Selection su JAHS-Bench-201

Testa le strategie di leaf selection su tutti e 3 i dataset JAHS:
- cifar10
- fashion_mnist  
- colorectal_histology

Usa il param_space corretto con tipi delle feature.

IMPORTANT: Eseguire con conda py39:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/alba_framework_potential/test_tpe_jahs_ab.py
"""

from __future__ import annotations

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')


# ============================================================================
# JAHS PARAM SPACE
# ============================================================================

def build_jahs_param_space(nepochs: int = 200) -> Dict[str, Any]:
    """Build JAHS param space with correct types for ALBA."""
    return {
        "Optimizer": ["SGD"],  # Fixed
        "epoch": [nepochs],    # Fixed
        "LearningRate": (1e-3, 1.0, "log"),      # Continuous log-scale
        "WeightDecay": (1e-5, 1e-2, "log"),      # Continuous log-scale
        "N": [1, 3, 5],                          # Categorical (3 choices)
        "W": [4, 8, 16],                         # Categorical (3 choices)
        "Resolution": [0.25, 0.5, 1.0],          # Categorical (3 choices)
        "Activation": ["ReLU", "Hardswish", "Mish"],  # Categorical
        "TrivialAugment": [True, False],         # Categorical (bool)
        "Op1": [0, 1, 2, 3, 4],                  # Categorical (5 choices)
        "Op2": [0, 1, 2, 3, 4],                  # Categorical
        "Op3": [0, 1, 2, 3, 4],                  # Categorical
        "Op4": [0, 1, 2, 3, 4],                  # Categorical
        "Op5": [0, 1, 2, 3, 4],                  # Categorical
        "Op6": [0, 1, 2, 3, 4],                  # Categorical
    }


# ============================================================================
# LEAF SELECTORS (copied from test_tpe_leaf_selection_ab.py)
# ============================================================================

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube


@dataclass
class RelativeDensityLeafSelector(UCBSoftmaxLeafSelector):
    """TPE-inspired: usa la densità relativa invece del good_ratio assoluto."""
    
    def _compute_ratio(self, c: Cube, leaves: List[Cube]) -> float:
        N_good_global = sum(leaf.n_good for leaf in leaves) + 1
        N_total_global = sum(leaf.n_trials for leaf in leaves) + 1
        
        n_good = c.n_good + 0.5
        n_trials = c.n_trials + 1
        
        global_good_ratio = N_good_global / N_total_global
        local_good_ratio = n_good / n_trials
        
        relative = local_good_ratio / (global_good_ratio + 1e-9)
        score = relative / (1 + relative)
        
        return score

    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")

        scores = []
        for c in leaves:
            ratio = self._compute_ratio(c, leaves)
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            scores.append(ratio + exploration + model_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]


@dataclass
class DensityNormalizedLeafSelector(UCBSoftmaxLeafSelector):
    """Good ratio normalizzato per densità di punti nella foglia."""
    
    def _compute_ratio(self, c: Cube, leaves: List[Cube]) -> float:
        good_ratio = (c.n_good + 1) / (c.n_trials + 2)
        volume = c.volume() + 1e-9
        local_density = c.n_trials / volume
        total_trials = sum(leaf.n_trials for leaf in leaves) + 1
        total_volume = sum(leaf.volume() for leaf in leaves) + 1e-9
        global_density = total_trials / total_volume
        density_factor = local_density / (global_density + 1e-9)
        density_factor = np.clip(density_factor, 0.2, 5.0)
        score = good_ratio * np.sqrt(density_factor)
        return score

    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")

        scores = []
        for c in leaves:
            ratio = self._compute_ratio(c, leaves)
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            scores.append(ratio + exploration + model_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

# Global benchmark cache to avoid reloading
_BENCH_CACHE = {}

def get_jahs_benchmark(task: str, save_dir: str = "/mnt/workspace/jahs_bench_data"):
    """Get or create JAHS benchmark (cached)."""
    if task not in _BENCH_CACHE:
        import jahs_bench
        print(f"    Loading JAHS benchmark for {task} (lazy mode)...", flush=True)
        _BENCH_CACHE[task] = jahs_bench.Benchmark(
            task=task,
            kind="surrogate",
            download=False,
            save_dir=save_dir,
            # Use lazy loading to speed up initialization
        )
        print(f"    Loaded!", flush=True)
    return _BENCH_CACHE[task]


def run_jahs_experiment(
    task: str,
    selector_class,
    n_trials: int,
    seed: int,
    bench,  # Pre-loaded benchmark
    nepochs: int = 200,
) -> float:
    """Run ALBA on JAHS benchmark with given leaf selector."""
    
    # Import ALBA framework
    from alba_framework_potential.optimizer import ALBA
    
    # Build param space
    param_space = build_jahs_param_space(nepochs)
    
    # Create selector
    selector = selector_class()
    
    # Create optimizer
    opt = ALBA(
        param_space=param_space,
        seed=seed,
        maximize=True,  # Maximize valid-acc
        leaf_selector=selector,
        total_budget=n_trials,
    )
    
    best_y = -np.inf
    for i in range(n_trials):
        cfg = opt.ask()
        result = bench(cfg, nepochs=nepochs)
        y = float(result[nepochs]["valid-acc"])
        opt.tell(cfg, y)
        
        if y > best_y:
            best_y = y
    
    return best_y


def main():
    print("="*70)
    print("  TEST A/B: TPE-inspired Leaf Selection su JAHS-Bench-201")
    print("="*70)
    
    # Check we're in py39
    try:
        import jahs_bench
        print("✓ jahs_bench imported successfully")
    except ImportError:
        print("ERROR: jahs_bench not found. Run with:")
        print("  source /mnt/workspace/miniconda3/bin/activate py39")
        print("  python thesis/alba_framework_potential/test_tpe_jahs_ab.py")
        return 1
    
    # Configuration - reduced for faster testing
    N_TRIALS = 50   # Budget per run (reduced from 100)
    N_SEEDS = 3     # Number of seeds (reduced from 5)
    NEPOCHS = 200
    
    # Test on single task first, then expand
    TASKS = ["cifar10"]  # Start with one task
    
    SELECTORS = {
        "Base": UCBSoftmaxLeafSelector,
        "RelDens": RelativeDensityLeafSelector,
        "DensNorm": DensityNormalizedLeafSelector,
    }
    
    all_results = {}
    
    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"  JAHS Task: {task}")
        print(f"{'='*60}")
        
        # Load benchmark once per task
        bench = get_jahs_benchmark(task)
        
        results = {name: [] for name in SELECTORS}
        
        for seed in range(N_SEEDS):
            for name, selector_class in SELECTORS.items():
                print(f"  Running {name} seed {seed}...", end=" ", flush=True)
                try:
                    best_acc = run_jahs_experiment(
                        task=task,
                        selector_class=selector_class,
                        n_trials=N_TRIALS,
                        seed=seed,
                        bench=bench,
                        nepochs=NEPOCHS,
                    )
                    results[name].append(best_acc)
                    print(f"acc={best_acc:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[name].append(0.0)
        
        # Print results for this task
        print(f"\n{'Method':<12} | {'Mean Acc':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if v > 0]
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
        all_results[task] = {"results": results, "means": means, "winner": winner}
        print(f"\n  🏆 Winner: {winner}")
    
    # Final Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY ACROSS ALL JAHS TASKS")
    print("="*70)
    
    counts = {name: 0 for name in SELECTORS}
    for task, data in all_results.items():
        winner = data["winner"]
        print(f"  {task:<25}: {winner}")
        if winner in counts:
            counts[winner] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get) if counts else "N/A"
    print(f"\n  🏆 Overall Winner: {overall}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/mnt/workspace/thesis/benchmark_results/jahs_tpe_ab_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    save_data = {
        "config": {"n_trials": N_TRIALS, "n_seeds": N_SEEDS, "nepochs": NEPOCHS},
        "results": {
            task: {
                "means": {k: convert(v) for k, v in data["means"].items()},
                "winner": data["winner"],
                "raw": {k: [convert(x) for x in v] for k, v in data["results"].items()},
            }
            for task, data in all_results.items()
        },
        "summary": {"counts": counts, "overall_winner": overall},
    }
    
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
