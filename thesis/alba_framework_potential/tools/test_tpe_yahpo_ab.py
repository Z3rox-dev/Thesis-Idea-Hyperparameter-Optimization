#!/usr/bin/env python3
"""
Test A/B: TPE-inspired Leaf Selection su YAHPO Benchmarks

YAHPO è molto più veloce di JAHS perché usa surrogates leggeri.
Testa le strategie di leaf selection su benchmark tabular ML:
- iaml_xgboost (XGBoost su diversi dataset OpenML)
- rbv2_xgboost (RandomBot v2 XGBoost)

Non richiede conda py39 - gira con python3 base.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')


# ============================================================================
# LEAF SELECTORS
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
# YAHPO WRAPPER
# ============================================================================

class YAHPOBenchmark:
    """Wrapper for YAHPO benchmark."""
    
    def __init__(self, scenario: str, instance: str):
        from yahpo_gym import BenchmarkSet, local_config
        local_config.init_config()
        local_config.set_data_path("/mnt/workspace/data")
        
        self.bench = BenchmarkSet(scenario)
        self.bench.set_instance(instance)
        self.config_space = self.bench.get_opt_space()
        
    def get_param_space(self) -> Dict[str, Any]:
        """Convert ConfigSpace to ALBA param_space format."""
        import ConfigSpace as CS
        
        param_space = {}
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            if isinstance(hp, CS.CategoricalHyperparameter):
                param_space[name] = list(hp.choices)
            elif isinstance(hp, CS.UniformFloatHyperparameter):
                if hp.log:
                    param_space[name] = (hp.lower, hp.upper, "log")
                else:
                    param_space[name] = (hp.lower, hp.upper)
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                if hp.log:
                    # ALBA doesn't have int_log, use regular int
                    param_space[name] = (hp.lower, hp.upper, "int")
                else:
                    param_space[name] = (hp.lower, hp.upper, "int")
            elif isinstance(hp, CS.OrdinalHyperparameter):
                param_space[name] = list(hp.sequence)
            elif isinstance(hp, CS.Constant):
                param_space[name] = [hp.value]
            else:
                # Fallback: treat as continuous
                param_space[name] = (0.0, 1.0)
        
        return param_space
    
    def __call__(self, config: Dict[str, Any]) -> float:
        """Evaluate configuration, return accuracy (to maximize)."""
        try:
            result = self.bench.objective_function(config)
        except Exception:
            return 0.0
            
        # YAHPO returns list of dicts, get first result
        if isinstance(result, list):
            result = result[0]
        
        # Try common accuracy keys
        for key in ["acc", "auc", "val_accuracy"]:
            if key in result:
                val = result[key]
                if np.isfinite(val):
                    return float(val)
        
        # mmce is error, convert to accuracy
        if "mmce" in result:
            val = result["mmce"]
            if np.isfinite(val):
                return 1.0 - float(val)
        
        return 0.0


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_yahpo_experiment(
    bench: YAHPOBenchmark,
    selector_class,
    n_trials: int,
    seed: int,
) -> float:
    """Run ALBA on YAHPO benchmark with given leaf selector."""
    from alba_framework_potential.optimizer import ALBA
    
    param_space = bench.get_param_space()
    selector = selector_class()
    
    opt = ALBA(
        param_space=param_space,
        seed=seed,
        maximize=True,
        leaf_selector=selector,
        total_budget=n_trials,
    )
    
    best_y = -np.inf
    for i in range(n_trials):
        cfg = opt.ask()
        try:
            y = bench(cfg)
        except Exception as e:
            y = 0.0  # Failed evaluation
        opt.tell(cfg, y)
        
        if y > best_y:
            best_y = y
    
    return best_y


def main():
    print("="*70)
    print("  TEST A/B: TPE-inspired Leaf Selection su YAHPO Benchmarks")
    print("="*70)
    
    # Configuration
    N_TRIALS = 100
    N_SEEDS = 5
    
    # YAHPO scenarios and instances (use valid instances for each scenario)
    BENCHMARKS = [
        ("iaml_xgboost", "40981"),   # XGBoost on Australian dataset
        ("iaml_xgboost", "1489"),    # XGBoost on Phoneme dataset
        ("iaml_ranger", "40981"),    # Random Forest on Australian
        ("iaml_ranger", "1489"),     # Random Forest on Phoneme
    ]
    
    SELECTORS = {
        "Base": UCBSoftmaxLeafSelector,
        "RelDens": RelativeDensityLeafSelector,
        "DensNorm": DensityNormalizedLeafSelector,
    }
    
    all_results = {}
    
    for scenario, instance in BENCHMARKS:
        bench_name = f"{scenario}/{instance}"
        print(f"\n{'='*60}")
        print(f"  Benchmark: {bench_name}")
        print(f"{'='*60}")
        
        try:
            bench = YAHPOBenchmark(scenario, instance)
            print(f"  Param space: {len(bench.get_param_space())} parameters")
        except Exception as e:
            print(f"  ERROR loading benchmark: {e}")
            continue
        
        results = {name: [] for name in SELECTORS}
        
        for seed in range(N_SEEDS):
            for name, selector_class in SELECTORS.items():
                print(f"  Running {name} seed {seed}...", end=" ", flush=True)
                try:
                    best_val = run_yahpo_experiment(
                        bench=bench,
                        selector_class=selector_class,
                        n_trials=N_TRIALS,
                        seed=seed,
                    )
                    results[name].append(best_val)
                    print(f"acc={best_val:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[name].append(0.0)
        
        # Print results
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
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
        all_results[bench_name] = {"means": means, "winner": winner, "raw": results}
        print(f"\n  🏆 Winner: {winner}")
    
    # Final Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {name: 0 for name in SELECTORS}
    for bench_name, data in all_results.items():
        winner = data["winner"]
        print(f"  {bench_name:<30}: {winner}")
        if winner in counts:
            counts[winner] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get) if any(counts.values()) else "N/A"
    print(f"\n  🏆 Overall Winner: {overall}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/mnt/workspace/thesis/benchmark_results/yahpo_tpe_ab_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    save_data = {
        "config": {"n_trials": N_TRIALS, "n_seeds": N_SEEDS},
        "results": {
            k: {
                "means": {mk: convert(mv) for mk, mv in v["means"].items()},
                "winner": v["winner"],
            }
            for k, v in all_results.items()
        },
        "summary": {"counts": {k: convert(v) for k, v in counts.items()}, "overall_winner": overall},
    }
    
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
