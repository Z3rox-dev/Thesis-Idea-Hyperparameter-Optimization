#!/usr/bin/env python3
"""
Test OVERKILL su YAHPO: Thompson vs InfoGain vs GP-UCB vs Base
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube
from alba_framework_potential.optimizer import ALBA


# ============================================================================
# COPY SELECTORS
# ============================================================================

@dataclass
class ThompsonSamplingLeafSelector(UCBSoftmaxLeafSelector):
    """Thompson Sampling: campiona dalla distribuzione Beta posterior."""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
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
class InformationGainLeafSelector(UCBSoftmaxLeafSelector):
    """Information Gain: scegli la foglia che massimizza l'informazione attesa."""
    
    def _entropy(self, alpha: float, beta: float) -> float:
        from scipy import special
        if alpha <= 0 or beta <= 0:
            return 0.0
        return (
            special.betaln(alpha, beta)
            - (alpha - 1) * special.digamma(alpha)
            - (beta - 1) * special.digamma(beta)
            + (alpha + beta - 2) * special.digamma(alpha + beta)
        )
    
    def _expected_info_gain(self, c: Cube) -> float:
        alpha = c.n_good + 1
        beta = (c.n_trials - c.n_good) + 1
        h_prior = self._entropy(alpha, beta)
        p = alpha / (alpha + beta)
        h_post_good = self._entropy(alpha + 1, beta)
        h_post_bad = self._entropy(alpha, beta + 1)
        h_expected = p * h_post_good + (1 - p) * h_post_bad
        ig = h_prior - h_expected
        return ig * p
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        scores = []
        for c in leaves:
            ig = self._expected_info_gain(c)
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(ig + exploration + model_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


@dataclass
class GPUCBLeafSelector(UCBSoftmaxLeafSelector):
    """GP-UCB style: UCB = mean + beta * std."""
    
    iteration: int = field(default=0, init=False)
    delta: float = 0.1
    
    def _get_beta(self, n_total: int) -> float:
        return np.sqrt(2 * np.log(max(1, n_total) / self.delta))
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        self.iteration += 1
        n_total = sum(c.n_trials for c in leaves)
        beta = self._get_beta(n_total)
        
        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            mean = alpha / (alpha + beta_param)
            var = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
            std = np.sqrt(var)
            ucb = mean + beta * std
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(ucb + exploration + model_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


# ============================================================================
# YAHPO WRAPPER
# ============================================================================

class YAHPOBenchmark:
    def __init__(self, scenario: str, task_id: str):
        from yahpo_gym import benchmark_set, local_config
        local_config.init_config()
        local_config.set_data_path("/mnt/workspace/data/")
        
        self.bench = benchmark_set.BenchmarkSet(scenario)
        self.bench.set_instance(task_id)
        self.config_space = self.bench.get_opt_space()
        self.hp_names = [hp.name for hp in self.config_space.get_hyperparameters()]
        self._build_bounds()
    
    def _build_bounds(self):
        import ConfigSpace as CS
        self.bounds = []
        self.hp_info = []
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, CS.UniformFloatHyperparameter):
                self.bounds.append((hp.lower, hp.upper))
                self.hp_info.append(("float", hp.log, hp.lower, hp.upper))
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                self.bounds.append((float(hp.lower), float(hp.upper)))
                self.hp_info.append(("int", hp.log, hp.lower, hp.upper))
            elif isinstance(hp, CS.CategoricalHyperparameter):
                n_choices = len(hp.choices)
                self.bounds.append((0.0, float(n_choices - 0.001)))
                self.hp_info.append(("cat", hp.choices))
            elif isinstance(hp, CS.OrdinalHyperparameter):
                n_seq = len(hp.sequence)
                self.bounds.append((0.0, float(n_seq - 0.001)))
                self.hp_info.append(("ord", hp.sequence))
    
    def __call__(self, x):
        config = {}
        for i, (name, info) in enumerate(zip(self.hp_names, self.hp_info)):
            if info[0] == "float":
                config[name] = x[i]
            elif info[0] == "int":
                config[name] = int(round(x[i]))
            elif info[0] == "cat":
                idx = min(int(x[i]), len(info[1]) - 1)
                config[name] = info[1][idx]
            elif info[0] == "ord":
                idx = min(int(x[i]), len(info[1]) - 1)
                config[name] = info[1][idx]
        
        result = self.bench.objective_function(config)[0]
        # auc is primary, maximize it (higher is better)
        return float(result.get("auc", 0.5))


# ============================================================================
# RUN
# ============================================================================

def run_yahpo(bench: YAHPOBenchmark, selector_class, n_trials: int, seed: int):
    selector = selector_class()
    opt = ALBA(
        bounds=bench.bounds,
        seed=seed,
        leaf_selector=selector,
        total_budget=n_trials,
        maximize=True,
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        try:
            y = bench(x)
        except Exception as e:
            y = 0.5  # Default AUC
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("="*70)
    print("  TEST OVERKILL su YAHPO: Thompson vs InfoGain vs GP-UCB vs Base")
    print("="*70)
    
    N_TRIALS = 50
    N_SEEDS = 5
    
    benchmarks = [
        ("iaml_xgboost", "40981"),  # credit-g
        ("iaml_xgboost", "1489"),   # phoneme
        ("iaml_ranger", "40981"),   # credit-g
        ("iaml_ranger", "1489"),    # phoneme
    ]
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "Thompson": ThompsonSamplingLeafSelector,
        "InfoGain": InformationGainLeafSelector,
        "GP-UCB": GPUCBLeafSelector,
    }
    
    all_winners = {}
    all_results = {}
    
    for scenario, task_id in benchmarks:
        bench_name = f"{scenario}_{task_id}"
        print(f"\n{'='*60}")
        print(f"  {bench_name}")
        print(f"{'='*60}")
        
        try:
            bench = YAHPOBenchmark(scenario, task_id)
        except Exception as e:
            print(f"  Failed to load benchmark: {e}")
            continue
        
        results = {name: [] for name in selectors}
        
        for seed in range(N_SEEDS):
            for name, selector_class in selectors.items():
                try:
                    val = run_yahpo(bench, selector_class, N_TRIALS, seed)
                    results[name].append(val)
                    print(f"  {name} seed {seed}: {val:.4f}")
                except Exception as e:
                    print(f"  Error {name} seed {seed}: {e}")
                    results[name].append(0.0)
        
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if v > 0.3]  # Filter very low AUC
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
        all_winners[bench_name] = winner
        all_results[bench_name] = means
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {name: 0 for name in selectors}
    for bn, w in all_winners.items():
        print(f"  {bn:<25}: {w}")
        if w in counts:
            counts[w] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get) if counts else "N/A"
    print(f"\n  🏆 Overall Winner: {overall}")


if __name__ == "__main__":
    main()
