#!/usr/bin/env python3
"""
Test OVERKILL su YAHPO - Versione corretta
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
from yahpo_gym import benchmark_set, local_config
import ConfigSpace as CS


# ============================================================================
# SELECTORS
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
# YAHPO WRAPPER - CORRETTO
# ============================================================================

class YAHPOWrapper:
    """Wrapper corretto per YAHPO che gestisce ConfigSpace."""
    
    def __init__(self, scenario: str, task_id: str):
        local_config.init_config()
        local_config.set_data_path("/mnt/workspace/data/")
        
        self.bench = benchmark_set.BenchmarkSet(scenario)
        self.bench.set_instance(task_id)
        self.task_id = task_id
        self.config_space = self.bench.get_opt_space()
        
        # Costruisci bounds - solo HP non-costanti
        self.bounds = []
        self.hp_specs = []  # (name, type, info)
        
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, CS.Constant):
                continue
                
            if isinstance(hp, CS.UniformFloatHyperparameter):
                if hp.log:
                    # Log-scale: trasforma in [log(lower), log(upper)]
                    self.bounds.append((np.log(hp.lower), np.log(hp.upper)))
                    self.hp_specs.append((hp.name, "float_log", hp.lower, hp.upper))
                else:
                    self.bounds.append((hp.lower, hp.upper))
                    self.hp_specs.append((hp.name, "float", hp.lower, hp.upper))
                    
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                if hp.log:
                    self.bounds.append((np.log(hp.lower), np.log(hp.upper)))
                    self.hp_specs.append((hp.name, "int_log", hp.lower, hp.upper))
                else:
                    self.bounds.append((float(hp.lower), float(hp.upper)))
                    self.hp_specs.append((hp.name, "int", hp.lower, hp.upper))
                    
            elif isinstance(hp, CS.CategoricalHyperparameter):
                n_choices = len(hp.choices)
                self.bounds.append((0.0, float(n_choices) - 1e-6))
                self.hp_specs.append((hp.name, "cat", hp.choices, None))
                
            elif isinstance(hp, CS.OrdinalHyperparameter):
                n_seq = len(hp.sequence)
                self.bounds.append((0.0, float(n_seq) - 1e-6))
                self.hp_specs.append((hp.name, "ord", hp.sequence, None))
        
        self.dim = len(self.bounds)
    
    def x_to_config(self, x: np.ndarray) -> dict:
        """Converte punto continuo in configurazione YAHPO."""
        config = {"task_id": self.task_id}
        
        for i, spec in enumerate(self.hp_specs):
            name = spec[0]
            hp_type = spec[1]
            
            if hp_type == "float":
                config[name] = float(x[i])
            elif hp_type == "float_log":
                config[name] = float(np.exp(x[i]))
            elif hp_type == "int":
                config[name] = int(round(x[i]))
            elif hp_type == "int_log":
                config[name] = int(round(np.exp(x[i])))
            elif hp_type == "cat":
                choices = spec[2]
                idx = min(int(x[i]), len(choices) - 1)
                config[name] = choices[idx]
            elif hp_type == "ord":
                sequence = spec[2]
                idx = min(int(x[i]), len(sequence) - 1)
                config[name] = sequence[idx]
        
        return config
    
    def __call__(self, x: np.ndarray) -> float:
        config = self.x_to_config(x)
        result = self.bench.objective_function(config)[0]
        return float(result.get("auc", 0.5))


# ============================================================================
# RUN
# ============================================================================

def run_yahpo(wrapper: YAHPOWrapper, selector_class, n_trials: int, seed: int):
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
        try:
            y = wrapper(x)
        except Exception as e:
            y = 0.5
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("="*70)
    print("  TEST OVERKILL su YAHPO - Versione Corretta")
    print("="*70)
    
    N_TRIALS = 50
    N_SEEDS = 5
    
    # Test su rbv2_xgboost (più semplice)
    benchmarks = [
        ("rbv2_xgboost", "3"),
        ("rbv2_xgboost", "31"),
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
            wrapper = YAHPOWrapper(scenario, task_id)
            print(f"  Dim: {wrapper.dim}")
            
            # Quick test
            test_x = np.array([(b[0] + b[1]) / 2 for b in wrapper.bounds])
            test_y = wrapper(test_x)
            print(f"  Test eval: {test_y:.4f}")
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        
        results = {name: [] for name in selectors}
        
        for seed in range(N_SEEDS):
            for name, selector_class in selectors.items():
                try:
                    val = run_yahpo(wrapper, selector_class, N_TRIALS, seed)
                    results[name].append(val)
                    print(f"  {name} seed {seed}: {val:.4f}")
                except Exception as e:
                    print(f"  Error {name} seed {seed}: {e}")
                    results[name].append(0.5)
        
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if v > 0.3]
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


if __name__ == "__main__":
    main()
