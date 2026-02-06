#!/usr/bin/env python3
"""
Test Leaf Selection su ParamNet Benchmark

Usa il venv paramnet e HPOBench ParamNet surrogates.
"""

import sys
import os

# Path setup
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import ConfigSpace as CS
from scipy import stats

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube
from alba_framework_potential.optimizer import ALBA

# Import ParamNet benchmarks
from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetMnistOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)


# ============================================================================
# LEAF SELECTORS
# ============================================================================

@dataclass
class ThompsonSamplingLeafSelector(UCBSoftmaxLeafSelector):
    """Thompson Sampling: campiona dalla Beta posterior."""
    
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
    """Thompson Optimistic: prende il max di k samples dalla Beta."""
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
    """PI (Probability of Improvement): P(X > best)"""
    
    def _compute_pi(self, alpha: float, beta: float, best: float) -> float:
        from scipy.stats import beta as beta_dist
        pi = 1 - beta_dist.cdf(best, alpha, beta)
        return pi
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        best_ratio = max(c.good_ratio() for c in leaves)
        
        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            
            pi = self._compute_pi(alpha, beta_param, best_ratio)
            
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
# PARAMNET WRAPPER
# ============================================================================

PARAMNET_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
    "mnist": ParamNetMnistOnStepsBenchmark,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark,
    "poker": ParamNetPokerOnStepsBenchmark,
}


class ParamNetWrapper:
    """Wrapper per ParamNet che mappa [0,1]^d a ConfigSpace."""
    
    def __init__(self, dataset: str):
        bench_cls = PARAMNET_MAP[dataset.lower()]
        self.bench = bench_cls()
        self.cs = self.bench.get_configuration_space()
        self.hps = self.cs.get_hyperparameters()
        
        self.bounds = []
        self.types = []
        
        for hp in self.hps:
            if isinstance(hp, CS.UniformFloatHyperparameter):
                self.bounds.append((float(hp.lower), float(hp.upper)))
                self.types.append("float")
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                self.bounds.append((float(hp.lower), float(hp.upper)))
                self.types.append("int")
            else:
                raise ValueError(f"Unsupported: {type(hp)}")
        
        self.dim = len(self.bounds)
    
    def x_to_config(self, x_norm: np.ndarray) -> CS.Configuration:
        """Mappa x_norm in [0,1]^d a Configuration."""
        values = {}
        for i, (hp, (lo, hi), t) in enumerate(zip(self.hps, self.bounds, self.types)):
            v = lo + float(x_norm[i]) * (hi - lo)
            if t == "int":
                v = int(round(v))
                v = max(int(hp.lower), min(int(hp.upper), int(v)))
            values[hp.name] = v
        return CS.Configuration(self.cs, values=values)
    
    def __call__(self, x_norm: np.ndarray) -> float:
        """Valuta e ritorna validation loss (da minimizzare)."""
        config = self.x_to_config(x_norm)
        result = self.bench.objective_function(config)
        # ParamNet ritorna validation loss - vogliamo minimizzare
        return float(result["function_value"])


# ============================================================================
# RUN
# ============================================================================

def run_paramnet(wrapper: ParamNetWrapper, selector_class, n_trials: int, seed: int) -> float:
    """Run ALBA su ParamNet, ritorna best validation loss."""
    selector = selector_class()
    
    # ParamNet usa bounds [0,1]^d
    unit_bounds = [(0.0, 1.0)] * wrapper.dim
    
    opt = ALBA(
        bounds=unit_bounds,
        seed=seed,
        leaf_selector=selector,
        total_budget=n_trials,
        maximize=False,  # Minimizziamo validation loss
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        try:
            y = wrapper(x)
        except Exception as e:
            y = 1.0  # Bad loss
        opt.tell(x, y)
    
    return opt.best_y


def main():
    print("="*70)
    print("  Leaf Selection Battle on ParamNet")
    print("="*70)
    
    N_TRIALS = 100
    N_SEEDS = 8
    
    datasets = ["adult", "letter", "optdigits", "mnist", "higgs", "poker"]  # 6 datasets
    
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
        print(f"  ParamNet: {dataset}")
        print(f"{'='*60}")
        
        try:
            wrapper = ParamNetWrapper(dataset)
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
                    val = run_paramnet(wrapper, selector_class, N_TRIALS, seed)
                    results[name].append(val)
                    print(f"{name}={val:.4f}", end=" ", flush=True)
                except Exception as e:
                    print(f"{name}=ERR", end=" ", flush=True)
                    results[name].append(1.0)
            print()
        
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if v < 0.9]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                best_val = np.min(vals)
                means[name] = mean_val
                print(f"{name:<12} | {mean_val:>10.4f} | {std_val:>10.4f} | {best_val:>10.4f}")
            else:
                means[name] = 1.0
                print(f"{name:<12} | {'N/A':>10}")
        
        # Lower is better for validation loss
        winner = min(means, key=means.get) if means else "N/A"
        all_winners[dataset] = winner
        all_results[dataset] = means
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {name: 0 for name in selectors}
    for ds, w in all_winners.items():
        print(f"  {ds:<15}: {w}")
        if w in counts:
            counts[w] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get) if counts else "N/A"
    print(f"\n  🏆 Overall Winner: {overall}")


if __name__ == "__main__":
    main()
