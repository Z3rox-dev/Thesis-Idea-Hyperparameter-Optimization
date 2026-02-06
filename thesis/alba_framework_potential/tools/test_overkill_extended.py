#!/usr/bin/env python3
"""
Test OVERKILL ESTESO: Thompson vs Base vs altri su molte funzioni

Testiamo su:
- 10 funzioni sintetiche
- Dimensioni: 5, 10
- 15 seeds per robustezza
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
# TEST FUNCTIONS
# ============================================================================

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = sum((w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2) for i in range(len(x) - 1))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def schwefel(x):
    return 418.9829 * len(x) - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def griewank(x):
    sum_sq = sum(xi**2 for xi in x) / 4000
    prod_cos = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_sq - prod_cos + 1

def michalewicz(x, m=10):
    return -sum(np.sin(xi) * np.sin((i+1) * xi**2 / np.pi)**(2*m) for i, xi in enumerate(x))

def styblinski_tang(x):
    return 0.5 * sum(xi**4 - 16*xi**2 + 5*xi for xi in x)

def dixon_price(x):
    term1 = (x[0] - 1)**2
    term2 = sum((i+1) * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x)))
    return term1 + term2


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

def run_experiment(func, bounds, selector_class, n_trials, seed):
    from alba_framework_potential.optimizer import ALBA
    
    def objective(x):
        return -func(x)
    
    selector = selector_class()
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        leaf_selector=selector,
        total_budget=n_trials,
        maximize=True,
    )
    
    for _ in range(n_trials):
        x = opt.ask()
        y = objective(x)
        opt.tell(x, y)
    
    return -opt.best_y


def main():
    print("="*80)
    print("  TEST OVERKILL ESTESO: Thompson Sampling vs Base vs altri")
    print("="*80)
    
    N_TRIALS = 100
    N_SEEDS = 15
    
    test_cases = [
        # (name, func, bounds_type, dim)
        ("Sphere-5D", sphere, "std", 5),
        ("Sphere-10D", sphere, "std", 10),
        ("Rosenbrock-5D", rosenbrock, "std", 5),
        ("Rosenbrock-10D", rosenbrock, "std", 10),
        ("Rastrigin-5D", rastrigin, "std", 5),
        ("Rastrigin-10D", rastrigin, "std", 10),
        ("Ackley-5D", ackley, "std", 5),
        ("Levy-5D", levy, "std", 5),
        ("Griewank-5D", griewank, "std", 5),
        ("Schwefel-5D", schwefel, "schwefel", 5),
        ("Michalewicz-5D", michalewicz, "mich", 5),
        ("StyblinskiTang-5D", styblinski_tang, "std", 5),
        ("DixonPrice-5D", dixon_price, "std", 5),
    ]
    
    def get_bounds(bounds_type, dim):
        if bounds_type == "std":
            return [(-5.0, 5.0)] * dim
        elif bounds_type == "schwefel":
            return [(-500.0, 500.0)] * dim
        elif bounds_type == "mich":
            return [(0.0, np.pi)] * dim
        return [(-5.0, 5.0)] * dim
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "Thompson": ThompsonSamplingLeafSelector,
        "InfoGain": InformationGainLeafSelector,
        "GP-UCB": GPUCBLeafSelector,
    }
    
    all_winners = {}
    all_results = {}
    
    for func_name, func, bounds_type, dim in test_cases:
        bounds = get_bounds(bounds_type, dim)
        print(f"\n{'='*60}")
        print(f"  {func_name}")
        print(f"{'='*60}")
        
        results = {name: [] for name in selectors}
        
        for seed in range(N_SEEDS):
            for name, selector_class in selectors.items():
                try:
                    val = run_experiment(func, bounds, selector_class, N_TRIALS, seed)
                    results[name].append(val)
                except Exception as e:
                    print(f"  Error {name} seed {seed}: {e}")
                    results[name].append(float('inf'))
        
        print(f"\n{'Method':<12} | {'Mean':>12} | {'Std':>12} | {'Best':>12}")
        print("-" * 56)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                best_val = np.min(vals)
                means[name] = mean_val
                print(f"{name:<12} | {mean_val:>12.4f} | {std_val:>12.4f} | {best_val:>12.4f}")
            else:
                means[name] = float('inf')
                print(f"{name:<12} | {'N/A':>12}")
        
        winner = min(means, key=means.get)
        all_winners[func_name] = winner
        all_results[func_name] = means
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    
    counts = {name: 0 for name in selectors}
    for fn, w in all_winners.items():
        print(f"  {fn:<20}: {w}")
        counts[w] += 1
    
    print("\n" + "-"*50)
    print("  WINS BY METHOD:")
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * c
        print(f"  {name:<12}: {c:>2} {bar}")
    
    overall = max(counts, key=counts.get)
    print(f"\n  🏆 Overall Winner: {overall} ({counts[overall]}/{len(test_cases)} wins)")
    
    # Pairwise comparison: Thompson vs Base
    print("\n" + "="*80)
    print("  Thompson vs Base HEAD-TO-HEAD")
    print("="*80)
    
    thompson_wins = 0
    base_wins = 0
    ties = 0
    
    for fn in all_results:
        t_val = all_results[fn].get("Thompson", float('inf'))
        b_val = all_results[fn].get("Base", float('inf'))
        
        diff_pct = (b_val - t_val) / b_val * 100 if b_val != 0 else 0
        
        if t_val < b_val * 0.99:  # Thompson wins by > 1%
            thompson_wins += 1
            result = "Thompson ✓"
        elif b_val < t_val * 0.99:  # Base wins by > 1%
            base_wins += 1
            result = "Base ✓"
        else:
            ties += 1
            result = "~Tie"
        
        print(f"  {fn:<20}: T={t_val:>10.4f} vs B={b_val:>10.4f} | {diff_pct:>+6.1f}% | {result}")
    
    print(f"\n  Thompson: {thompson_wins} wins")
    print(f"  Base: {base_wins} wins")
    print(f"  Ties: {ties}")
    
    if thompson_wins > base_wins:
        print(f"\n  🏆 Thompson è SIGNIFICATIVAMENTE migliore di Base!")
    elif base_wins > thompson_wins:
        print(f"\n  🏆 Base è migliore di Thompson!")
    else:
        print(f"\n  ≈ Risultati simili")


if __name__ == "__main__":
    main()
