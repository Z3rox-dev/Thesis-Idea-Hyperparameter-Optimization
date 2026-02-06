#!/usr/bin/env python3
"""
Test A/B: Strategie TPE-inspired per good_ratio

Testa diverse varianti di come calcolare il "punteggio di qualità" di una foglia:

1. ALBA_base: good_ratio = (n_good + 1) / (n_trials + 2)
2. Relative Density: quanto la foglia è sovra-rappresentata nei buoni
   = (n_good / N_good_global) / (n_trials / N_trials_global)
3. TPE l/g ratio: densità buoni / densità cattivi nella foglia
4. Volume-weighted: good_ratio normalizzato per volume della foglia
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.gamma import QuantileAnnealedGammaScheduler
from alba_framework_potential.leaf_selection import LeafSelector, UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube


# ============================================================================
# TPE-INSPIRED LEAF SELECTORS
# ============================================================================

@dataclass
class RelativeDensityLeafSelector(UCBSoftmaxLeafSelector):
    """
    TPE-inspired: usa la densità relativa invece del good_ratio assoluto.
    
    relative_score = (n_good / N_good_global) / (n_trials / N_total_global)
                   = (n_good * N_total_global) / (n_trials * N_good_global)
    
    Se > 1: la foglia è sovra-rappresentata nei buoni
    Se < 1: la foglia è sotto-rappresentata nei buoni
    """
    
    def _compute_ratio(self, c: Cube, leaves: List[Cube]) -> float:
        # Calcola totali globali
        N_good_global = sum(leaf.n_good for leaf in leaves) + 1  # +1 per evitare div by 0
        N_total_global = sum(leaf.n_trials for leaf in leaves) + 1
        
        n_good = c.n_good + 0.5  # Smoothing
        n_trials = c.n_trials + 1  # Smoothing
        
        # Relative density ratio
        global_good_ratio = N_good_global / N_total_global
        local_good_ratio = n_good / n_trials
        
        # Rapporto: quanto questa foglia è "sovra-rappresentata" nei buoni
        relative = local_good_ratio / (global_good_ratio + 1e-9)
        
        # Normalizza in [0, 1] con sigmoid-like trasformation
        # relative=1 -> 0.5, relative>1 -> >0.5, relative<1 -> <0.5
        score = relative / (1 + relative)  # Varia in (0, 1)
        
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
class TPELGRatioLeafSelector(UCBSoftmaxLeafSelector):
    """
    TPE-inspired: usa il rapporto l(x)/g(x) dove:
    - l(x) = densità dei punti BUONI nella foglia (n_good / volume)
    - g(x) = densità dei punti CATTIVI nella foglia (n_bad / volume)
    
    score = l / (l + g) = n_good / (n_good + n_bad) = n_good / n_trials
    
    Ma ponderiamo per il volume per favorire foglie più piccole con alta densità.
    """
    
    def _compute_ratio(self, c: Cube, leaves: List[Cube]) -> float:
        n_good = c.n_good + 1  # Beta prior
        n_bad = (c.n_trials - c.n_good) + 1
        
        # l/g ratio
        lg_ratio = n_good / (n_bad + 1e-9)
        
        # Fattore volume: foglie più piccole con stessa densità sono preferite
        volume = c.volume()
        total_volume = sum(leaf.volume() for leaf in leaves)
        volume_ratio = volume / (total_volume + 1e-9)
        
        # Score: favorisci alto l/g ratio, penalizza foglie troppo grandi
        # Se volume_ratio piccolo (foglia piccola), il fattore aumenta lo score
        volume_factor = 1.0 / (volume_ratio + 0.1)  # Boost per foglie piccole
        volume_factor = min(volume_factor, 5.0)  # Cap per evitare estremi
        
        # Combina: lg_ratio * volume_factor, poi normalizza
        raw_score = lg_ratio * volume_factor
        
        # Normalizza con log per comprimere range
        score = np.log1p(raw_score) / 5.0  # Scala approssimativa
        score = min(score, 1.0)
        
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
    """
    Good ratio normalizzato per densità di punti nella foglia.
    
    Idea: una foglia con 5 buoni su 10 in un volume piccolo è migliore
    di una foglia con 5 buoni su 10 in un volume grande.
    
    score = good_ratio * density_factor
    dove density_factor = (n_trials / volume) / (N_total / total_volume)
    """
    
    def _compute_ratio(self, c: Cube, leaves: List[Cube]) -> float:
        # Good ratio base
        good_ratio = (c.n_good + 1) / (c.n_trials + 2)
        
        # Calcola densità locale e globale
        volume = c.volume() + 1e-9
        local_density = c.n_trials / volume
        
        total_trials = sum(leaf.n_trials for leaf in leaves) + 1
        total_volume = sum(leaf.volume() for leaf in leaves) + 1e-9
        global_density = total_trials / total_volume
        
        # Fattore di densità relativa
        density_factor = local_density / (global_density + 1e-9)
        
        # Clamp per evitare estremi
        density_factor = np.clip(density_factor, 0.2, 5.0)
        
        # Score finale
        score = good_ratio * np.sqrt(density_factor)  # sqrt per smussare
        
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


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

def run_experiment(func, bounds, leaf_selector_class, n_trials, seed):
    """Run optimization with a specific leaf selector."""
    
    def objective(x):
        return -func(x)  # ALBA massimizza
    
    # Create selector instance
    selector = leaf_selector_class()
    
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
    print("="*70)
    print("  TEST A/B: TPE-inspired Leaf Selection Strategies")
    print("="*70)
    
    N_TRIALS = 100
    N_SEEDS = 15
    DIM = 5
    
    bounds_std = [(-5.0, 5.0)] * DIM
    bounds_schwefel = [(-500.0, 500.0)] * DIM
    
    test_cases = [
        ("Sphere", sphere, bounds_std),
        ("Rosenbrock", rosenbrock, bounds_std),
        ("Rastrigin", rastrigin, bounds_std),
        ("Ackley", ackley, bounds_std),
        ("Levy", levy, bounds_std),
        ("Griewank", griewank, bounds_std),
        ("Schwefel", schwefel, bounds_schwefel),
    ]
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "RelDens": RelativeDensityLeafSelector,
        "LGRatio": TPELGRatioLeafSelector,
        "DensNorm": DensityNormalizedLeafSelector,
    }
    
    all_winners = {}
    all_results = {}
    
    for func_name, func, bounds in test_cases:
        print(f"\n{'='*60}")
        print(f"  {func_name} (dim={len(bounds)})")
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
        
        print(f"\n{'Method':<12} | {'Mean':>10} | {'Std':>10} | {'Best':>10}")
        print("-" * 50)
        
        means = {}
        for name, vals in results.items():
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                best_val = np.min(vals)
                means[name] = mean_val
                print(f"{name:<12} | {mean_val:>10.4f} | {std_val:>10.4f} | {best_val:>10.4f}")
            else:
                means[name] = float('inf')
                print(f"{name:<12} | {'N/A':>10}")
        
        winner = min(means, key=means.get)
        all_winners[func_name] = winner
        all_results[func_name] = means
        print(f"\n  🏆 Winner: {winner}")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    counts = {name: 0 for name in selectors}
    for fn, w in all_winners.items():
        print(f"  {fn:<15}: {w}")
        counts[w] += 1
    
    print("\n" + "-"*40)
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {c} wins")
    
    overall = max(counts, key=counts.get)
    print(f"\n  🏆 Overall Winner: {overall}")
    
    # Detailed comparison table
    print("\n" + "="*70)
    print("  MEAN VALUES BY FUNCTION")
    print("="*70)
    header = f"{'Function':<15} | " + " | ".join(f"{n:>10}" for n in selectors.keys())
    print(header)
    print("-" * len(header))
    for fn in [t[0] for t in test_cases]:
        if fn in all_results:
            row = f"{fn:<15} | " + " | ".join(
                f"{all_results[fn].get(n, float('inf')):>10.2f}" for n in selectors.keys()
            )
            print(row)


if __name__ == "__main__":
    main()
