#!/usr/bin/env python3
"""
Test OVERKILL: Strategie avanzate di Leaf Selection

Varianti sofisticate ispirate a:
1. Thompson Sampling - campionamento dalla Beta posterior
2. Information Gain - massimizzare quanto impariamo
3. UCB con decay temporale - penalizzare foglie "stale"
4. GP-UCB style - upper confidence bound bayesiano
5. Ensemble adattivo - combinazione pesata di strategie
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube


# ============================================================================
# OVERKILL LEAF SELECTORS
# ============================================================================

@dataclass
class ThompsonSamplingLeafSelector(UCBSoftmaxLeafSelector):
    """
    Thompson Sampling: campiona dalla distribuzione Beta posterior.
    
    Invece di calcolare un punteggio deterministico, campiona dalla 
    distribuzione Beta(n_good+1, n_bad+1) per ogni foglia.
    
    Questo bilancia automaticamente exploration/exploitation in modo
    probabilistico ottimale.
    """
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        samples = []
        for c in leaves:
            # Beta posterior: Beta(successes + 1, failures + 1)
            alpha = c.n_good + 1
            beta = (c.n_trials - c.n_good) + 1
            
            # Campiona dalla Beta
            sample = rng.beta(alpha, beta)
            
            # Aggiungi exploration bonus per foglie poco esplorate
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            # Model bonus
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            samples.append(sample + exploration + model_bonus)
        
        # Scegli la foglia con sample più alto (greedy su Thompson sample)
        return leaves[int(np.argmax(samples))]


@dataclass
class InformationGainLeafSelector(UCBSoftmaxLeafSelector):
    """
    Information Gain: scegli la foglia che massimizza l'informazione attesa.
    
    IG(leaf) = H(prior) - E[H(posterior)]
    
    Favorisce foglie con alta incertezza MA anche alta probabilità di essere buone.
    """
    
    def _entropy(self, alpha: float, beta: float) -> float:
        """Entropy of Beta(alpha, beta) distribution."""
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
        """Expected information gain from sampling this leaf."""
        alpha = c.n_good + 1
        beta = (c.n_trials - c.n_good) + 1
        
        # Current entropy
        h_prior = self._entropy(alpha, beta)
        
        # Expected posterior entropy
        # E[H(posterior)] ≈ p * H(alpha+1, beta) + (1-p) * H(alpha, beta+1)
        p = alpha / (alpha + beta)  # P(good)
        h_post_good = self._entropy(alpha + 1, beta)
        h_post_bad = self._entropy(alpha, beta + 1)
        h_expected = p * h_post_good + (1 - p) * h_post_bad
        
        # Information gain
        ig = h_prior - h_expected
        
        # Weight by probability of being good (focus on promising leaves)
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
    """
    GP-UCB style: Upper Confidence Bound con incertezza bayesiana.
    
    UCB = mean + beta * std
    
    dove mean e std sono stimati dalla Beta posterior.
    beta aumenta logaritmicamente col tempo (come in GP-UCB).
    """
    
    iteration: int = field(default=0, init=False)
    delta: float = 0.1  # Confidence level
    
    def _get_beta(self, n_total: int) -> float:
        """Compute beta coefficient for UCB."""
        # GP-UCB: beta = 2 * log(n^(d/2+2) * pi^2 / (3 * delta))
        # Simplified version for leaf selection
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
            
            # Mean and std of Beta distribution
            mean = alpha / (alpha + beta_param)
            var = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
            std = np.sqrt(var)
            
            # UCB score
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


@dataclass
class TemporalDecayLeafSelector(UCBSoftmaxLeafSelector):
    """
    Temporal Decay: penalizza foglie non visitate da tempo.
    
    Mantiene un contatore per ogni foglia e applica un decadimento
    esponenziale per favorire foglie "stale".
    """
    
    last_visit: Dict[int, int] = field(default_factory=dict, init=False)
    global_step: int = field(default=0, init=False)
    decay_rate: float = 0.1
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        self.global_step += 1
        
        scores = []
        for i, c in enumerate(leaves):
            ratio = c.good_ratio()
            
            # Temporal decay bonus
            last = self.last_visit.get(id(c), 0)
            staleness = self.global_step - last
            decay_bonus = self.decay_rate * np.log1p(staleness)
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(ratio + exploration + model_bonus + decay_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        
        idx = int(rng.choice(len(leaves), p=probs))
        self.last_visit[id(leaves[idx])] = self.global_step
        return leaves[idx]


@dataclass
class EnsembleLeafSelector(UCBSoftmaxLeafSelector):
    """
    Ensemble: combina più strategie con pesi adattivi.
    
    Vota tra Thompson, UCB, e Information Gain, pesando in base
    a quanto ogni strategia ha funzionato storicamente.
    """
    
    ts_selector: Any = field(default=None, init=False)
    ig_selector: Any = field(default=None, init=False)
    gpucb_selector: Any = field(default=None, init=False)
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0], init=False)
    history: List[List[float]] = field(default_factory=lambda: [[], [], []], init=False)
    last_choice: int = field(default=0, init=False)
    last_value: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        self.ts_selector = ThompsonSamplingLeafSelector()
        self.ig_selector = InformationGainLeafSelector()
        self.gpucb_selector = GPUCBLeafSelector()
    
    def _update_weights(self, reward: float):
        """Update weights based on last selection's reward."""
        # Exponential weights update
        eta = 0.1
        self.history[self.last_choice].append(reward)
        
        # Update weight based on recent performance
        for i in range(3):
            if self.history[i]:
                recent = self.history[i][-10:]  # Last 10 rewards
                self.weights[i] = np.exp(eta * np.mean(recent))
        
        # Normalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        selectors = [self.ts_selector, self.ig_selector, self.gpucb_selector]
        
        # Each selector votes for a leaf
        votes = []
        for sel in selectors:
            chosen = sel.select(leaves, dim, stagnating, rng)
            votes.append(leaves.index(chosen))
        
        # Weighted voting
        leaf_scores = np.zeros(len(leaves))
        for i, vote in enumerate(votes):
            leaf_scores[vote] += self.weights[i]
        
        # Softmax selection
        leaf_scores = leaf_scores - leaf_scores.max()
        probs = np.exp(leaf_scores * 5.0)
        probs = probs / probs.sum()
        
        idx = int(rng.choice(len(leaves), p=probs))
        self.last_choice = votes.index(idx) if idx in votes else 0
        return leaves[idx]


@dataclass  
class BestScoreWeightedLeafSelector(UCBSoftmaxLeafSelector):
    """
    Combina good_ratio con best_score normalizzato.
    
    score = alpha * good_ratio + (1-alpha) * normalized_best_score
    
    Bilancia la "densità di buoni" con il "miglior valore trovato".
    """
    
    alpha: float = 0.6  # Weight for good_ratio
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")
        
        # Normalize best_scores across leaves
        best_scores = [c.best_score for c in leaves]
        min_bs, max_bs = min(best_scores), max(best_scores)
        range_bs = max_bs - min_bs if max_bs > min_bs else 1.0
        
        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            
            # Normalized best score in [0, 1]
            norm_best = (c.best_score - min_bs) / range_bs if range_bs > 0 else 0.5
            
            # Combined score
            combined = self.alpha * ratio + (1 - self.alpha) * norm_best
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(combined + exploration + model_bonus)
        
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
    print("="*70)
    print("  TEST OVERKILL: Strategie Avanzate di Leaf Selection")
    print("="*70)
    
    N_TRIALS = 100
    N_SEEDS = 10
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
        "Thompson": ThompsonSamplingLeafSelector,
        "InfoGain": InformationGainLeafSelector,
        "GP-UCB": GPUCBLeafSelector,
        "Temporal": TemporalDecayLeafSelector,
        "BestW": BestScoreWeightedLeafSelector,
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
    header = f"{'Function':<12} | " + " | ".join(f"{n:>8}" for n in selectors.keys())
    print(header)
    print("-" * len(header))
    for fn in [t[0] for t in test_cases]:
        if fn in all_results:
            row = f"{fn:<12} | " + " | ".join(
                f"{all_results[fn].get(n, float('inf')):>8.2f}" for n in selectors.keys()
            )
            print(row)


if __name__ == "__main__":
    main()
