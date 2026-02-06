#!/usr/bin/env python3
"""
BATTLE ROYALE: Tutte le strategie di Leaf Selection

Strategie testate:
1. Base (good_ratio deterministico)
2. Thompson Sampling (Beta posterior)
3. EI (Expected Improvement)
4. PI (Probability of Improvement)
5. UCB Classico (mean + c*std)
6. Thompson Optimistic (max di k samples)
7. Boltzmann (softmax con temperature decay)

Funzioni: 15+ funzioni sintetiche
Dimensioni: 3, 5, 10, 15
Seeds: 10 per robustezza
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector
from alba_framework_potential.cube import Cube


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
            # Prendi il max di k samples - più ottimistico
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
class ExpectedImprovementLeafSelector(UCBSoftmaxLeafSelector):
    """
    EI (Expected Improvement): E[max(0, X - best)]
    
    Per una Beta(a,b), calcoliamo EI rispetto alla best_ratio globale.
    """
    
    def _compute_ei(self, alpha: float, beta: float, best: float) -> float:
        """Compute Expected Improvement for Beta distribution."""
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        std = np.sqrt(var) + 1e-8
        
        # Approssimazione gaussiana della Beta
        z = (mean - best) / std
        ei = std * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
        return max(0, ei)
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        # Best ratio osservato
        best_ratio = max(c.good_ratio() for c in leaves)
        
        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta = (c.n_trials - c.n_good) + 1
            
            ei = self._compute_ei(alpha, beta, best_ratio)
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(ei + exploration + model_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


@dataclass
class ProbabilityOfImprovementLeafSelector(UCBSoftmaxLeafSelector):
    """
    PI (Probability of Improvement): P(X > best)
    """
    
    def _compute_pi(self, alpha: float, beta: float, best: float) -> float:
        """Compute P(X > best) for Beta(alpha, beta)."""
        # P(X > best) = 1 - CDF(best)
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


@dataclass
class UCBClassicLeafSelector(UCBSoftmaxLeafSelector):
    """UCB classico: mean + c * std, con c fisso."""
    c: float = 2.0  # Exploration constant
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        scores = []
        for c_leaf in leaves:
            alpha = c_leaf.n_good + 1
            beta = (c_leaf.n_trials - c_leaf.n_good) + 1
            
            mean = alpha / (alpha + beta)
            var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            std = np.sqrt(var)
            
            ucb = mean + self.c * std
            
            exploration = self.base_exploration / np.sqrt(1 + c_leaf.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c_leaf.lgs_model is not None:
                n_pts = len(c_leaf.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(ucb + exploration + model_bonus)
        
        return leaves[int(np.argmax(scores))]


@dataclass
class BoltzmannLeafSelector(UCBSoftmaxLeafSelector):
    """
    Boltzmann/Softmax con temperature che decresce.
    
    Temperature alta = exploration, bassa = exploitation.
    """
    initial_temp: float = 1.0
    final_temp: float = 0.1
    n_calls: int = field(default=0, init=False)
    total_calls: int = 100
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        self.n_calls += 1
        
        # Temperature decay
        progress = min(1.0, self.n_calls / self.total_calls)
        temp = self.initial_temp * (1 - progress) + self.final_temp * progress
        
        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus
            
            scores.append(ratio + exploration + model_bonus)
        
        # Boltzmann distribution
        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = (scores_arr - scores_arr.max()) / temp
        probs = np.exp(scores_arr)
        probs = probs / probs.sum()
        
        return leaves[int(rng.choice(len(leaves), p=probs))]


@dataclass
class ThompsonPessimisticLeafSelector(UCBSoftmaxLeafSelector):
    """Thompson Pessimistic: prende la mediana di k samples (più cauto)."""
    k_samples: int = 5
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("Empty leaves")
        
        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta = (c.n_trials - c.n_good) + 1
            samples = rng.beta(alpha, beta, size=self.k_samples)
            sample = np.median(samples)
            
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

def zakharov(x):
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i+1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4

def powell(x):
    n = len(x)
    result = 0
    for i in range(n // 4):
        idx = 4 * i
        if idx + 3 < n:
            result += (x[idx] + 10*x[idx+1])**2
            result += 5 * (x[idx+2] - x[idx+3])**2
            result += (x[idx+1] - 2*x[idx+2])**4
            result += 10 * (x[idx] - x[idx+3])**4
    return result

def sum_squares(x):
    return sum((i+1) * xi**2 for i, xi in enumerate(x))

def trid(x):
    return sum((xi - 1)**2 for xi in x) - sum(x[i] * x[i-1] for i in range(1, len(x)))

def booth(x):
    """2D function extended to nD"""
    result = 0
    for i in range(0, len(x)-1, 2):
        result += (x[i] + 2*x[i+1] - 7)**2 + (2*x[i] + x[i+1] - 5)**2
    return result


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

def run_experiment(func, bounds, selector_class, n_trials, seed):
    from alba_framework_potential.optimizer import ALBA
    
    def objective(x):
        return -func(x)
    
    selector = selector_class()
    if hasattr(selector, 'total_calls'):
        selector.total_calls = n_trials
    
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
    print("  BATTLE ROYALE: Leaf Selection Strategies")
    print("="*80)
    
    N_TRIALS = 100
    N_SEEDS = 10
    
    # Test su varie dimensioni
    dimensions = [3, 5, 10, 15]
    
    # Funzioni con bounds appropriati
    functions = [
        ("Sphere", sphere, "std"),
        ("Rosenbrock", rosenbrock, "std"),
        ("Rastrigin", rastrigin, "std"),
        ("Ackley", ackley, "std"),
        ("Levy", levy, "std"),
        ("Griewank", griewank, "std"),
        ("Schwefel", schwefel, "schwefel"),
        ("Michalewicz", michalewicz, "mich"),
        ("StyblinskiTang", styblinski_tang, "std"),
        ("DixonPrice", dixon_price, "std"),
        ("Zakharov", zakharov, "std"),
        ("SumSquares", sum_squares, "std"),
        ("Trid", trid, "trid"),
    ]
    
    def get_bounds(bounds_type, dim):
        if bounds_type == "std":
            return [(-5.0, 5.0)] * dim
        elif bounds_type == "schwefel":
            return [(-500.0, 500.0)] * dim
        elif bounds_type == "mich":
            return [(0.0, np.pi)] * dim
        elif bounds_type == "trid":
            return [(-dim**2, dim**2)] * dim
        return [(-5.0, 5.0)] * dim
    
    selectors = {
        "Base": UCBSoftmaxLeafSelector,
        "Thompson": ThompsonSamplingLeafSelector,
        "TS-Opt": ThompsonOptimisticLeafSelector,
        "EI": ExpectedImprovementLeafSelector,
        "PI": ProbabilityOfImprovementLeafSelector,
        "UCB": UCBClassicLeafSelector,
        "Boltzmann": BoltzmannLeafSelector,
    }
    
    # Risultati per dimensione
    dim_wins = {d: {name: 0 for name in selectors} for d in dimensions}
    all_results = []
    
    for dim in dimensions:
        print(f"\n{'#'*80}")
        print(f"  DIMENSION: {dim}D")
        print(f"{'#'*80}")
        
        for func_name, func, bounds_type in functions:
            bounds = get_bounds(bounds_type, dim)
            test_name = f"{func_name}-{dim}D"
            print(f"\n  {test_name}...", end=" ", flush=True)
            
            results = {name: [] for name in selectors}
            
            for seed in range(N_SEEDS):
                for name, selector_class in selectors.items():
                    try:
                        val = run_experiment(func, bounds, selector_class, N_TRIALS, seed)
                        results[name].append(val)
                    except Exception as e:
                        results[name].append(float('inf'))
            
            # Find winner
            means = {}
            for name, vals in results.items():
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    means[name] = np.mean(vals)
                else:
                    means[name] = float('inf')
            
            winner = min(means, key=means.get)
            dim_wins[dim][winner] += 1
            all_results.append((test_name, dim, winner, means))
            print(f"🏆 {winner} ({means[winner]:.4f})")
    
    # Summary per dimensione
    print("\n" + "="*80)
    print("  RESULTS BY DIMENSION")
    print("="*80)
    
    for dim in dimensions:
        print(f"\n  {dim}D:")
        for name, count in sorted(dim_wins[dim].items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"    {name:<12}: {count:>2} {bar}")
    
    # Overall summary
    print("\n" + "="*80)
    print("  OVERALL SUMMARY")
    print("="*80)
    
    total_wins = {name: 0 for name in selectors}
    for dim in dimensions:
        for name in selectors:
            total_wins[name] += dim_wins[dim][name]
    
    total_tests = len(functions) * len(dimensions)
    print(f"\n  Total tests: {total_tests}")
    print()
    
    for name, count in sorted(total_wins.items(), key=lambda x: -x[1]):
        pct = count / total_tests * 100
        bar = "█" * (count // 2)
        print(f"  {name:<12}: {count:>3} wins ({pct:>5.1f}%) {bar}")
    
    overall_winner = max(total_wins, key=total_wins.get)
    print(f"\n  🏆 CHAMPION: {overall_winner} ({total_wins[overall_winner]}/{total_tests} wins)")
    
    # Head-to-head: Thompson vs all
    print("\n" + "="*80)
    print("  HEAD-TO-HEAD: Thompson vs Others")
    print("="*80)
    
    for other in selectors:
        if other == "Thompson":
            continue
        
        t_wins = 0
        o_wins = 0
        ties = 0
        
        for test_name, dim, winner, means in all_results:
            t_val = means.get("Thompson", float('inf'))
            o_val = means.get(other, float('inf'))
            
            if t_val < o_val * 0.99:
                t_wins += 1
            elif o_val < t_val * 0.99:
                o_wins += 1
            else:
                ties += 1
        
        print(f"  Thompson vs {other:<12}: T={t_wins:>2} | {other[:1]}={o_wins:>2} | Tie={ties:>2}")


if __name__ == "__main__":
    main()
