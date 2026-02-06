"""
Test A/B: good_ratio vs TPE-KDE vs Density Count

Confrontiamo 3 strategie per calcolare il bonus potenziale:

1. good_ratio: (n_good + 1) / (n_trials + 2) - solo info locale
2. TPE-KDE: l(x)/g(x) con KDE - info globale smooth
3. Density Count: conta punti buoni globali nella foglia - info globale discreta

Density Count è concettualmente più pulito di TPE-KDE:
- Non usa KDE (computazionalmente più leggero)
- Conta direttamente quanti punti "buoni" globali cadono nella foglia
- Normalizza per volume
"""

import numpy as np
from scipy.stats import gaussian_kde
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube
from alba_framework_potential.leaf_selection import ThompsonSamplingLeafSelector


# ============================================================================
# DENSITY COUNT CALCULATOR
# ============================================================================

class DensityCountCalculator:
    """
    Calcola il density bonus per ogni foglia contando i punti buoni globali.
    
    density_score = n_good_in_leaf / volume_leaf
    
    Concettualmente più semplice di TPE-KDE, usa la stessa info globale.
    """
    
    def __init__(self, gamma: float = 0.25):
        self.gamma = gamma
        self._good_points: List[np.ndarray] = []
        self._all_points: List[np.ndarray] = []
        self._all_scores: List[float] = []
        self._gamma_threshold: float = -np.inf
    
    def record(self, x: np.ndarray, score: float):
        """Record observation."""
        self._all_points.append(x.copy())
        self._all_scores.append(score)
        
        # Update gamma threshold and good points periodically
        if len(self._all_scores) >= 4:
            n_good = max(1, int(self.gamma * len(self._all_scores)))
            sorted_scores = sorted(self._all_scores, reverse=True)
            self._gamma_threshold = sorted_scores[min(n_good, len(sorted_scores)-1)]
            self._good_points = [
                p for p, s in zip(self._all_points, self._all_scores) 
                if s >= self._gamma_threshold
            ]
    
    def get_leaf_density(self, leaf: Cube) -> float:
        """
        Conta quanti punti buoni globali cadono in questa foglia.
        Normalizza per volume.
        """
        if not self._good_points:
            return 0.5
        
        # Count good points in leaf
        n_good_in_leaf = sum(1 for p in self._good_points if leaf.contains(p))
        
        # Normalize by volume and total good points
        volume = leaf.volume()
        if volume < 1e-12:
            volume = 1e-12
        
        # Density = (punti buoni in foglia / punti buoni totali) / (volume foglia / volume totale)
        # Ma volume totale è costante, quindi:
        density = n_good_in_leaf / (volume * len(self._good_points) + 1e-9)
        
        return density
    
    def get_leaf_score(self, leaf: Cube) -> float:
        """
        Restituisce score normalizzato [0, 1].
        Alto = molti punti buoni in piccolo volume = promettente
        """
        density = self.get_leaf_density(leaf)
        # Sigmoid per normalizzare
        return 1.0 / (1.0 + np.exp(-density * 10))


# ============================================================================
# TPE-KDE (from previous test)
# ============================================================================

class TPERatioCalculator:
    def __init__(self, gamma: float = 0.25):
        self.gamma = gamma
        self._l_kde = None
        self._g_kde = None
        self._is_fitted = False
    
    def fit(self, all_points: np.ndarray, all_scores: np.ndarray):
        if len(all_points) < 4:
            self._is_fitted = False
            return
        
        n = len(all_points)
        n_good = max(1, int(self.gamma * n))
        idx_sorted = np.argsort(all_scores)[::-1]
        
        good_idx = idx_sorted[:n_good]
        bad_idx = idx_sorted[n_good:]
        
        good_points = all_points[good_idx]
        bad_points = all_points[bad_idx] if len(bad_idx) > 0 else all_points
        
        try:
            if good_points.shape[0] >= 2:
                self._l_kde = gaussian_kde(good_points.T, bw_method='scott')
            if bad_points.shape[0] >= 2:
                self._g_kde = gaussian_kde(bad_points.T, bw_method='scott')
            self._is_fitted = self._l_kde is not None
        except Exception:
            self._is_fitted = False
    
    def get_ratio(self, x: np.ndarray) -> float:
        if not self._is_fitted:
            return 0.5
        x = np.atleast_2d(x).T
        l_density = self._l_kde(x)[0] if self._l_kde is not None else 1e-9
        g_density = self._g_kde(x)[0] if self._g_kde is not None else 1e-9
        l_density = max(l_density, 1e-12)
        g_density = max(g_density, 1e-12)
        ratio = l_density / g_density
        log_ratio = np.log(ratio + 1e-12)
        return 1.0 / (1.0 + np.exp(-log_ratio))
    
    def get_leaf_ratio(self, leaf: Cube) -> float:
        center = leaf.center()
        return self.get_ratio(center)


# ============================================================================
# LEAF SELECTORS
# ============================================================================

@dataclass
class DensityBonusLeafSelector(ThompsonSamplingLeafSelector):
    """Thompson + Density Count bonus."""
    potential_weight: float = 0.5
    density_calc: Any = field(default=None)
    
    def __post_init__(self):
        self.density_calc = DensityCountCalculator()
    
    def record_observation(self, x: np.ndarray, score: float):
        self.density_calc.record(x, score)
    
    def select(self, leaves: list, dim: int, stagnating: bool, rng) -> "Cube":
        if len(leaves) == 1:
            return leaves[0]

        scores = []
        for c in leaves:
            # Thompson base
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            sample = rng.beta(alpha, beta_param)
            if not np.isfinite(sample):
                sample = 0.5

            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            # Density bonus - use raw density, scale appropriately
            density_bonus = 0.0
            if len(self.density_calc._good_points) > 0:
                n_good_in_leaf = sum(1 for p in self.density_calc._good_points if c.contains(p))
                # Bonus based on fraction of good points in this leaf
                density_bonus = self.potential_weight * n_good_in_leaf / len(self.density_calc._good_points)

            scores.append(sample + exploration + model_bonus + density_bonus)

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
class TPEBonusLeafSelector(ThompsonSamplingLeafSelector):
    """Thompson + TPE-KDE bonus."""
    potential_weight: float = 0.5
    tpe_calc: Any = field(default=None)
    _all_points: List = field(default_factory=list)
    _all_scores: List = field(default_factory=list)
    
    def __post_init__(self):
        if self.tpe_calc is None:
            self.tpe_calc = TPERatioCalculator()
    
    def record_observation(self, x: np.ndarray, score: float):
        self._all_points.append(x.copy())
        self._all_scores.append(score)
        if len(self._all_points) >= 10 and len(self._all_points) % 5 == 0:
            self.tpe_calc.fit(np.array(self._all_points), np.array(self._all_scores))
    
    def select(self, leaves: list, dim: int, stagnating: bool, rng) -> "Cube":
        if len(leaves) == 1:
            return leaves[0]

        scores = []
        for c in leaves:
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            sample = rng.beta(alpha, beta_param)
            if not np.isfinite(sample):
                sample = 0.5

            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            # TPE bonus
            tpe_bonus = 0.0
            if self.tpe_calc._is_fitted:
                tpe_ratio = self.tpe_calc.get_leaf_ratio(c)
                tpe_bonus = self.potential_weight * tpe_ratio

            scores.append(sample + exploration + model_bonus + tpe_bonus)

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
class GoodRatioBonusLeafSelector(ThompsonSamplingLeafSelector):
    """Thompson + good_ratio bonus (baseline)."""
    potential_weight: float = 0.5
    
    def select(self, leaves: list, dim: int, stagnating: bool, rng) -> "Cube":
        if len(leaves) == 1:
            return leaves[0]

        # Compute good_ratio for all leaves
        ratios = [c.good_ratio() for c in leaves]
        max_ratio = max(ratios) if ratios else 1.0
        min_ratio = min(ratios) if ratios else 0.0
        range_ratio = max_ratio - min_ratio if max_ratio > min_ratio else 1.0

        scores = []
        for i, c in enumerate(leaves):
            alpha = c.n_good + 1
            beta_param = (c.n_trials - c.n_good) + 1
            sample = rng.beta(alpha, beta_param)
            if not np.isfinite(sample):
                sample = 0.5

            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            # Good ratio bonus (normalized)
            gr_norm = (ratios[i] - min_ratio) / range_ratio if range_ratio > 0 else 0.5
            gr_bonus = self.potential_weight * gr_norm

            scores.append(sample + exploration + model_bonus + gr_bonus)

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

def sphere(x): return np.sum(x**2)
def rosenbrock(x): return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))
def rastrigin(x): return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
def ackley(x):
    n = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e
def levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0])**2 + np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2)) + (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
def griewank(x): return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1


# ============================================================================
# A/B TEST
# ============================================================================

def run_ab_test():
    print("=" * 70)
    print("A/B TEST: good_ratio vs TPE-KDE vs Density Count")
    print("=" * 70)
    
    functions = [
        ("Sphere-5D", sphere, 5, [(-5, 5)] * 5),
        ("Sphere-10D", sphere, 10, [(-5, 5)] * 10),
        ("Rosenbrock-3D", rosenbrock, 3, [(-2, 2)] * 3),
        ("Rosenbrock-5D", rosenbrock, 5, [(-2, 2)] * 5),
        ("Rastrigin-5D", rastrigin, 5, [(-5.12, 5.12)] * 5),
        ("Ackley-5D", ackley, 5, [(-5, 5)] * 5),
        ("Levy-5D", levy, 5, [(-10, 10)] * 5),
        ("Griewank-5D", griewank, 5, [(-10, 10)] * 5),
    ]
    
    seeds = [42, 123, 456, 789, 1011, 2022]
    budget = 100
    
    results = {'good_ratio': {}, 'tpe': {}, 'density': {}, 'base': {}}
    wins = {'good_ratio': 0, 'tpe': 0, 'density': 0}
    
    for fname, func, dim, bounds in functions:
        print(f"\n{fname}:")
        for k in results: results[k][fname] = []
        
        for seed in seeds:
            # 1. good_ratio
            gr_sel = GoodRatioBonusLeafSelector()
            opt_gr = ALBA(bounds=bounds, seed=seed, total_budget=budget, use_potential_field=False, leaf_selector=gr_sel)
            for _ in range(budget):
                x = opt_gr.ask()
                y = func(x)
                opt_gr.tell(x, -y)
            results['good_ratio'][fname].append(-opt_gr.best_y)
            
            # 2. TPE-KDE
            tpe_sel = TPEBonusLeafSelector()
            opt_tpe = ALBA(bounds=bounds, seed=seed, total_budget=budget, use_potential_field=False, leaf_selector=tpe_sel)
            for _ in range(budget):
                x = opt_tpe.ask()
                y = func(x)
                opt_tpe.tell(x, -y)
                tpe_sel.record_observation(x, -y)
            results['tpe'][fname].append(-opt_tpe.best_y)
            
            # 3. Density Count
            den_sel = DensityBonusLeafSelector()
            opt_den = ALBA(bounds=bounds, seed=seed, total_budget=budget, use_potential_field=False, leaf_selector=den_sel)
            for _ in range(budget):
                x = opt_den.ask()
                y = func(x)
                opt_den.tell(x, -y)
                den_sel.record_observation(x, -y)  # Register BEFORE tell
            results['density'][fname].append(-opt_den.best_y)
            
            # 4. Base Thompson (no bonus) - use different seed offset to ensure different results
            opt_base = ALBA(bounds=bounds, seed=seed + 10000, total_budget=budget, use_potential_field=False)
            for _ in range(budget):
                x = opt_base.ask()
                y = func(x)
                opt_base.tell(x, -y)
            results['base'][fname].append(-opt_base.best_y)
        
        gr_mean = np.mean(results['good_ratio'][fname])
        tpe_mean = np.mean(results['tpe'][fname])
        den_mean = np.mean(results['density'][fname])
        base_mean = np.mean(results['base'][fname])
        
        # Find winner (lower is better for minimization)
        means = {'good_ratio': gr_mean, 'tpe': tpe_mean, 'density': den_mean}
        best_name = min(means, key=means.get)
        best_val = means[best_name]
        
        # Check for ties (within 1%)
        second_best = sorted(means.values())[1]
        if abs(best_val - second_best) < 0.01 * abs(second_best + 1e-9):
            winner = "TIE"
        else:
            winner = best_name
            wins[best_name] += 1
        
        print(f"  good_ratio: {gr_mean:12.4f} ± {np.std(results['good_ratio'][fname]):.4f}")
        print(f"  TPE-KDE:    {tpe_mean:12.4f} ± {np.std(results['tpe'][fname]):.4f}")
        print(f"  Density:    {den_mean:12.4f} ± {np.std(results['density'][fname]):.4f}")
        print(f"  Base:       {base_mean:12.4f} ± {np.std(results['base'][fname]):.4f}")
        print(f"  → Winner: {winner.upper()}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"good_ratio wins: {wins['good_ratio']}")
    print(f"TPE-KDE wins:    {wins['tpe']}")
    print(f"Density wins:    {wins['density']}")
    
    total = sum(wins.values())
    best = max(wins, key=wins.get)
    print(f"\n→ BEST: {best.upper()} ({wins[best]}/{total} vittorie)")
    
    if best == 'density':
        print("\n✓ Density Count vince! Più semplice di TPE-KDE, stessa info globale.")
    elif best == 'good_ratio':
        print("\n✓ good_ratio vince! Info locale sufficiente, mantieni così.")
    else:
        print("\n✓ TPE-KDE vince! Se serve, implementa in coherence.py.")
    
    return results, wins


if __name__ == "__main__":
    results, wins = run_ab_test()
