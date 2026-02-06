"""
Test A/B: good_ratio vs TPE ratio nel CoherenceTracker

Confrontiamo due strategie per calcolare la "qualità" di una foglia nel potential field:

1. good_ratio (attuale): (n_good + 1) / (n_trials + 2)
   - Locale: usa solo i dati della foglia
   - Empirico: conta la proporzione di buoni

2. TPE ratio: l(center) / g(center)
   - Globale: usa KDE su tutti i punti
   - Spaziale: considera la densità nel centro della foglia
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
from alba_framework_potential.leaf_selection import UCBSoftmaxLeafSelector, ThompsonSamplingLeafSelector


# ============================================================================
# TPE RATIO CALCULATOR
# ============================================================================

class TPERatioCalculator:
    """
    Calcola il TPE ratio per ogni foglia basandosi su KDE globale.
    
    TPE ratio = l(x) / g(x)
    dove:
    - l(x) = densità dei punti "buoni" (top γ%)
    - g(x) = densità dei punti "cattivi" (resto)
    """
    
    def __init__(self, gamma: float = 0.25, bandwidth: str = 'scott'):
        self.gamma = gamma
        self.bandwidth = bandwidth
        self._l_kde = None  # KDE per good points
        self._g_kde = None  # KDE per bad points
        self._is_fitted = False
    
    def fit(self, all_points: np.ndarray, all_scores: np.ndarray):
        """
        Costruisce i KDE dai punti osservati.
        
        Parameters
        ----------
        all_points : np.ndarray, shape (n_obs, dim)
            Tutti i punti valutati
        all_scores : np.ndarray, shape (n_obs,)
            Score di ogni punto (più alto = migliore per maximize)
        """
        if len(all_points) < 4:
            self._is_fitted = False
            return
        
        n = len(all_points)
        n_good = max(1, int(self.gamma * n))
        
        # Sort by score (descending - best first)
        idx_sorted = np.argsort(all_scores)[::-1]
        
        good_idx = idx_sorted[:n_good]
        bad_idx = idx_sorted[n_good:]
        
        good_points = all_points[good_idx]
        bad_points = all_points[bad_idx] if len(bad_idx) > 0 else all_points
        
        try:
            # Transpose for scipy KDE (expects shape (dim, n_samples))
            if good_points.shape[0] >= 2:
                self._l_kde = gaussian_kde(good_points.T, bw_method=self.bandwidth)
            else:
                self._l_kde = None
            
            if bad_points.shape[0] >= 2:
                self._g_kde = gaussian_kde(bad_points.T, bw_method=self.bandwidth)
            else:
                self._g_kde = None
            
            self._is_fitted = self._l_kde is not None
        except Exception:
            self._is_fitted = False
    
    def get_ratio(self, x: np.ndarray) -> float:
        """
        Calcola TPE ratio l(x) / g(x) per un punto.
        
        Returns
        -------
        ratio : float
            Alto = buona regione, basso = cattiva regione
        """
        if not self._is_fitted:
            return 0.5  # Neutral
        
        x = np.atleast_2d(x).T  # Shape (dim, 1)
        
        l_density = self._l_kde(x)[0] if self._l_kde is not None else 1e-9
        g_density = self._g_kde(x)[0] if self._g_kde is not None else 1e-9
        
        # Clip to avoid division issues
        l_density = max(l_density, 1e-12)
        g_density = max(g_density, 1e-12)
        
        ratio = l_density / g_density
        
        # Normalize to [0, 1] range roughly
        # log(ratio) in [-inf, inf], sigmoid maps to [0, 1]
        log_ratio = np.log(ratio + 1e-12)
        normalized = 1.0 / (1.0 + np.exp(-log_ratio))
        
        return float(normalized)
    
    def get_leaf_ratio(self, leaf: Cube) -> float:
        """Calcola TPE ratio per il centro di una foglia."""
        center = leaf.center()
        return self.get_ratio(center)


# ============================================================================
# MODIFIED COHERENCE TRACKER WITH TPE
# ============================================================================

def compute_potentials_with_tpe(leaves: List[Cube], tpe_calc: TPERatioCalculator) -> Dict[int, float]:
    """
    Calcola i potenziali usando TPE ratio invece di good_ratio.
    
    Il potenziale è invertito: alto TPE ratio = bassa promettenza = alto potenziale? No!
    Alto TPE ratio = alta densità di buoni = BASSA potenziale (buono)
    """
    n = len(leaves)
    if n == 0:
        return {}
    
    # Calcola TPE ratio per ogni foglia
    tpe_ratios = np.array([tpe_calc.get_leaf_ratio(leaf) for leaf in leaves])
    
    # Handle edge cases
    if not np.any(np.isfinite(tpe_ratios)):
        return {i: 0.5 for i in range(n)}
    
    tpe_ratios = np.nan_to_num(tpe_ratios, nan=0.5)
    tpe_ratios = np.clip(tpe_ratios, 0.0, 1.0)
    
    # Potenziale: alto TPE ratio → basso potenziale (buono)
    # Normalizza a [0, 1]
    if tpe_ratios.max() - tpe_ratios.min() > 1e-9:
        potentials = 1.0 - (tpe_ratios - tpe_ratios.min()) / (tpe_ratios.max() - tpe_ratios.min())
    else:
        potentials = np.full(n, 0.5)
    
    return {i: float(potentials[i]) for i in range(n)}


def compute_potentials_with_good_ratio(leaves: List[Cube]) -> Dict[int, float]:
    """Calcola potenziali usando good_ratio standard."""
    n = len(leaves)
    if n == 0:
        return {}
    
    ratios = np.array([leaf.good_ratio() for leaf in leaves])
    ratios = np.nan_to_num(ratios, nan=0.5)
    ratios = np.clip(ratios, 0.0, 1.0)
    
    # Alto good_ratio → basso potenziale
    if ratios.max() - ratios.min() > 1e-9:
        potentials = 1.0 - (ratios - ratios.min()) / (ratios.max() - ratios.min())
    else:
        potentials = np.full(n, 0.5)
    
    return {i: float(potentials[i]) for i in range(n)}


# ============================================================================
# CUSTOM LEAF SELECTORS
# ============================================================================

@dataclass
class TPEPotentialLeafSelector(ThompsonSamplingLeafSelector):
    """
    Thompson Sampling + TPE-based potential bonus.
    """
    potential_weight: float = 0.5
    tpe_calc: Any = None
    _all_points: List = field(default_factory=list)
    _all_scores: List = field(default_factory=list)
    
    def record_observation(self, x: np.ndarray, score: float):
        """Record observation for TPE fitting."""
        self._all_points.append(x.copy())
        self._all_scores.append(score)
        
        # Refit TPE periodically
        if len(self._all_points) >= 10 and len(self._all_points) % 5 == 0:
            if self.tpe_calc is None:
                self.tpe_calc = TPERatioCalculator()
            self.tpe_calc.fit(
                np.array(self._all_points),
                np.array(self._all_scores)
            )
    
    def select(self, leaves: list, dim: int, stagnating: bool, rng) -> "Cube":
        """Select leaf using Thompson + TPE potential."""
        if len(leaves) == 1:
            return leaves[0]

        scores = []
        for c in leaves:
            # Thompson Sampling base
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

            # TPE potential bonus
            tpe_bonus = 0.0
            if self.tpe_calc is not None and self.tpe_calc._is_fitted:
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
class GoodRatioPotentialLeafSelector(ThompsonSamplingLeafSelector):
    """
    Thompson Sampling + good_ratio-based potential bonus (baseline).
    """
    potential_weight: float = 0.5
    
    def select(self, leaves: list, dim: int, stagnating: bool, rng) -> "Cube":
        """Select leaf using Thompson + good_ratio potential."""
        if len(leaves) == 1:
            return leaves[0]

        # Compute good_ratio potentials
        potentials = compute_potentials_with_good_ratio(leaves)

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

            # Good ratio potential bonus (inverted: low potential = good)
            pot = potentials.get(i, 0.5)
            gr_bonus = self.potential_weight * (1.0 - pot)

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

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def rastrigin(x):
    return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)

def ackley(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return term1 + term2 + term3

def griewank(x):
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return sum_sq - prod_cos + 1


# ============================================================================
# A/B TEST
# ============================================================================

def run_ab_test():
    """
    Test A/B: Thompson+good_ratio vs Thompson+TPE nel leaf selection.
    """
    print("=" * 70)
    print("A/B TEST: good_ratio vs TPE ratio nel potential calculation")
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
    
    results = {
        'good_ratio': {},
        'tpe': {},
        'base_thompson': {}  # Thompson senza potential (controllo)
    }
    
    wins = {'good_ratio': 0, 'tpe': 0, 'tie': 0}
    
    for fname, func, dim, bounds in functions:
        print(f"\n{fname}:")
        results['good_ratio'][fname] = []
        results['tpe'][fname] = []
        results['base_thompson'][fname] = []
        
        for seed in seeds:
            # 1. Thompson + good_ratio potential
            gr_selector = GoodRatioPotentialLeafSelector()
            opt_gr = ALBA(
                bounds=bounds, 
                seed=seed, 
                total_budget=budget,
                use_potential_field=False,  # We handle it manually
                leaf_selector=gr_selector
            )
            for _ in range(budget):
                x = opt_gr.ask()
                y = func(x)
                opt_gr.tell(x, -y)  # Minimize
            results['good_ratio'][fname].append(-opt_gr.best_y)
            
            # 2. Thompson + TPE potential
            tpe_selector = TPEPotentialLeafSelector()
            opt_tpe = ALBA(
                bounds=bounds,
                seed=seed,
                total_budget=budget,
                use_potential_field=False,
                leaf_selector=tpe_selector
            )
            for _ in range(budget):
                x = opt_tpe.ask()
                y = func(x)
                opt_tpe.tell(x, -y)
                # Record for TPE
                tpe_selector.record_observation(x, -y)
            results['tpe'][fname].append(-opt_tpe.best_y)
            
            # 3. Base Thompson (no potential)
            opt_base = ALBA(
                bounds=bounds,
                seed=seed,
                total_budget=budget,
                use_potential_field=False
            )
            for _ in range(budget):
                x = opt_base.ask()
                y = func(x)
                opt_base.tell(x, -y)
            results['base_thompson'][fname].append(-opt_base.best_y)
        
        gr_mean = np.mean(results['good_ratio'][fname])
        tpe_mean = np.mean(results['tpe'][fname])
        base_mean = np.mean(results['base_thompson'][fname])
        
        # Winner between gr and tpe
        if abs(gr_mean - tpe_mean) < 0.01 * min(abs(gr_mean), abs(tpe_mean) + 1e-9):
            winner = "TIE"
            wins['tie'] += 1
        elif gr_mean < tpe_mean:
            winner = "good_ratio"
            wins['good_ratio'] += 1
        else:
            winner = "TPE"
            wins['tpe'] += 1
        
        print(f"  good_ratio: {gr_mean:.6f} ± {np.std(results['good_ratio'][fname]):.4f}")
        print(f"  TPE:        {tpe_mean:.6f} ± {np.std(results['tpe'][fname]):.4f}")
        print(f"  Base:       {base_mean:.6f} ± {np.std(results['base_thompson'][fname]):.4f}")
        print(f"  Winner: {winner}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"good_ratio wins: {wins['good_ratio']}")
    print(f"TPE wins:        {wins['tpe']}")
    print(f"Ties:            {wins['tie']}")
    
    total = wins['good_ratio'] + wins['tpe'] + wins['tie']
    if wins['good_ratio'] > wins['tpe']:
        print(f"\n→ MANTIENI good_ratio ({wins['good_ratio']}/{total} vittorie)")
    elif wins['tpe'] > wins['good_ratio']:
        print(f"\n→ USA TPE ({wins['tpe']}/{total} vittorie)")
    else:
        print(f"\n→ EQUIVALENTI - mantieni good_ratio (più semplice)")
    
    return results, wins


# ============================================================================
# ANALYSIS: Correlation with true optimum distance
# ============================================================================

def analyze_potential_quality():
    """
    Verifica quale metodo produce potenziali più correlati 
    con la distanza dall'ottimo.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Correlazione potenziale vs distanza dall'ottimo")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    dim = 5
    n_points = 200
    
    # Genera punti random
    points = rng.uniform(-5, 5, (n_points, dim))
    
    # Sphere con ottimo in 0
    scores = -np.sum(points**2, axis=1)  # Negativo per maximization
    
    # Distanza dall'ottimo
    distances = np.sqrt(np.sum(points**2, axis=1))
    
    # Simula foglie con questi punti
    # Dividi lo spazio in regioni e assegna punti
    n_leaves = 10
    
    # Creiamo foglie semplici (bins lungo prima dimensione)
    bins = np.linspace(-5, 5, n_leaves + 1)
    leaves = []
    leaf_points = {i: [] for i in range(n_leaves)}
    leaf_scores = {i: [] for i in range(n_leaves)}
    
    for i, (pt, sc) in enumerate(zip(points, scores)):
        bin_idx = min(n_leaves - 1, max(0, int((pt[0] + 5) / 10 * n_leaves)))
        leaf_points[bin_idx].append(pt)
        leaf_scores[bin_idx].append(sc)
    
    # Crea foglie mock
    gamma_threshold = np.percentile(scores, 75)  # Top 25%
    
    for i in range(n_leaves):
        c = Cube(np.full(dim, bins[i]), np.full(dim, bins[i+1]))
        c.n_trials = len(leaf_points[i])
        c.n_good = sum(1 for s in leaf_scores[i] if s >= gamma_threshold)
        leaves.append(c)
    
    # Calcola potenziali
    pot_gr = compute_potentials_with_good_ratio(leaves)
    
    tpe_calc = TPERatioCalculator()
    tpe_calc.fit(points, scores)
    pot_tpe = compute_potentials_with_tpe(leaves, tpe_calc)
    
    # Distanza media dall'ottimo per ogni foglia
    leaf_distances = []
    for i in range(n_leaves):
        if leaf_points[i]:
            avg_dist = np.mean([np.sqrt(np.sum(p**2)) for p in leaf_points[i]])
        else:
            avg_dist = 5.0  # Default
        leaf_distances.append(avg_dist)
    
    # Correlazione: potenziale basso dovrebbe correlare con distanza bassa
    pot_gr_arr = np.array([pot_gr[i] for i in range(n_leaves)])
    pot_tpe_arr = np.array([pot_tpe[i] for i in range(n_leaves)])
    
    corr_gr = np.corrcoef(pot_gr_arr, leaf_distances)[0, 1]
    corr_tpe = np.corrcoef(pot_tpe_arr, leaf_distances)[0, 1]
    
    print(f"\nCorrelazione potenziale vs distanza dall'ottimo:")
    print(f"  (Alta correlazione = potenziale alto per foglie lontane = BUONO)")
    print(f"  good_ratio: {corr_gr:.4f}")
    print(f"  TPE:        {corr_tpe:.4f}")
    
    if corr_gr > corr_tpe:
        print(f"\n→ good_ratio correla meglio con distanza")
    else:
        print(f"\n→ TPE correla meglio con distanza")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run A/B test
    results, wins = run_ab_test()
    
    # Analysis
    analyze_potential_quality()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONE")
    print("=" * 70)
    print("""
TPE ratio usa informazione GLOBALE (densità di tutti i punti)
good_ratio usa informazione LOCALE (solo punti nella foglia)

Se TPE vince: la densità spaziale è più informativa
Se good_ratio vince: la proporzione locale è sufficiente

Considera anche:
- TPE ha costo O(n) per KDE fitting
- good_ratio ha costo O(1)
""")
