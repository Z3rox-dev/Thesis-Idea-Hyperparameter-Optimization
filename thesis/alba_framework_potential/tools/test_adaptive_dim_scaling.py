#!/usr/bin/env python3
"""
Test: Scala adattiva e dipendente dalla dimensione

Approcci da testare:
1. Scala fissa (baseline)
2. Scala adattiva - aumenta se stagnazione
3. Scala dipendente dalla dim
4. Combinazione di entrambi
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import GaussianLocalSearchSampler
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

np.set_printoptions(precision=4, suppress=True)

# Funzioni di test
def sphere(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(x**2))

def rosenbrock(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def rastrigin(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def ackley(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    n = len(x)
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e)


@dataclass
class AdaptiveCovSampler:
    """Sampler con scala adattiva e/o dipendente dalla dim."""
    
    # Strategia scala
    base_scale: float = 3.0
    use_adaptive: bool = False
    use_dim_scaling: bool = False
    
    # Parametri adattivi
    stagnation_threshold: int = 5
    scale_increase_factor: float = 1.5
    scale_max: float = 20.0
    
    # Parametri dim scaling
    dim_scale_formula: str = "sqrt"  # "sqrt", "linear", "log"
    
    # Stato interno (mutable)
    _current_scale: float = field(default=3.0, init=False)
    _stagnation_count: int = field(default=0, init=False)
    _last_best_y: float = field(default=float('inf'), init=False)
    _call_count: int = field(default=0, init=False)
    _scale_history: List[float] = field(default_factory=list, init=False)
    
    # Altri parametri
    radius_start: float = 0.15
    radius_end: float = 0.01
    top_k_fraction: float = 0.15
    min_points_fit: int = 10
    
    def __post_init__(self):
        self._current_scale = self.base_scale
        self._scale_history = []
    
    def _compute_dim_scale(self, dim: int) -> float:
        """Calcola scala basata sulla dimensione."""
        if self.dim_scale_formula == "sqrt":
            # In basse dim serve più esplorazione
            return self.base_scale * (3.0 / np.sqrt(dim))
        elif self.dim_scale_formula == "linear":
            return self.base_scale * (10.0 / dim)
        elif self.dim_scale_formula == "log":
            return self.base_scale * (2.0 / np.log(dim + 1))
        elif self.dim_scale_formula == "inverse":
            # Più piccola la dim, più grande la scala
            return self.base_scale * (5.0 / dim)
        else:
            return self.base_scale
    
    def _update_adaptive_scale(self, y_history: List[float]):
        """Aggiorna scala basandosi su stagnazione."""
        if not y_history:
            return
        
        # Trova il miglior y (più alto = migliore, è fitness)
        current_best = max(y_history)
        
        if current_best > self._last_best_y + 1e-6:
            # Miglioramento! Reset stagnazione, riduci scala
            self._stagnation_count = 0
            self._current_scale = max(self.base_scale, self._current_scale * 0.9)
            self._last_best_y = current_best
        else:
            # Stagnazione
            self._stagnation_count += 1
            
            if self._stagnation_count >= self.stagnation_threshold:
                # Aumenta scala per esplorare di più
                self._current_scale = min(
                    self._current_scale * self.scale_increase_factor,
                    self.scale_max
                )
                self._stagnation_count = 0  # Reset per non aumentare troppo
    
    def sample(
        self,
        best_x: Optional[np.ndarray],
        bounds: List[Tuple[float, float]],
        global_widths: np.ndarray,
        progress: float,
        rng: np.random.Generator,
        X_history: Optional[List[np.ndarray]] = None,
        y_history: Optional[List[float]] = None,
    ) -> np.ndarray:
        self._call_count += 1
        dim = len(bounds)
        
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        if not np.isfinite(progress):
            progress = 0.5
        progress = float(np.clip(progress, 0.0, 1.0))
        
        best_x = np.array(best_x, dtype=float)
        for i in range(dim):
            if not np.isfinite(best_x[i]):
                best_x[i] = (bounds[i][0] + bounds[i][1]) / 2
        
        # Calcola scala
        if self.use_adaptive and y_history:
            self._update_adaptive_scale(y_history)
        
        if self.use_dim_scaling:
            dim_scale = self._compute_dim_scale(dim)
        else:
            dim_scale = self.base_scale
        
        # Scala finale: combina dim_scale con adattivo
        if self.use_adaptive:
            final_scale = self._current_scale
        else:
            final_scale = dim_scale
        
        self._scale_history.append(final_scale)
        
        # Progress decay
        base_radius = self.radius_start * (1 - progress) + self.radius_end
        base_radius = max(base_radius, 1e-6)

        can_fit = False
        x_candidate = None

        if X_history is not None and y_history is not None:
            n = len(X_history)
            min_needed = max(self.min_points_fit, dim + 2)
            
            if n >= min_needed:
                k = max(min_needed, int(n * self.top_k_fraction))
                indices = np.argsort(y_history)
                top_indices = indices[-k:][::-1]
                
                top_X = np.array([X_history[i] for i in top_indices])
                
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X, axis=0, weights=weights)
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                C += 1e-6 * np.eye(dim)
                
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    condition = eigvals.max() / max(eigvals.min(), 1e-10)
                    if condition > 1000:
                        can_fit = False
                    else:
                        can_fit = True
                except Exception:
                    can_fit = False

                if can_fit:
                    try:
                        z = rng.multivariate_normal(np.zeros(dim), C)
                        cov_scale = base_radius * final_scale
                        x = best_x + (z * cov_scale)
                        x_candidate = x
                    except Exception:
                        can_fit = False
        
        if not can_fit:
            noise = rng.normal(0, base_radius, dim) * global_widths
            x_candidate = best_x + noise

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)


def run_test(func, dim, sampler, seed, budget=100):
    """Esegue un singolo test."""
    bounds = [(-5.0, 10.0)] * dim
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        use_drilling=False,
        seed=seed
    )
    
    for _ in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
    
    return opt.best_y


def test_dim_scaling():
    """Test scala dipendente dalla dimensione."""
    print("="*80)
    print("TEST 1: Scala dipendente dalla dimensione")
    print("="*80)
    
    functions = [
        (sphere, "Sphere"),
        (rosenbrock, "Rosenbrock"),
        (rastrigin, "Rastrigin"),
    ]
    
    dims = [3, 5, 10, 20]
    formulas = ["none", "sqrt", "inverse", "linear"]
    n_seeds = 5
    
    for func, func_name in functions:
        print(f"\n--- {func_name} ---")
        print(f"{'Dim':>4} | " + " | ".join(f"{f:>10}" for f in formulas))
        print("-" * (6 + 13 * len(formulas)))
        
        for dim in dims:
            row = f"{dim:4d} |"
            for formula in formulas:
                ys = []
                for seed in range(n_seeds):
                    if formula == "none":
                        sampler = AdaptiveCovSampler(base_scale=3.0, use_dim_scaling=False)
                    else:
                        sampler = AdaptiveCovSampler(base_scale=3.0, use_dim_scaling=True, 
                                                     dim_scale_formula=formula)
                    y = run_test(func, dim, sampler, seed, budget=100)
                    ys.append(y)
                mean_y = np.mean(ys)
                row += f" {mean_y:10.2f} |"
            print(row)


def test_adaptive_scaling():
    """Test scala adattiva."""
    print("\n" + "="*80)
    print("TEST 2: Scala adattiva (aumenta con stagnazione)")
    print("="*80)
    
    functions = [
        (sphere, "Sphere"),
        (rosenbrock, "Rosenbrock"),
        (rastrigin, "Rastrigin"),
    ]
    
    configs = [
        ("Fixed 3.0", {"base_scale": 3.0, "use_adaptive": False}),
        ("Fixed 8.0", {"base_scale": 8.0, "use_adaptive": False}),
        ("Adaptive base=3", {"base_scale": 3.0, "use_adaptive": True, "stagnation_threshold": 3}),
        ("Adaptive base=3 slow", {"base_scale": 3.0, "use_adaptive": True, "stagnation_threshold": 5}),
        ("Adaptive base=5", {"base_scale": 5.0, "use_adaptive": True, "stagnation_threshold": 3}),
    ]
    
    dims = [3, 10]
    n_seeds = 10
    
    for func, func_name in functions:
        print(f"\n--- {func_name} ---")
        
        for dim in dims:
            print(f"\n  Dim={dim}:")
            print(f"  {'Config':>20} | {'Mean':>10} | {'Std':>8} | {'Scale history (last 5 calls)'}")
            print("  " + "-" * 70)
            
            for config_name, config_params in configs:
                ys = []
                all_scales = []
                for seed in range(n_seeds):
                    sampler = AdaptiveCovSampler(**config_params)
                    y = run_test(func, dim, sampler, seed, budget=100)
                    ys.append(y)
                    if sampler._scale_history:
                        all_scales.append(sampler._scale_history[-5:])
                
                mean_y = np.mean(ys)
                std_y = np.std(ys)
                
                # Media delle scale finali
                if all_scales:
                    avg_final_scales = np.mean([s[-1] if s else 3.0 for s in all_scales])
                    scale_info = f"avg_final={avg_final_scales:.1f}"
                else:
                    scale_info = "N/A"
                
                print(f"  {config_name:>20} | {mean_y:10.2f} | {std_y:8.2f} | {scale_info}")


def test_adaptive_detailed_trace():
    """Trace dettagliato di una run adattiva."""
    print("\n" + "="*80)
    print("TEST 3: Trace dettagliato scala adattiva su Rosenbrock 3D")
    print("="*80)
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    seed = 3  # Il seed problematico
    
    sampler = AdaptiveCovSampler(
        base_scale=3.0, 
        use_adaptive=True, 
        stagnation_threshold=3,
        scale_increase_factor=1.5,
        scale_max=20.0
    )
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        use_drilling=False,
        seed=seed
    )
    
    history = []
    for i in range(budget):
        x = opt.ask()
        y = rosenbrock(x)
        opt.tell(x, y)
        
        current_scale = sampler._scale_history[-1] if sampler._scale_history else 3.0
        history.append({
            'iter': i,
            'y': y,
            'best_y': opt.best_y,
            'scale': current_scale,
            'stagnation': sampler._stagnation_count
        })
    
    print(f"\n{'Iter':>4} | {'y':>10} | {'best_y':>10} | {'scale':>6} | {'stag':>4}")
    print("-" * 50)
    
    for h in history[::5]:  # Ogni 5 iterazioni
        print(f"{h['iter']:4d} | {h['y']:10.2f} | {h['best_y']:10.2f} | {h['scale']:6.1f} | {h['stagnation']:4d}")
    
    print(f"\nFinale: best_y = {opt.best_y:.2f}")
    print(f"Scala finale: {sampler._current_scale:.1f}")
    print(f"Range scale durante run: {min(sampler._scale_history):.1f} - {max(sampler._scale_history):.1f}")
    
    # Confronta con fixed
    sampler_fixed = AdaptiveCovSampler(base_scale=3.0, use_adaptive=False)
    opt_fixed = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                     local_search_sampler=sampler_fixed, use_drilling=False, seed=seed)
    for _ in range(budget):
        x = opt_fixed.ask()
        opt_fixed.tell(x, rosenbrock(x))
    
    print(f"\nConfronto seed {seed}:")
    print(f"  Fixed scale=3.0: {opt_fixed.best_y:.2f}")
    print(f"  Adaptive:        {opt.best_y:.2f}")


def test_combined_strategies():
    """Test combinazione di strategie."""
    print("\n" + "="*80)
    print("TEST 4: Combinazione dim_scaling + adaptive")
    print("="*80)
    
    functions = [
        (sphere, "Sphere"),
        (rosenbrock, "Rosenbrock"),
        (rastrigin, "Rastrigin"),
    ]
    
    configs = [
        ("Fixed 3.0", {"base_scale": 3.0}),
        ("Dim sqrt", {"base_scale": 3.0, "use_dim_scaling": True, "dim_scale_formula": "sqrt"}),
        ("Adaptive", {"base_scale": 3.0, "use_adaptive": True}),
        ("Dim sqrt + Adaptive", {"base_scale": 3.0, "use_dim_scaling": True, "dim_scale_formula": "sqrt", 
                                  "use_adaptive": True}),
    ]
    
    n_seeds = 10
    
    for dim in [3, 10]:
        print(f"\n--- Dim = {dim} ---")
        print(f"{'Config':>25} | " + " | ".join(f"{f[1]:>12}" for f in functions))
        print("-" * (28 + 15 * len(functions)))
        
        for config_name, config_params in configs:
            row = f"{config_name:>25} |"
            for func, _ in functions:
                ys = []
                for seed in range(n_seeds):
                    sampler = AdaptiveCovSampler(**config_params)
                    y = run_test(func, dim, sampler, seed, budget=100)
                    ys.append(y)
                mean_y = np.mean(ys)
                row += f" {mean_y:12.2f} |"
            print(row)


def main():
    test_dim_scaling()
    test_adaptive_scaling()
    test_adaptive_detailed_trace()
    test_combined_strategies()


if __name__ == "__main__":
    main()
