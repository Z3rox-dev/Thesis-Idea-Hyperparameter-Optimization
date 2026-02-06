#!/usr/bin/env python3
"""
Ricerca ancora più approfondita: Perché in 10D Gaussian vince?

Ipotesi:
1. In alta dim, la Cov ha troppi pochi punti per stimare bene la matrice
2. Il condizionamento esplode e scatta sempre il fallback
3. La scala che funziona in 3D non funziona in 10D
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import GaussianLocalSearchSampler, CovarianceLocalSearchSampler
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

np.set_printoptions(precision=4, suppress=True)


# Funzioni
def sphere(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(x**2))

def rosenbrock(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def rastrigin(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))


# =============================================================================
# Sampler con diagnostica dettagliata sul PERCHÉ del fallback
# =============================================================================

@dataclass
class DetailedDiagnosticSampler:
    """Traccia ogni decisione."""
    
    base_scale: float = 3.0
    radius_start: float = 0.15
    radius_end: float = 0.01
    top_k_fraction: float = 0.15
    min_points_fit: int = 10
    
    # Diagnostica
    _call_count: int = field(default=0, init=False)
    _cov_used: int = field(default=0, init=False)
    _fallback_not_enough_points: int = field(default=0, init=False)
    _fallback_bad_condition: int = field(default=0, init=False)
    _fallback_exception: int = field(default=0, init=False)
    _conditions: List[float] = field(default_factory=list, init=False)
    _top_k_used: List[int] = field(default_factory=list, init=False)
    
    def get_stats(self):
        return {
            'calls': self._call_count,
            'cov_used': self._cov_used,
            'fallback_points': self._fallback_not_enough_points,
            'fallback_condition': self._fallback_bad_condition,
            'fallback_exception': self._fallback_exception,
            'avg_condition': np.mean(self._conditions) if self._conditions else 0,
            'max_condition': max(self._conditions) if self._conditions else 0,
            'avg_top_k': np.mean(self._top_k_used) if self._top_k_used else 0,
        }
    
    def sample(self, best_x, bounds, global_widths, progress, rng, X_history=None, y_history=None):
        self._call_count += 1
        dim = len(bounds)
        
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        progress = float(np.clip(progress if np.isfinite(progress) else 0.5, 0.0, 1.0))
        best_x = np.array(best_x, dtype=float)
        
        scale = self.radius_start * (1 - progress) + self.radius_end
        scale = max(scale, 1e-6)

        can_fit = False
        x_candidate = None

        if X_history is not None and y_history is not None:
            n = len(X_history)
            min_needed = max(self.min_points_fit, dim + 2)
            
            if n < min_needed:
                self._fallback_not_enough_points += 1
            else:
                k = max(min_needed, int(n * self.top_k_fraction))
                self._top_k_used.append(k)
                
                indices = np.argsort(y_history)[-k:][::-1]
                top_X = np.array([X_history[i] for i in indices])
                
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X, axis=0, weights=weights)
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                C += 1e-6 * np.eye(dim)
                
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    condition = eigvals.max() / max(eigvals.min(), 1e-10)
                    self._conditions.append(condition)
                    
                    if condition > 1000:
                        self._fallback_bad_condition += 1
                        can_fit = False
                    else:
                        can_fit = True
                except Exception:
                    self._fallback_exception += 1
                    can_fit = False

                if can_fit:
                    try:
                        z = rng.multivariate_normal(np.zeros(dim), C)
                        x = best_x + (z * scale * self.base_scale)
                        x_candidate = x
                        self._cov_used += 1
                    except Exception:
                        self._fallback_exception += 1
                        can_fit = False
        
        if not can_fit:
            noise = rng.normal(0, scale, dim) * global_widths
            x_candidate = best_x + noise

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)


def run_detailed(func, dim, sampler, seed, budget=100):
    bounds = [(-5.0, 10.0)] * dim
    opt = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
               local_search_sampler=sampler, use_drilling=False, seed=seed)
    for _ in range(budget):
        x = opt.ask()
        opt.tell(x, func(x))
    return opt.best_y, sampler.get_stats()


def analyze_why_10d_fails():
    """Perché in 10D il Cov non funziona bene?"""
    print("="*80)
    print("INDAGINE: Perché Cov non funziona in 10D?")
    print("="*80)
    
    n_seeds = 10
    
    for func, func_name in [(sphere, "Sphere"), (rosenbrock, "Rosenbrock")]:
        print(f"\n--- {func_name} ---")
        
        for dim in [3, 5, 10, 20]:
            all_stats = []
            ys = []
            
            for seed in range(n_seeds):
                sampler = DetailedDiagnosticSampler(base_scale=5.0)
                y, stats = run_detailed(func, dim, sampler, seed, budget=100)
                ys.append(y)
                all_stats.append(stats)
            
            avg_cov_used = np.mean([s['cov_used'] for s in all_stats])
            avg_fallback_cond = np.mean([s['fallback_condition'] for s in all_stats])
            avg_fallback_pts = np.mean([s['fallback_points'] for s in all_stats])
            avg_condition = np.mean([s['avg_condition'] for s in all_stats])
            max_condition = np.max([s['max_condition'] for s in all_stats])
            avg_top_k = np.mean([s['avg_top_k'] for s in all_stats])
            
            print(f"  Dim {dim:2d}: y={np.mean(ys):10.1f} | cov_used={avg_cov_used:5.1f} | "
                  f"fallback_cond={avg_fallback_cond:5.1f} | avg_cond={avg_condition:8.0f} | "
                  f"top_k={avg_top_k:4.1f}")


def analyze_top_k_impact():
    """Impatto del top_k_fraction sulla stima della covarianza."""
    print("\n" + "="*80)
    print("INDAGINE: Impatto di top_k_fraction")
    print("="*80)
    
    fractions = [0.10, 0.15, 0.20, 0.30, 0.50]
    n_seeds = 15
    
    for func, func_name in [(rosenbrock, "Rosenbrock")]:
        print(f"\n--- {func_name} ---")
        
        for dim in [3, 10]:
            print(f"\n  Dim = {dim}:")
            print(f"  {'Fraction':>10} | {'Mean y':>10} | {'Cov used':>10} | {'Avg Cond':>10}")
            print("  " + "-" * 50)
            
            for frac in fractions:
                ys = []
                cov_used_all = []
                cond_all = []
                
                for seed in range(n_seeds):
                    sampler = DetailedDiagnosticSampler(base_scale=5.0)
                    sampler.top_k_fraction = frac
                    y, stats = run_detailed(func, dim, sampler, seed)
                    ys.append(y)
                    cov_used_all.append(stats['cov_used'])
                    cond_all.append(stats['avg_condition'])
                
                print(f"  {frac:10.2f} | {np.mean(ys):10.1f} | {np.mean(cov_used_all):10.1f} | {np.mean(cond_all):10.0f}")


def analyze_min_points_impact():
    """Impatto del min_points_fit."""
    print("\n" + "="*80)
    print("INDAGINE: Impatto di min_points_fit")
    print("="*80)
    
    min_points_values = [5, 10, 15, 20, 30]
    n_seeds = 15
    
    for dim in [3, 10]:
        print(f"\n--- Dim = {dim} (Rosenbrock) ---")
        print(f"{'min_points':>12} | {'Mean y':>10} | {'Cov used':>10} | {'Fallback pts':>12}")
        print("-" * 55)
        
        for mp in min_points_values:
            ys = []
            cov_used_all = []
            fallback_pts_all = []
            
            for seed in range(n_seeds):
                sampler = DetailedDiagnosticSampler(base_scale=5.0)
                sampler.min_points_fit = mp
                y, stats = run_detailed(rosenbrock, dim, sampler, seed)
                ys.append(y)
                cov_used_all.append(stats['cov_used'])
                fallback_pts_all.append(stats['fallback_points'])
            
            print(f"{mp:12d} | {np.mean(ys):10.1f} | {np.mean(cov_used_all):10.1f} | {np.mean(fallback_pts_all):12.1f}")


def analyze_condition_threshold_impact():
    """Impatto della soglia di condizionamento."""
    print("\n" + "="*80)
    print("INDAGINE: Impatto soglia condizionamento")
    print("="*80)
    
    # La soglia attuale è 1000. Proviamo altre.
    thresholds = [100, 500, 1000, 5000, 10000, float('inf')]
    n_seeds = 15
    
    for dim in [3, 10]:
        print(f"\n--- Dim = {dim} (Rosenbrock) ---")
        print(f"{'Threshold':>12} | {'Mean y':>10} | {'Cov used':>10} | {'Fallback cond':>12}")
        print("-" * 55)
        
        for thresh in thresholds:
            ys = []
            cov_used_all = []
            fallback_cond_all = []
            
            for seed in range(n_seeds):
                sampler = DetailedDiagnosticSampler(base_scale=5.0)
                # Modifica temporaneamente la soglia
                original_sample = sampler.sample
                
                def patched_sample(best_x, bounds, global_widths, progress, rng, 
                                  X_history=None, y_history=None, threshold=thresh):
                    sampler._call_count += 1
                    dim_inner = len(bounds)
                    
                    if best_x is None:
                        return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

                    progress = float(np.clip(progress if np.isfinite(progress) else 0.5, 0.0, 1.0))
                    best_x = np.array(best_x, dtype=float)
                    
                    scale = sampler.radius_start * (1 - progress) + sampler.radius_end
                    scale = max(scale, 1e-6)

                    can_fit = False
                    x_candidate = None

                    if X_history is not None and y_history is not None:
                        n = len(X_history)
                        min_needed = max(sampler.min_points_fit, dim_inner + 2)
                        
                        if n < min_needed:
                            sampler._fallback_not_enough_points += 1
                        else:
                            k = max(min_needed, int(n * sampler.top_k_fraction))
                            sampler._top_k_used.append(k)
                            
                            indices = np.argsort(y_history)[-k:][::-1]
                            top_X = np.array([X_history[i] for i in indices])
                            
                            weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                            weights = weights / np.sum(weights)
                            
                            mu_w = np.average(top_X, axis=0, weights=weights)
                            centered = top_X - mu_w
                            C = np.dot((centered.T * weights), centered)
                            C += 1e-6 * np.eye(dim_inner)
                            
                            try:
                                eigvals = np.linalg.eigvalsh(C)
                                condition = eigvals.max() / max(eigvals.min(), 1e-10)
                                sampler._conditions.append(condition)
                                
                                if condition > threshold:  # Parametrico!
                                    sampler._fallback_bad_condition += 1
                                    can_fit = False
                                else:
                                    can_fit = True
                            except Exception:
                                sampler._fallback_exception += 1
                                can_fit = False

                            if can_fit:
                                try:
                                    z = rng.multivariate_normal(np.zeros(dim_inner), C)
                                    x = best_x + (z * scale * sampler.base_scale)
                                    x_candidate = x
                                    sampler._cov_used += 1
                                except Exception:
                                    sampler._fallback_exception += 1
                                    can_fit = False
                    
                    if not can_fit:
                        noise = rng.normal(0, scale, dim_inner) * global_widths
                        x_candidate = best_x + noise

                    return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim_inner)], dtype=float)
                
                sampler.sample = patched_sample
                
                bounds = [(-5.0, 10.0)] * dim
                opt = ALBA(bounds=bounds, total_budget=100, local_search_ratio=0.3,
                           local_search_sampler=sampler, use_drilling=False, seed=seed)
                for _ in range(100):
                    x = opt.ask()
                    opt.tell(x, rosenbrock(x))
                
                ys.append(opt.best_y)
                cov_used_all.append(sampler._cov_used)
                fallback_cond_all.append(sampler._fallback_bad_condition)
            
            thresh_str = f"{thresh:.0f}" if thresh != float('inf') else "inf"
            print(f"{thresh_str:>12} | {np.mean(ys):10.1f} | {np.mean(cov_used_all):10.1f} | {np.mean(fallback_cond_all):12.1f}")


def analyze_regularization_impact():
    """Impatto della regolarizzazione della matrice."""
    print("\n" + "="*80)
    print("INDAGINE: Impatto regolarizzazione (C += eps * I)")
    print("="*80)
    
    eps_values = [1e-8, 1e-6, 1e-4, 1e-2, 0.1]
    n_seeds = 15
    
    for dim in [3, 10]:
        print(f"\n--- Dim = {dim} (Rosenbrock) ---")
        print(f"{'Epsilon':>12} | {'Mean y':>10} | {'Avg Cond':>10}")
        print("-" * 40)
        
        for eps in eps_values:
            ys = []
            conds = []
            
            for seed in range(n_seeds):
                bounds = [(-5.0, 10.0)] * dim
                rng = np.random.default_rng(seed)
                
                # Simula una ottimizzazione semplificata
                X_history = [rng.uniform(-5, 10, dim) for _ in range(50)]
                y_history = [rosenbrock(x) for x in X_history]
                y_history = [-y for y in y_history]  # Converti in fitness
                
                # Calcola covarianza con regolarizzazione eps
                k = 10
                indices = np.argsort(y_history)[-k:][::-1]
                top_X = np.array([X_history[i] for i in indices])
                
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X, axis=0, weights=weights)
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                C += eps * np.eye(dim)
                
                eigvals = np.linalg.eigvalsh(C)
                condition = eigvals.max() / eigvals.min()
                conds.append(condition)
                
                # Run ALBA con questa regolarizzazione
                sampler = DetailedDiagnosticSampler(base_scale=5.0)
                y, _ = run_detailed(rosenbrock, dim, sampler, seed)
                ys.append(y)
            
            print(f"{eps:12.1e} | {np.mean(ys):10.1f} | {np.mean(conds):10.0f}")


def main():
    analyze_why_10d_fails()
    analyze_top_k_impact()
    analyze_min_points_impact()
    analyze_condition_threshold_impact()
    analyze_regularization_impact()
    
    print("\n" + "="*80)
    print("FINE INDAGINE APPROFONDITA")
    print("="*80)


if __name__ == "__main__":
    main()
