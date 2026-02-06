#!/usr/bin/env python3
"""
Test: Scala Cov - è un tradeoff o una scala più alta aiuta sempre?

Confrontiamo diverse scale su diverse funzioni per capire:
1. Se una scala più alta aiuta sempre
2. Se c'è un tradeoff (alcune funzioni preferiscono scala bassa)
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import GaussianLocalSearchSampler
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CovarianceSamplerWithScale:
    """CovarianceLocalSearchSampler con scala configurabile."""
    
    scale_multiplier: float = 3.0  # Il parametro che testiamo
    radius_start: float = 0.15
    radius_end: float = 0.01
    top_k_fraction: float = 0.15
    min_points_fit: int = 10
    
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
        
        scale = self.radius_start * (1 - progress) + self.radius_end
        scale = max(scale, 1e-6)

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
                        # QUESTO È IL PARAMETRO CHE TESTIAMO
                        cov_scale = scale * self.scale_multiplier
                        x = best_x + (z * cov_scale)
                        x_candidate = x
                    except Exception:
                        can_fit = False
        
        if not can_fit:
            noise = rng.normal(0, scale, dim) * global_widths
            x_candidate = best_x + noise

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)


def run_test(func, func_name, scale_mult, seed, budget=100):
    """Esegue un singolo test."""
    bounds = [(-5.0, 10.0)] * 3
    
    sampler = CovarianceSamplerWithScale(scale_multiplier=scale_mult)
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


def main():
    print("="*80)
    print("TEST: Impatto della scala di Cov su diverse funzioni")
    print("="*80)
    
    functions = [
        (sphere, "Sphere"),
        (rosenbrock, "Rosenbrock"),
        (rastrigin, "Rastrigin"),
        (ackley, "Ackley"),
    ]
    
    # Scale da testare
    scales = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    n_seeds = 10
    
    results = {}
    
    for func, func_name in functions:
        print(f"\n--- {func_name} ---")
        results[func_name] = {}
        
        for scale in scales:
            ys = []
            for seed in range(n_seeds):
                y = run_test(func, func_name, scale, seed)
                ys.append(y)
            
            mean_y = np.mean(ys)
            std_y = np.std(ys)
            results[func_name][scale] = {'mean': mean_y, 'std': std_y}
            print(f"  scale={scale:4.1f}: mean={mean_y:10.2f} ± {std_y:8.2f}")
    
    # Trova la scala ottimale per ogni funzione
    print("\n" + "="*80)
    print("SCALA OTTIMALE PER FUNZIONE")
    print("="*80)
    
    for func_name in results:
        best_scale = min(results[func_name].keys(), 
                        key=lambda s: results[func_name][s]['mean'])
        best_mean = results[func_name][best_scale]['mean']
        
        # Anche la scala 3.0 (attuale)
        current_mean = results[func_name][3.0]['mean']
        
        print(f"{func_name:12s}: optimal={best_scale:.1f} (mean={best_mean:.2f}), "
              f"current=3.0 (mean={current_mean:.2f})")
    
    # Analisi tradeoff
    print("\n" + "="*80)
    print("ANALISI TRADEOFF")
    print("="*80)
    
    # Per ogni scala, calcola il rank medio su tutte le funzioni
    scale_ranks = {s: [] for s in scales}
    
    for func_name in results:
        # Rank le scale per questa funzione (1 = migliore)
        sorted_scales = sorted(scales, key=lambda s: results[func_name][s]['mean'])
        for rank, scale in enumerate(sorted_scales, 1):
            scale_ranks[scale].append(rank)
    
    print(f"\n{'Scale':>6} | {'Mean Rank':>10} | {'Note'}")
    print("-" * 40)
    
    for scale in scales:
        mean_rank = np.mean(scale_ranks[scale])
        note = "← MIGLIORE" if mean_rank == min(np.mean(r) for r in scale_ranks.values()) else ""
        print(f"{scale:6.1f} | {mean_rank:10.2f} | {note}")
    
    # Confronto con Gaussian
    print("\n" + "="*80)
    print("CONFRONTO CON GAUSSIAN (baseline)")
    print("="*80)
    
    for func, func_name in functions:
        gauss_ys = []
        for seed in range(n_seeds):
            bounds = [(-5.0, 10.0)] * 3
            opt = ALBA(
                bounds=bounds,
                total_budget=100,
                local_search_ratio=0.3,
                local_search_sampler=GaussianLocalSearchSampler(),
                use_drilling=False,
                seed=seed
            )
            for _ in range(100):
                x = opt.ask()
                y = func(x)
                opt.tell(x, y)
            gauss_ys.append(opt.best_y)
        
        gauss_mean = np.mean(gauss_ys)
        
        # Migliore Cov
        best_cov_scale = min(results[func_name].keys(), 
                            key=lambda s: results[func_name][s]['mean'])
        best_cov_mean = results[func_name][best_cov_scale]['mean']
        
        winner = "Cov" if best_cov_mean < gauss_mean else "Gauss"
        diff = gauss_mean - best_cov_mean
        
        print(f"{func_name:12s}: Gauss={gauss_mean:8.2f}, Cov(scale={best_cov_scale:.1f})={best_cov_mean:8.2f} → {winner} ({diff:+.2f})")


if __name__ == "__main__":
    main()
