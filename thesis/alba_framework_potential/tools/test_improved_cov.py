#!/usr/bin/env python3
"""
Test correzioni basate sulle scoperte:

1. top_k_fraction più alta (0.3-0.5) per ridurre condizionamento
2. Regolarizzazione proporzionale alla dimensione
3. Fallback adattivo: usare più Cov ma con regolarizzazione aggressiva
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from dataclasses import dataclass, field


def rosenbrock(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))

def sphere(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(x**2))


@dataclass
class ImprovedCovSampler:
    """
    Sampler con miglioramenti basati sull'indagine:
    1. top_k_fraction adattivo: usa più punti in alta dim
    2. Regolarizzazione proporzionale alla dimensione
    3. Nessun check di condizionamento (regolarizzazione risolve tutto)
    """
    
    base_scale: float = 5.0
    radius_start: float = 0.15
    radius_end: float = 0.01
    
    # Nuovo: fraction adattiva
    base_top_k_fraction: float = 0.15
    use_adaptive_fraction: bool = True
    
    # Nuovo: regolarizzazione proporzionale
    use_dim_regularization: bool = True
    base_epsilon: float = 1e-2  # Epsilon base
    
    def sample(self, best_x, bounds, global_widths, progress, rng, X_history=None, y_history=None):
        dim = len(bounds)
        
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        progress = float(np.clip(progress if np.isfinite(progress) else 0.5, 0.0, 1.0))
        best_x = np.array(best_x, dtype=float)
        
        scale = self.radius_start * (1 - progress) + self.radius_end
        scale = max(scale, 1e-6)

        x_candidate = None

        if X_history is not None and y_history is not None:
            n = len(X_history)
            
            # Adattivo: usa più punti in alta dimensione
            if self.use_adaptive_fraction:
                # In alta dim servono più punti per stimare la covarianza
                # Formula: fraction = min(0.5, base + 0.02 * dim)
                fraction = min(0.5, self.base_top_k_fraction + 0.02 * dim)
            else:
                fraction = self.base_top_k_fraction
            
            min_needed = max(10, dim + 2)
            k = max(min_needed, int(n * fraction))
            
            if n >= min_needed:
                indices = np.argsort(y_history)[-k:][::-1]
                top_X = np.array([X_history[i] for i in indices])
                
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X, axis=0, weights=weights)
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                
                # Regolarizzazione proporzionale alla dimensione
                if self.use_dim_regularization:
                    # Epsilon cresce con dim^2 per compensare l'aumento del condizionamento
                    eps = self.base_epsilon * (1 + 0.1 * dim)
                else:
                    eps = 1e-6
                
                C += eps * np.eye(dim)
                
                try:
                    z = rng.multivariate_normal(np.zeros(dim), C)
                    x_candidate = best_x + (z * scale * self.base_scale)
                except:
                    # Fallback Gaussiano solo su eccezione
                    noise = rng.normal(0, scale, dim) * global_widths
                    x_candidate = best_x + noise
            else:
                noise = rng.normal(0, scale, dim) * global_widths
                x_candidate = best_x + noise
        else:
            noise = rng.normal(0, scale, dim) * global_widths
            x_candidate = best_x + noise

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)


@dataclass
class GaussianSampler:
    radius_start: float = 0.15
    radius_end: float = 0.01
    
    def sample(self, best_x, bounds, global_widths, progress, rng, X_history=None, y_history=None):
        dim = len(bounds)
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
        
        progress = float(np.clip(progress if np.isfinite(progress) else 0.5, 0.0, 1.0))
        best_x = np.array(best_x, dtype=float)
        scale = self.radius_start * (1 - progress) + self.radius_end
        
        noise = rng.normal(0, scale, dim) * global_widths
        x_candidate = best_x + noise
        
        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)


def run_test(func, dim, sampler, seed, budget=100):
    bounds = [(-5.0, 10.0)] * dim
    opt = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
               local_search_sampler=sampler, use_drilling=False, seed=seed)
    for _ in range(budget):
        x = opt.ask()
        opt.tell(x, func(x))
    return opt.best_y


def main():
    n_seeds = 20
    
    print("="*80)
    print("TEST: Sampler Cov Migliorato vs Gaussiano")
    print("="*80)
    
    configs = [
        ("Gaussian", lambda: GaussianSampler()),
        ("Cov_orig (scale=3)", lambda: ImprovedCovSampler(base_scale=3.0, use_adaptive_fraction=False, use_dim_regularization=False)),
        ("Cov_orig (scale=5)", lambda: ImprovedCovSampler(base_scale=5.0, use_adaptive_fraction=False, use_dim_regularization=False)),
        ("Cov_adaptive_frac", lambda: ImprovedCovSampler(base_scale=5.0, use_adaptive_fraction=True, use_dim_regularization=False)),
        ("Cov_dim_reg", lambda: ImprovedCovSampler(base_scale=5.0, use_adaptive_fraction=False, use_dim_regularization=True)),
        ("Cov_full (adaptive+reg)", lambda: ImprovedCovSampler(base_scale=5.0, use_adaptive_fraction=True, use_dim_regularization=True)),
    ]
    
    for func, func_name in [(rosenbrock, "Rosenbrock"), (sphere, "Sphere")]:
        print(f"\n{'='*60}")
        print(f"Funzione: {func_name}")
        print(f"{'='*60}")
        
        for dim in [3, 5, 10]:
            print(f"\n--- Dim = {dim} ---")
            print(f"{'Config':<25} | {'Mean':>10} | {'Std':>10} | {'Median':>10} | {'P90':>10}")
            print("-" * 75)
            
            for cfg_name, sampler_factory in configs:
                ys = []
                for seed in range(n_seeds):
                    y = run_test(func, dim, sampler_factory(), seed)
                    ys.append(y)
                
                ys = np.array(ys)
                print(f"{cfg_name:<25} | {np.mean(ys):10.1f} | {np.std(ys):10.1f} | "
                      f"{np.median(ys):10.1f} | {np.percentile(ys, 90):10.1f}")
    
    # Head-to-head
    print("\n" + "="*80)
    print("HEAD-TO-HEAD: Quante volte vince ciascun config (Rosenbrock)?")
    print("="*80)
    
    for dim in [3, 5, 10]:
        print(f"\n--- Dim = {dim} ---")
        
        results_per_seed = {cfg[0]: [] for cfg in configs}
        
        for seed in range(n_seeds):
            for cfg_name, sampler_factory in configs:
                y = run_test(rosenbrock, dim, sampler_factory(), seed)
                results_per_seed[cfg_name].append(y)
        
        # Conta vittorie
        wins = {cfg[0]: 0 for cfg in configs}
        for seed in range(n_seeds):
            best_cfg = min(results_per_seed.keys(), key=lambda c: results_per_seed[c][seed])
            wins[best_cfg] += 1
        
        for cfg_name, win_count in sorted(wins.items(), key=lambda x: -x[1]):
            print(f"  {cfg_name:<25}: {win_count}/{n_seeds} vittorie")
    
    # Test specifico con regolarizzazioni più aggressive
    print("\n" + "="*80)
    print("TEST: Vari livelli di regolarizzazione in 10D")
    print("="*80)
    
    dim = 10
    eps_values = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2]
    
    print(f"\n{'Epsilon':<12} | {'Mean':>10} | {'Std':>10} | {'Median':>10}")
    print("-" * 50)
    
    for eps in eps_values:
        ys = []
        for seed in range(n_seeds):
            sampler = ImprovedCovSampler(base_scale=5.0, base_epsilon=eps,
                                         use_adaptive_fraction=True, use_dim_regularization=True)
            y = run_test(rosenbrock, dim, sampler, seed)
            ys.append(y)
        
        ys = np.array(ys)
        print(f"{eps:<12.1e} | {np.mean(ys):10.1f} | {np.std(ys):10.1f} | {np.median(ys):10.1f}")
    
    # Gaussian per confronto
    ys_gauss = [run_test(rosenbrock, dim, GaussianSampler(), seed) for seed in range(n_seeds)]
    print(f"{'Gaussian':<12} | {np.mean(ys_gauss):10.1f} | {np.std(ys_gauss):10.1f} | {np.median(ys_gauss):10.1f}")


if __name__ == "__main__":
    main()
