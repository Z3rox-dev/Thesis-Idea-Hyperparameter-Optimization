#!/usr/bin/env python3
"""
Test: Confronto Covariance su TUTTE le dims vs SOLO CONTINUE.

Simula il fix proposto senza modificare il codice sorgente.
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import List, Tuple, Optional
from alba_framework_potential.optimizer import ALBA

# =============================================================================
# COVARIANCE SAMPLER MODIFICATO (solo continue)
# =============================================================================

class CovarianceContinuousOnlySampler:
    """
    Covariance sampler che opera SOLO sulle dimensioni continue.
    Le dimensioni categoriche vengono lasciate intatte (saranno gestite dal CategoricalSampler).
    """
    
    radius_start: float = 0.15
    radius_end: float = 0.01
    base_top_k_fraction: float = 0.15
    min_points_fit: int = 10
    scale_multiplier: float = 5.0
    base_regularization: float = 1e-2
    
    def __init__(self, categorical_dims: List[Tuple[int, int]] = None):
        self.categorical_dims = categorical_dims or []
        self.cat_indices = set(d for d, _ in self.categorical_dims)
    
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
        full_dim = len(bounds)
        
        # Identifica dimensioni continue
        cont_indices = [i for i in range(full_dim) if i not in self.cat_indices]
        n_cont = len(cont_indices)
        
        if best_x is None:
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)

        # Sanitize
        if not np.isfinite(progress):
            progress = 0.5
        progress = float(np.clip(progress, 0.0, 1.0))
        
        best_x = np.array(best_x, dtype=float)
        for i in range(full_dim):
            if not np.isfinite(best_x[i]):
                best_x[i] = (bounds[i][0] + bounds[i][1]) / 2
        
        scale = self.radius_start * (1 - progress) + self.radius_end
        scale = max(scale, 1e-6)

        # Output: inizia da best_x
        x_candidate = best_x.copy()
        
        can_fit = False

        if n_cont > 0 and X_history is not None and y_history is not None:
            n = len(X_history)
            min_needed = max(self.min_points_fit, n_cont + 2)
            
            if n >= min_needed:
                # Estrai SOLO le dimensioni continue
                adaptive_fraction = min(0.5, self.base_top_k_fraction + 0.02 * n_cont)
                k = max(min_needed, int(n * adaptive_fraction))
                
                indices = np.argsort(y_history)
                top_indices = indices[-k:][::-1]
                
                # Proietta su dimensioni continue
                top_X_cont = np.array([[X_history[i][j] for j in cont_indices] for i in top_indices])
                
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X_cont, axis=0, weights=weights)
                centered = top_X_cont - mu_w
                C = np.dot((centered.T * weights), centered)
                
                eps = self.base_regularization * (1 + 0.1 * n_cont)
                C += eps * np.eye(n_cont)
                
                try:
                    # Genera campione nello spazio continuo ridotto
                    z_cont = rng.multivariate_normal(np.zeros(n_cont), C)
                    
                    # Applica perturbazione SOLO alle dimensioni continue
                    best_x_cont = np.array([best_x[j] for j in cont_indices])
                    cov_scale = scale * self.scale_multiplier
                    x_cont_new = best_x_cont + (z_cont * cov_scale)
                    
                    # Scrivi nel vettore completo
                    for idx, j in enumerate(cont_indices):
                        x_candidate[j] = x_cont_new[idx]
                    
                    can_fit = True
                except Exception:
                    can_fit = False
        
        if not can_fit and n_cont > 0:
            # Fallback Gaussian solo per continue
            for j in cont_indices:
                noise = rng.normal(0, scale) * global_widths[j]
                x_candidate[j] = best_x[j] + noise
        
        # Clip alle bounds
        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) 
                        for i in range(full_dim)], dtype=float)


# =============================================================================
# FUNZIONI BENCHMARK (stesse di prima)
# =============================================================================

def nn_mixed_score(x: np.ndarray) -> float:
    """3 continue + 3 categoriche con interazioni."""
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    lr, wd, momentum = x[0], x[1], x[2]
    activation = discretize(x[3], 4)
    optimizer = discretize(x[4], 3)
    batch_size = discretize(x[5], 5)
    
    score = 0.0
    score -= 10 * (lr - 0.3)**2  
    score -= 5 * (wd - 0.1)**2
    score -= 8 * (momentum - 0.7)**2
    score -= 3 * (lr + momentum - 1.0)**2
    
    act_scores = [0.0, -0.1, 0.2, -0.05]
    score += act_scores[activation]
    opt_scores = [0.15, -0.2, 0.0]
    score += opt_scores[optimizer]
    batch_scores = [-0.1, 0.0, 0.15, 0.1, -0.05]
    score += batch_scores[batch_size]
    
    if activation == 2 and lr < 0.4:
        score += 0.1
    if optimizer == 0 and momentum > 0.6:
        score += 0.1
    
    return score + 1.0


def rosenbrock_with_categorical(x: np.ndarray) -> float:
    """4 continue + 2 categoriche."""
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    cont = [(x[i] - 0.5) * 4 for i in range(4)]
    rosen = sum(100 * (cont[i+1] - cont[i]**2)**2 + (1 - cont[i])**2 
                for i in range(3))
    
    cat1 = discretize(x[4], 4)
    cat2 = discretize(x[5], 3)
    
    cat_penalty = 0
    if cat1 != 1:
        cat_penalty += 10
    if cat2 != 0:
        cat_penalty += 5
    
    return -(rosen + cat_penalty)


def sphere_with_categorical(x: np.ndarray) -> float:
    """5 continue + 3 categoriche."""
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    cont = [(x[i] - 0.5) * 2 for i in range(5)]
    sphere = sum(c**2 for c in cont)
    
    cat1 = discretize(x[5], 3)
    cat2 = discretize(x[6], 4) 
    cat3 = discretize(x[7], 2)
    
    regime_penalty = 0
    if cat1 != 0:
        regime_penalty += 0.5
    if cat2 != 2:
        regime_penalty += 0.3
    if cat3 != 1:
        regime_penalty += 0.2
    
    return -(sphere + regime_penalty)


# =============================================================================
# IMPORTA IL SAMPLER ORIGINALE
# =============================================================================
from alba_framework_potential.local_search import CovarianceLocalSearchSampler


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(func, dim, budget, n_seeds, cat_dims, sampler_type: str):
    """
    sampler_type: 'all' = covariance su tutte le dims
                  'cont_only' = covariance solo su continue
    """
    results = []
    
    for seed in range(n_seeds):
        if sampler_type == 'all':
            sampler = CovarianceLocalSearchSampler()
        else:
            sampler = CovarianceContinuousOnlySampler(categorical_dims=cat_dims)
        
        opt = ALBA(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            maximize=True,
            seed=100 + seed,
            total_budget=budget,
            categorical_dims=cat_dims,
            use_potential_field=False,
        )
        
        opt._local_search_sampler = sampler
        
        for _ in range(budget):
            x = opt.ask()
            score = func(x)
            opt.tell(x, score)
        
        results.append(opt.best_y_internal)
    
    return results


def main():
    print("=" * 75)
    print("TEST: Covariance su TUTTE le dims vs SOLO CONTINUE")
    print("=" * 75)
    
    n_seeds = 20
    
    benchmarks = [
        (nn_mixed_score, "NN Mixed (3C+3K)", 6, 200, [(3, 4), (4, 3), (5, 5)]),
        (rosenbrock_with_categorical, "Rosen+Cat (4C+2K)", 6, 250, [(4, 4), (5, 3)]),
        (sphere_with_categorical, "Sphere+Cat (5C+3K)", 8, 200, [(5, 3), (6, 4), (7, 2)]),
    ]
    
    results_table = []
    
    print(f"\nEsecuzione con {n_seeds} seeds per configurazione...")
    print("-" * 75)
    
    for func, name, dim, budget, cat_dims in benchmarks:
        print(f"  Testing {name}...", end=" ", flush=True)
        
        # Originale: covariance su tutte le dims
        results_all = run_benchmark(func, dim, budget, n_seeds, cat_dims, 'all')
        
        # Fix: covariance solo su continue
        results_cont = run_benchmark(func, dim, budget, n_seeds, cat_dims, 'cont_only')
        
        mean_all = np.mean(results_all)
        mean_cont = np.mean(results_cont)
        
        wins_cont = sum(1 for a, b in zip(results_cont, results_all) if a > b)
        
        results_table.append({
            'name': name,
            'mean_all': mean_all,
            'mean_cont': mean_cont,
            'wins_cont': wins_cont,
        })
        
        print(f"done (ALL: {mean_all:.4f}, CONT_ONLY: {mean_cont:.4f})")
    
    print("\n" + "=" * 75)
    print("RISULTATI: Covariance ALL dims vs CONT_ONLY")
    print("=" * 75)
    
    print(f"\n{'Benchmark':<22} {'ALL dims':>12} {'CONT only':>12} {'Δ%':>8} {'Wins FIX':>10}")
    print("-" * 66)
    
    for r in results_table:
        delta = r['mean_cont'] - r['mean_all']
        delta_pct = delta / abs(r['mean_all']) * 100 if r['mean_all'] != 0 else 0
        status = "✅" if delta_pct > 1 else ("❌" if delta_pct < -1 else "➖")
        print(f"{r['name']:<22} {r['mean_all']:>12.4f} {r['mean_cont']:>12.4f} {delta_pct:>+7.1f}% {r['wins_cont']:>6}/{n_seeds} {status}")
    
    print("\n" + "=" * 75)
    print("ANALISI")
    print("=" * 75)
    
    total_wins_cont = sum(r['wins_cont'] for r in results_table)
    total_runs = len(results_table) * n_seeds
    
    print(f"\n  Wins CONT_ONLY (fix): {total_wins_cont}/{total_runs}")
    print(f"  Wins ALL dims (orig): {total_runs - total_wins_cont}/{total_runs}")
    
    avg_delta_pct = np.mean([(r['mean_cont'] - r['mean_all']) / abs(r['mean_all']) * 100 
                             if r['mean_all'] != 0 else 0 
                             for r in results_table])
    print(f"\n  Delta % medio: {avg_delta_pct:+.2f}%")
    
    print("\n" + "=" * 75)
    if total_wins_cont > total_runs * 0.55:
        print("🎉 VERDETTO: Il FIX (solo continue) MIGLIORA le performance!")
        print("   → Implementare nel codice sorgente")
    elif total_wins_cont < total_runs * 0.45:
        print("❌ VERDETTO: Il FIX (solo continue) PEGGIORA le performance!")
        print("   → NON implementare")
    else:
        print("➖ VERDETTO: Nessuna differenza significativa")
        print("   → Fix opzionale (riduce calcolo inutile)")
    print("=" * 75)


if __name__ == "__main__":
    main()
