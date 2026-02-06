#!/usr/bin/env python3
"""
Ricerca approfondita: Scala Cov - analisi sistematica

Domande da rispondere:
1. Perché l'adattivo a volte peggiora? (es. Rosenbrock 3D: 26.79 vs 22.88)
2. Quando e perché dim_scaling non funziona in alta dim?
3. Qual è l'interazione tra stagnation_threshold e scale_increase_factor?
4. I risultati sono robusti su più seed?
5. Cosa succede su funzioni anisotropiche (Cigar, Discus)?
6. C'è correlazione tra condizionamento della Cov e scala ottimale?
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import GaussianLocalSearchSampler
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# FUNZIONI DI TEST (incluse anisotropiche)
# =============================================================================

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

def cigar(x):
    """Anisotropic: first dimension much more important."""
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(x[0]**2 + 1e6 * np.sum(x[1:]**2))

def discus(x):
    """Anisotropic: first dimension much less important."""
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(1e6 * x[0]**2 + np.sum(x[1:]**2))

def ellipsoid(x):
    """Gradually anisotropic."""
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    n = len(x)
    weights = 10 ** (6 * np.arange(n) / (n - 1)) if n > 1 else np.array([1.0])
    return float(np.sum(weights * x**2))


# =============================================================================
# SAMPLER CON DIAGNOSTICA DETTAGLIATA
# =============================================================================

@dataclass
class DiagnosticCovSampler:
    """Sampler con diagnostica completa."""
    
    base_scale: float = 3.0
    use_adaptive: bool = False
    stagnation_threshold: int = 5
    scale_increase_factor: float = 1.5
    scale_max: float = 20.0
    
    radius_start: float = 0.15
    radius_end: float = 0.01
    top_k_fraction: float = 0.15
    min_points_fit: int = 10
    
    # Stato
    _current_scale: float = field(default=3.0, init=False)
    _stagnation_count: int = field(default=0, init=False)
    _last_best_y: float = field(default=float('-inf'), init=False)
    
    # Diagnostica
    _scale_history: List[float] = field(default_factory=list, init=False)
    _cov_used_count: int = field(default=0, init=False)
    _fallback_count: int = field(default=0, init=False)
    _condition_history: List[float] = field(default_factory=list, init=False)
    _improvement_iters: List[int] = field(default_factory=list, init=False)
    _call_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        self._current_scale = self.base_scale
    
    def get_diagnostics(self):
        return {
            'cov_used': self._cov_used_count,
            'fallback': self._fallback_count,
            'final_scale': self._current_scale,
            'scale_range': (min(self._scale_history) if self._scale_history else self.base_scale,
                           max(self._scale_history) if self._scale_history else self.base_scale),
            'avg_condition': np.mean(self._condition_history) if self._condition_history else 0,
            'improvements': len(self._improvement_iters),
        }
    
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
        
        # Adaptive update
        if self.use_adaptive and y_history:
            current_best = max(y_history)
            if current_best > self._last_best_y + 1e-8:
                self._stagnation_count = 0
                self._current_scale = max(self.base_scale, self._current_scale * 0.9)
                self._last_best_y = current_best
                self._improvement_iters.append(self._call_count)
            else:
                self._stagnation_count += 1
                if self._stagnation_count >= self.stagnation_threshold:
                    self._current_scale = min(self._current_scale * self.scale_increase_factor, self.scale_max)
                    self._stagnation_count = 0
        
        self._scale_history.append(self._current_scale)
        
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
                    self._condition_history.append(condition)
                    
                    if condition > 1000:
                        can_fit = False
                    else:
                        can_fit = True
                except Exception:
                    can_fit = False

                if can_fit:
                    try:
                        z = rng.multivariate_normal(np.zeros(dim), C)
                        cov_scale = base_radius * self._current_scale
                        x = best_x + (z * cov_scale)
                        x_candidate = x
                        self._cov_used_count += 1
                    except Exception:
                        can_fit = False
        
        if not can_fit:
            noise = rng.normal(0, base_radius, dim) * global_widths
            x_candidate = best_x + noise
            self._fallback_count += 1

        return np.array([np.clip(x_candidate[i], bounds[i][0], bounds[i][1]) for i in range(dim)], dtype=float)


def run_with_diagnostics(func, dim, sampler, seed, budget=100):
    """Esegue con diagnostica."""
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
    
    return opt.best_y, sampler.get_diagnostics()


# =============================================================================
# ANALISI 1: Perché l'adattivo a volte peggiora?
# =============================================================================

def analyze_adaptive_failures():
    """Analizza casi dove l'adattivo peggiora."""
    print("="*80)
    print("ANALISI 1: Quando l'adattivo peggiora?")
    print("="*80)
    
    n_seeds = 30
    
    for func, func_name in [(rosenbrock, "Rosenbrock"), (rastrigin, "Rastrigin")]:
        print(f"\n--- {func_name} 3D ---")
        
        fixed_results = []
        adaptive_results = []
        
        for seed in range(n_seeds):
            # Fixed
            sampler_f = DiagnosticCovSampler(base_scale=3.0, use_adaptive=False)
            y_f, diag_f = run_with_diagnostics(func, 3, sampler_f, seed)
            fixed_results.append({'seed': seed, 'y': y_f, 'diag': diag_f})
            
            # Adaptive
            sampler_a = DiagnosticCovSampler(base_scale=3.0, use_adaptive=True, stagnation_threshold=3)
            y_a, diag_a = run_with_diagnostics(func, 3, sampler_a, seed)
            adaptive_results.append({'seed': seed, 'y': y_a, 'diag': diag_a})
        
        # Trova casi dove adattivo peggiora
        worse_cases = []
        better_cases = []
        for i in range(n_seeds):
            diff = adaptive_results[i]['y'] - fixed_results[i]['y']
            if diff > 1.0:  # Adattivo peggio
                worse_cases.append((i, fixed_results[i]['y'], adaptive_results[i]['y'], 
                                   adaptive_results[i]['diag']))
            elif diff < -1.0:  # Adattivo meglio
                better_cases.append((i, fixed_results[i]['y'], adaptive_results[i]['y'],
                                    adaptive_results[i]['diag']))
        
        print(f"  Adattivo PEGGIO in {len(worse_cases)}/{n_seeds} casi")
        print(f"  Adattivo MEGLIO in {len(better_cases)}/{n_seeds} casi")
        
        if worse_cases:
            print(f"\n  Casi dove adattivo peggiora:")
            for seed, y_f, y_a, diag in worse_cases[:5]:
                print(f"    Seed {seed}: fixed={y_f:.1f}, adaptive={y_a:.1f}, "
                      f"final_scale={diag['final_scale']:.1f}, improvements={diag['improvements']}")
        
        if better_cases:
            print(f"\n  Casi dove adattivo migliora:")
            for seed, y_f, y_a, diag in better_cases[:5]:
                print(f"    Seed {seed}: fixed={y_f:.1f}, adaptive={y_a:.1f}, "
                      f"final_scale={diag['final_scale']:.1f}, improvements={diag['improvements']}")


# =============================================================================
# ANALISI 2: Impatto dei parametri adattivi
# =============================================================================

def analyze_adaptive_parameters():
    """Analizza impatto di stagnation_threshold e scale_increase_factor."""
    print("\n" + "="*80)
    print("ANALISI 2: Impatto parametri adattivi")
    print("="*80)
    
    thresholds = [2, 3, 5, 8]
    factors = [1.2, 1.5, 2.0]
    n_seeds = 20
    
    for func, func_name in [(rosenbrock, "Rosenbrock"), (sphere, "Sphere")]:
        print(f"\n--- {func_name} 3D ---")
        print(f"{'Threshold':>10} | {'Factor':>6} | {'Mean':>8} | {'Std':>8} | {'Avg Final Scale':>15}")
        print("-" * 60)
        
        for thresh in thresholds:
            for factor in factors:
                ys = []
                final_scales = []
                for seed in range(n_seeds):
                    sampler = DiagnosticCovSampler(
                        base_scale=3.0, 
                        use_adaptive=True,
                        stagnation_threshold=thresh,
                        scale_increase_factor=factor
                    )
                    y, diag = run_with_diagnostics(func, 3, sampler, seed)
                    ys.append(y)
                    final_scales.append(diag['final_scale'])
                
                print(f"{thresh:10d} | {factor:6.1f} | {np.mean(ys):8.2f} | {np.std(ys):8.2f} | {np.mean(final_scales):15.1f}")


# =============================================================================
# ANALISI 3: Funzioni anisotropiche
# =============================================================================

def analyze_anisotropic():
    """Analizza comportamento su funzioni anisotropiche."""
    print("\n" + "="*80)
    print("ANALISI 3: Funzioni anisotropiche (Cigar, Discus, Ellipsoid)")
    print("="*80)
    
    functions = [
        (cigar, "Cigar"),
        (discus, "Discus"),
        (ellipsoid, "Ellipsoid"),
    ]
    
    scales = [3.0, 5.0, 8.0, 12.0]
    n_seeds = 15
    
    for dim in [3, 10]:
        print(f"\n--- Dim = {dim} ---")
        print(f"{'Function':>10} | " + " | ".join(f"s={s:<5}" for s in scales) + " | Gaussian")
        print("-" * (15 + 10 * (len(scales) + 1)))
        
        for func, func_name in functions:
            row = f"{func_name:>10} |"
            
            for scale in scales:
                ys = []
                for seed in range(n_seeds):
                    sampler = DiagnosticCovSampler(base_scale=scale, use_adaptive=False)
                    y, _ = run_with_diagnostics(func, dim, sampler, seed)
                    ys.append(y)
                row += f" {np.mean(ys):8.1f} |"
            
            # Gaussian baseline
            gauss_ys = []
            for seed in range(n_seeds):
                bounds = [(-5.0, 10.0)] * dim
                opt = ALBA(bounds=bounds, total_budget=100, local_search_ratio=0.3,
                          local_search_sampler=GaussianLocalSearchSampler(), use_drilling=False, seed=seed)
                for _ in range(100):
                    x = opt.ask()
                    opt.tell(x, func(x))
                gauss_ys.append(opt.best_y)
            row += f" {np.mean(gauss_ys):8.1f}"
            
            print(row)


# =============================================================================
# ANALISI 4: Correlazione condizionamento-scala ottimale
# =============================================================================

def analyze_condition_scale_correlation():
    """Correlazione tra condizionamento della Cov e scala ottimale."""
    print("\n" + "="*80)
    print("ANALISI 4: Correlazione condizionamento e scala ottimale")
    print("="*80)
    
    functions = [
        (sphere, "Sphere"),
        (rosenbrock, "Rosenbrock"),
        (ellipsoid, "Ellipsoid"),
    ]
    
    n_seeds = 20
    
    for func, func_name in functions:
        print(f"\n--- {func_name} ---")
        
        for dim in [3, 10]:
            # Raccogli condizionamento tipico
            all_conditions = []
            for seed in range(n_seeds):
                sampler = DiagnosticCovSampler(base_scale=5.0, use_adaptive=False)
                _, diag = run_with_diagnostics(func, dim, sampler, seed)
                all_conditions.extend(sampler._condition_history)
            
            avg_cond = np.mean(all_conditions) if all_conditions else 0
            max_cond = np.max(all_conditions) if all_conditions else 0
            
            # Trova scala ottimale
            best_scale = None
            best_mean = float('inf')
            for scale in [2.0, 3.0, 5.0, 8.0, 12.0]:
                ys = []
                for seed in range(n_seeds):
                    sampler = DiagnosticCovSampler(base_scale=scale, use_adaptive=False)
                    y, _ = run_with_diagnostics(func, dim, sampler, seed)
                    ys.append(y)
                if np.mean(ys) < best_mean:
                    best_mean = np.mean(ys)
                    best_scale = scale
            
            print(f"  Dim {dim:2d}: avg_cond={avg_cond:8.1f}, max_cond={max_cond:10.1f}, optimal_scale={best_scale}")


# =============================================================================
# ANALISI 5: Distribuzione risultati (non solo media)
# =============================================================================

def analyze_result_distribution():
    """Analizza distribuzione completa, non solo media."""
    print("\n" + "="*80)
    print("ANALISI 5: Distribuzione risultati (percentili)")
    print("="*80)
    
    n_seeds = 30
    
    configs = [
        ("Fixed 3.0", {"base_scale": 3.0, "use_adaptive": False}),
        ("Fixed 8.0", {"base_scale": 8.0, "use_adaptive": False}),
        ("Adaptive base=3", {"base_scale": 3.0, "use_adaptive": True, "stagnation_threshold": 3}),
        ("Adaptive base=5", {"base_scale": 5.0, "use_adaptive": True, "stagnation_threshold": 3}),
    ]
    
    for func, func_name in [(rosenbrock, "Rosenbrock"), (rastrigin, "Rastrigin")]:
        print(f"\n--- {func_name} 3D ---")
        print(f"{'Config':>20} | {'Mean':>8} | {'Median':>8} | {'P10':>8} | {'P90':>8} | {'Max':>8}")
        print("-" * 75)
        
        for config_name, config_params in configs:
            ys = []
            for seed in range(n_seeds):
                sampler = DiagnosticCovSampler(**config_params)
                y, _ = run_with_diagnostics(func, 3, sampler, seed)
                ys.append(y)
            
            ys = np.array(ys)
            print(f"{config_name:>20} | {np.mean(ys):8.2f} | {np.median(ys):8.2f} | "
                  f"{np.percentile(ys, 10):8.2f} | {np.percentile(ys, 90):8.2f} | {np.max(ys):8.2f}")


# =============================================================================
# ANALISI 6: Quando Cov è meglio/peggio di Gaussian?
# =============================================================================

def analyze_cov_vs_gaussian_detailed():
    """Analisi dettagliata Cov vs Gaussian."""
    print("\n" + "="*80)
    print("ANALISI 6: Quando Cov batte Gaussian e quando no?")
    print("="*80)
    
    functions = [
        (sphere, "Sphere"),
        (rosenbrock, "Rosenbrock"),
        (rastrigin, "Rastrigin"),
        (cigar, "Cigar"),
        (discus, "Discus"),
    ]
    
    n_seeds = 20
    
    for dim in [3, 10]:
        print(f"\n--- Dim = {dim} ---")
        print(f"{'Function':>12} | {'Gauss':>10} | {'Cov s=3':>10} | {'Cov s=8':>10} | {'Best':>10}")
        print("-" * 65)
        
        for func, func_name in functions:
            # Gaussian
            gauss_ys = []
            for seed in range(n_seeds):
                bounds = [(-5.0, 10.0)] * dim
                opt = ALBA(bounds=bounds, total_budget=100, local_search_ratio=0.3,
                          local_search_sampler=GaussianLocalSearchSampler(), use_drilling=False, seed=seed)
                for _ in range(100):
                    x = opt.ask()
                    opt.tell(x, func(x))
                gauss_ys.append(opt.best_y)
            
            # Cov s=3
            cov3_ys = []
            for seed in range(n_seeds):
                sampler = DiagnosticCovSampler(base_scale=3.0, use_adaptive=False)
                y, _ = run_with_diagnostics(func, dim, sampler, seed)
                cov3_ys.append(y)
            
            # Cov s=8
            cov8_ys = []
            for seed in range(n_seeds):
                sampler = DiagnosticCovSampler(base_scale=8.0, use_adaptive=False)
                y, _ = run_with_diagnostics(func, dim, sampler, seed)
                cov8_ys.append(y)
            
            means = {
                'Gauss': np.mean(gauss_ys),
                'Cov s=3': np.mean(cov3_ys),
                'Cov s=8': np.mean(cov8_ys),
            }
            best = min(means, key=means.get)
            
            print(f"{func_name:>12} | {means['Gauss']:10.2f} | {means['Cov s=3']:10.2f} | "
                  f"{means['Cov s=8']:10.2f} | {best:>10}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    analyze_adaptive_failures()
    analyze_adaptive_parameters()
    analyze_anisotropic()
    analyze_condition_scale_correlation()
    analyze_result_distribution()
    analyze_cov_vs_gaussian_detailed()
    
    print("\n" + "="*80)
    print("FINE ANALISI APPROFONDITA")
    print("="*80)


if __name__ == "__main__":
    main()
