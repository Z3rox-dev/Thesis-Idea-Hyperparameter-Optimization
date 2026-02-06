"""
ALBA Bug Hunter - Cerca anomalie numeriche su molte funzioni diverse.

Monitora:
1. Risultati che esplodono (best_y >> optimum atteso)
2. NaN/Inf nei risultati
3. Convergenza anomala (peggiora invece di migliorare)
4. Varianza estrema tra seeds
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')
import numpy as np
import traceback
from typing import Dict, List, Callable, Tuple, Any

from alba_framework_potential.optimizer import ALBA

# ============================================================================
# SUITE DI FUNZIONI TEST
# ============================================================================

def sphere(x):
    """Optimum: 0 at origin"""
    return np.sum(x**2)

def rosenbrock(x):
    """Optimum: 0 at (1,1,...,1)"""
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def rastrigin(x):
    """Optimum: 0 at origin, many local minima"""
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

def ackley(x):
    """Optimum: 0 at origin"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def schwefel(x):
    """Optimum: 0 at (420.9687, ...), tricky landscape"""
    # Scaled to [0,1] -> [0, 500]
    x_scaled = x * 500
    n = len(x)
    return 418.9829 * n - np.sum(x_scaled * np.sin(np.sqrt(np.abs(x_scaled))))

def levy(x):
    """Optimum: 0 at (1,1,...,1)"""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def griewank(x):
    """Optimum: 0 at origin"""
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1

def michalewicz(x):
    """Optimum depends on dimension, steep ridges"""
    m = 10
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2*m))

def ill_conditioned(x):
    """Scaling esponenziale: 1, 10, 100, 1000, ..."""
    scales = np.array([10**i for i in range(len(x))])
    return np.sum(scales * x**2)

def extreme_scale(x):
    """Scaling estremo: 1, 100, 10000, ..."""
    scales = np.array([10**(2*i) for i in range(len(x))])
    return np.sum(scales * x**2)

def discontinuous(x):
    """Funzione discontinua"""
    return np.sum(np.floor(np.abs(x * 10)))

def noisy_sphere(x):
    """Sphere con rumore"""
    return np.sum(x**2) + np.random.normal(0, 0.1)

def plateau(x):
    """Funzione piatta a tratti"""
    return np.sum(np.floor(np.abs(x * 5)))

def cliff(x):
    """Cliff function - salto brusco"""
    if x[0] < 0.5:
        return np.sum(x**2)
    else:
        return np.sum(x**2) + 100

def ridge(x):
    """Narrow ridge"""
    return x[0] + 100 * np.sum(x[1:]**2)

def different_powers(x):
    """Different powers: x1^2 + x2^4 + x3^6 + ..."""
    powers = np.arange(2, 2 + 2*len(x), 2)
    return np.sum(np.abs(x) ** powers)

def bent_cigar(x):
    """Bent cigar - high conditioning"""
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

def discus(x):
    """Discus - inverse of bent cigar"""
    return 1e6 * x[0]**2 + np.sum(x[1:]**2)

def ellipsoid(x):
    """Ellipsoid with condition number 10^6"""
    n = len(x)
    coeffs = np.array([10**(6 * i / (n-1)) for i in range(n)])
    return np.sum(coeffs * x**2)

def sharp_ridge(x):
    """Sharp ridge"""
    return x[0]**2 + 100 * np.sqrt(np.sum(x[1:]**2) + 1e-10)

def sum_of_different_squares(x):
    """Sum of different squares"""
    i = np.arange(1, len(x) + 1)
    return np.sum(i * x**2)


# ============================================================================
# FUNZIONI CON OTTIMO NOTO
# ============================================================================

FUNCTIONS = {
    # Semplici (optimum ≈ 0)
    "Sphere": (sphere, 0.0),
    "SumDiffSquares": (sum_of_different_squares, 0.0),
    
    # Moderate (optimum ≈ 0, ma più difficili)
    "Rosenbrock": (rosenbrock, 0.0),
    "Levy": (levy, 0.0),
    "Ackley": (ackley, 0.0),
    "Griewank": (griewank, 0.0),
    
    # Multimodali
    "Rastrigin": (rastrigin, 0.0),
    "Schwefel": (schwefel, 0.0),  # Optimum non a 0 con scaling
    "Michalewicz": (michalewicz, None),  # Optimum dipende da dim
    
    # Ill-conditioned (potenziali problemi numerici)
    "IllConditioned": (ill_conditioned, 0.0),
    "ExtremeScale": (extreme_scale, 0.0),
    "BentCigar": (bent_cigar, 0.0),
    "Discus": (discus, 0.0),
    "Ellipsoid": (ellipsoid, 0.0),
    "DifferentPowers": (different_powers, 0.0),
    
    # Strutture particolari
    "Ridge": (ridge, 0.0),
    "SharpRidge": (sharp_ridge, 0.0),
    
    # Non-smooth / discontinue
    "Discontinuous": (discontinuous, 0.0),
    "Plateau": (plateau, 0.0),
    "Cliff": (cliff, 0.0),
    
    # Noisy
    "NoisySphere": (noisy_sphere, 0.0),
}


# ============================================================================
# BUG HUNTER
# ============================================================================

class BugHunter:
    def __init__(self, dim: int = 5, budget: int = 150, n_seeds: int = 5):
        self.dim = dim
        self.budget = budget
        self.n_seeds = n_seeds
        self.results: Dict[str, Dict] = {}
        self.anomalies: List[Dict] = []
    
    def run_single(self, func_name: str, func: Callable, seed: int) -> Dict:
        """Esegue una singola run e monitora anomalie."""
        result = {
            "seed": seed,
            "best_y": None,
            "history": [],
            "error": None,
            "warnings": [],
        }
        
        try:
            opt = ALBA(
                bounds=[(0.0, 1.0)] * self.dim,
                maximize=False,
                use_drilling=True,
                use_potential_field=True,
                use_coherence_gating=True,
                seed=seed,
                total_budget=self.budget,
            )
            
            prev_best = float('inf')
            stagnation_count = 0
            
            for i in range(self.budget):
                x = opt.ask()
                y = func(np.array(x))
                
                # Check per NaN/Inf
                if np.isnan(y) or np.isinf(y):
                    result["warnings"].append(f"iter {i}: y={y} (NaN/Inf)")
                    y = 1e10  # Fallback
                
                opt.tell(x, y)
                result["history"].append(opt.best_y)
                
                # Check per peggioramento
                if opt.best_y > prev_best * 1.01:
                    result["warnings"].append(f"iter {i}: best peggiorato {prev_best:.2e} -> {opt.best_y:.2e}")
                
                # Check stagnation
                if abs(opt.best_y - prev_best) < 1e-10:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                prev_best = opt.best_y
            
            result["best_y"] = opt.best_y
            result["final_stagnation"] = stagnation_count
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        
        return result
    
    def analyze_function(self, func_name: str, func: Callable, optimum: float = None):
        """Analizza una funzione su tutti i seeds."""
        print(f"  Testing {func_name}...", end=" ", flush=True)
        
        runs = []
        for seed in range(self.n_seeds):
            runs.append(self.run_single(func_name, func, seed))
        
        # Aggregazione
        valid_runs = [r for r in runs if r["best_y"] is not None]
        
        if not valid_runs:
            print("❌ FAILED (all runs errored)")
            self.anomalies.append({
                "func": func_name,
                "type": "ALL_FAILED",
                "errors": [r["error"] for r in runs],
            })
            return
        
        bests = [r["best_y"] for r in valid_runs]
        mean_best = np.mean(bests)
        std_best = np.std(bests)
        min_best = np.min(bests)
        max_best = np.max(bests)
        
        # Analisi anomalie
        anomalies = []
        
        # 1. NaN/Inf in output
        for r in runs:
            if r["error"]:
                anomalies.append({"type": "ERROR", "detail": r["error"]})
        
        # 2. Varianza estrema tra seeds
        if std_best > mean_best * 2:
            anomalies.append({
                "type": "HIGH_VARIANCE",
                "detail": f"std={std_best:.2e} >> mean={mean_best:.2e}",
            })
        
        # 3. Gap enorme tra min e max
        if max_best > min_best * 100 and min_best > 0:
            anomalies.append({
                "type": "EXTREME_GAP",
                "detail": f"max/min = {max_best/min_best:.0f}x",
            })
        
        # 4. Risultato >> optimum atteso (se noto)
        if optimum is not None and optimum >= 0:
            if mean_best > 1e6:
                anomalies.append({
                    "type": "EXPLOSION",
                    "detail": f"mean={mean_best:.2e} >> optimum={optimum}",
                })
        
        # 5. Warnings durante run
        total_warnings = sum(len(r["warnings"]) for r in runs)
        if total_warnings > 0:
            anomalies.append({
                "type": "WARNINGS",
                "detail": f"{total_warnings} warnings across runs",
                "examples": [w for r in runs for w in r["warnings"][:2]],
            })
        
        # Report
        self.results[func_name] = {
            "mean": mean_best,
            "std": std_best,
            "min": min_best,
            "max": max_best,
            "runs": runs,
            "anomalies": anomalies,
        }
        
        if anomalies:
            print(f"⚠️  {mean_best:.2e} ± {std_best:.2e} ({len(anomalies)} anomalies)")
            for a in anomalies:
                self.anomalies.append({"func": func_name, **a})
        else:
            print(f"✓ {mean_best:.4f} ± {std_best:.4f}")
    
    def run_all(self):
        """Esegue tutti i test."""
        print("="*70)
        print(f"  ALBA BUG HUNTER (dim={self.dim}, budget={self.budget}, seeds={self.n_seeds})")
        print("="*70)
        
        for func_name, (func, optimum) in FUNCTIONS.items():
            self.analyze_function(func_name, func, optimum)
        
        self.report()
    
    def report(self):
        """Report finale delle anomalie trovate."""
        print("\n" + "="*70)
        print("  ANOMALIE RILEVATE")
        print("="*70)
        
        if not self.anomalies:
            print("  ✓ Nessuna anomalia rilevata!")
            return
        
        # Raggruppa per tipo
        by_type = {}
        for a in self.anomalies:
            t = a["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(a)
        
        for anomaly_type, items in by_type.items():
            print(f"\n  [{anomaly_type}] - {len(items)} casi:")
            for item in items:
                func = item.get("func", "?")
                detail = item.get("detail", "")
                print(f"    • {func}: {detail}")
                if "examples" in item:
                    for ex in item["examples"][:3]:
                        print(f"        - {ex}")
        
        # Tabella riassuntiva
        print("\n" + "="*70)
        print("  FUNZIONI DA INVESTIGARE (ordinate per gravità)")
        print("="*70)
        
        # Ordina per numero anomalie
        funcs_with_issues = {}
        for a in self.anomalies:
            func = a.get("func", "?")
            if func not in funcs_with_issues:
                funcs_with_issues[func] = []
            funcs_with_issues[func].append(a["type"])
        
        sorted_funcs = sorted(funcs_with_issues.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"\n{'Funzione':<20} {'#Issues':<10} {'Tipi':<40}")
        print("-"*70)
        for func, issues in sorted_funcs:
            print(f"{func:<20} {len(issues):<10} {', '.join(set(issues)):<40}")


if __name__ == "__main__":
    hunter = BugHunter(dim=5, budget=150, n_seeds=5)
    hunter.run_all()
