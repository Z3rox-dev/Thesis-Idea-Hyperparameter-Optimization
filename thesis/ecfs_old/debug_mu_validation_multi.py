#!/usr/bin/env python3
"""
Validazione su funzioni diverse: μ=0 vs μ=mean su varie topologie.

Funzioni testate:
1. Sphere - convessa, simmetrica, unimodale
2. Rosenbrock - valle stretta, non-convessa
3. Rastrigin - multimodale (tantissimi minimi locali)
4. Ackley - multimodale con plateau
5. Schwefel - multimodale, ingannevole
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from ecfs import ECFS

np.set_printoptions(precision=3, suppress=True)


# =============================================================================
# Funzioni Test
# =============================================================================

def sphere(x):
    """Convessa, simmetrica. Opt: [0,...,0] = 0"""
    return float(np.sum(x**2))


def rosenbrock(x):
    """Valle stretta. Opt: [1,...,1] = 0"""
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


def rastrigin(x):
    """Multimodale (10^d minimi locali). Opt: [0,...,0] = 0"""
    d = len(x)
    return float(10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x)))


def ackley(x):
    """Multimodale con plateau. Opt: [0,...,0] = 0"""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return float(-20*np.exp(-0.2*np.sqrt(sum1/d)) - np.exp(sum2/d) + 20 + np.e)


def schwefel(x):
    """Multimodale, minimo lontano dal centro. Opt: [420.9,...,420.9] = 0"""
    d = len(x)
    return float(418.9829*d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


FUNCTIONS = {
    "Sphere": {"fn": sphere, "bounds": (-5, 5), "opt": "0", "type": "unimodal"},
    "Rosenbrock": {"fn": rosenbrock, "bounds": (-5, 10), "opt": "1s", "type": "valley"},
    "Rastrigin": {"fn": rastrigin, "bounds": (-5.12, 5.12), "opt": "0", "type": "multimodal"},
    "Ackley": {"fn": ackley, "bounds": (-5, 5), "opt": "0", "type": "multimodal"},
    "Schwefel": {"fn": schwefel, "bounds": (-500, 500), "opt": "420.9", "type": "deceptive"},
}


# =============================================================================
# Test Suite
# =============================================================================

def run_comparison(fn, bounds_tuple, d, budget, n_seeds):
    """Confronta μ=mean vs μ=0 vs μ=0+no_ratio."""
    bounds = [bounds_tuple] * d
    
    configs = {
        "μ=mean (default)": {"mu_zero": False, "use_ratio": True},
        "μ=mean, no_ratio": {"mu_zero": False, "use_ratio": False},
        "μ=0, ratio":       {"mu_zero": True, "use_ratio": True},
        "μ=0, no_ratio":    {"mu_zero": True, "use_ratio": False},
    }
    
    results = {}
    for name, kwargs in configs.items():
        scores = []
        for seed in range(n_seeds):
            opt = ECFS(bounds, seed=seed, **kwargs)
            for _ in range(budget):
                x = opt.ask()
                y = fn(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        results[name] = (np.mean(scores), np.std(scores))
    
    return results


def analyze_elite_dynamics(fn, bounds_tuple, d, fn_name):
    """Analizza come evolvono gli elite per una funzione."""
    bounds = [bounds_tuple] * d
    opt = ECFS(bounds, seed=42)
    
    print(f"\n  Dinamica elite per {fn_name}:")
    print(f"  " + "-" * 50)
    
    for i in range(201):
        x = opt.ask()
        y = fn(x)
        opt.tell(x, y)
        
        if i in [30, 60, 100, 150, 200]:
            tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
            elite_mask = opt.y_hist <= tau
            
            Xn_elite = opt.Xn_hist[elite_mask]
            n_elite = len(Xn_elite)
            
            # Spread (quanto sono diversi tra loro gli elite)
            if n_elite > 1:
                pairwise_dists = []
                for j in range(min(n_elite, 20)):
                    for k in range(j+1, min(n_elite, 20)):
                        pairwise_dists.append(np.linalg.norm(Xn_elite[j] - Xn_elite[k]))
                spread = np.mean(pairwise_dists) if pairwise_dists else 0
            else:
                spread = 0
            
            # μ direction analysis
            anchor_Xn = (opt.best_x - opt.lower) / opt._range
            deltas = Xn_elite - anchor_Xn
            mu = deltas.mean(axis=0)
            mu_norm = np.linalg.norm(mu)
            
            # Covarianza
            if n_elite > d:
                cov = np.cov(deltas.T)
                sigma_avg = np.sqrt(np.diag(cov).mean())
                ratio = mu_norm / (sigma_avg + 1e-9)
            else:
                sigma_avg = 0
                ratio = float('inf')
            
            print(f"    iter {i:3d}: n_elite={n_elite:2d}, spread={spread:.3f}, "
                  f"||μ||={mu_norm:.4f}, σ_avg={sigma_avg:.4f}, μ/σ={ratio:.2f}")


def main():
    print("=" * 70)
    print("VALIDAZIONE μ=0 vs μ=mean SU FUNZIONI DIVERSE")
    print("=" * 70)
    
    d = 10
    budget = 300
    n_seeds = 10
    
    # Prima: confronto performance
    print("\n" + "=" * 70)
    print("PARTE 1: Confronto Performance")
    print("=" * 70)
    
    for fn_name, info in FUNCTIONS.items():
        fn = info["fn"]
        bounds_tuple = (info["bounds"][0], info["bounds"][1]) if isinstance(info["bounds"], tuple) else info["bounds"]
        
        print(f"\n  {fn_name} ({info['type']}, opt≈{info['opt']}):")
        print(f"  " + "-" * 50)
        
        results = run_comparison(fn, bounds_tuple, d, budget, n_seeds)
        
        # Trova il migliore
        best_config = min(results.keys(), key=lambda k: results[k][0])
        
        for name, (mean, std) in results.items():
            marker = " ✓ BEST" if name == best_config else ""
            print(f"    {name:20s}: {mean:12.4f} ± {std:8.4f}{marker}")
    
    # Seconda: dinamica elite
    print("\n" + "=" * 70)
    print("PARTE 2: Dinamica Elite (clustering)")
    print("=" * 70)
    
    for fn_name, info in FUNCTIONS.items():
        fn = info["fn"]
        bounds_tuple = (info["bounds"][0], info["bounds"][1]) if isinstance(info["bounds"], tuple) else info["bounds"]
        analyze_elite_dynamics(fn, bounds_tuple, d, fn_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  Se μ=0 vince su TUTTE le funzioni → l'ipotesi è validata:
    Il centroide degli elite NON è un buon indicatore della direzione
    ottimale in contesti di ottimizzazione iterativa con ECFS.
    
  Se μ=mean vince su alcune funzioni → μ può essere utile in certi casi,
    e potrebbe valere la pena di una strategia adattiva.
    """)


if __name__ == "__main__":
    main()
