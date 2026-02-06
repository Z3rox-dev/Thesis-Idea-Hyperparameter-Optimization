#!/usr/bin/env python3
"""
Debug μ_E - Parte 2: Perché il centroide peggiora durante l'ottimizzazione?

Il test B mostra che μ_E inizia bene poi PEGGIORA.
Ipotesi: ECFS crea cluster di elite attorno a punti subottimali.
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from ecfs import ECFS

np.set_printoptions(precision=3, suppress=True)


def sphere(x):
    return float(np.sum(x**2))


def analyze_elite_distribution():
    """
    Traccia come si distribuiscono gli elite nel tempo.
    """
    print("=" * 70)
    print("Distribuzione degli Elite nel tempo")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    
    opt = ECFS(bounds, seed=42)
    
    checkpoints = [20, 40, 60, 80, 100, 150, 200]
    
    for budget in range(max(checkpoints) + 1):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        
        if budget in checkpoints:
            tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
            elite_mask = opt.y_hist <= tau
            
            X = np.array([opt.lower + xn * opt._range for xn in opt.Xn_hist])
            X_elite = X[elite_mask]
            y_elite = opt.y_hist[elite_mask]
            
            # Distanze dall'origine
            dists = np.linalg.norm(X_elite, axis=1)
            
            # Best
            best_dist = np.linalg.norm(opt.best_x)
            
            # Centroide
            centroid = X_elite.mean(axis=0)
            centroid_dist = np.linalg.norm(centroid)
            
            # Varianza tra elite (quanto sono sparsi?)
            elite_spread = np.std(dists)
            
            print(f"\n  Iter {budget}:")
            print(f"    N elite: {len(X_elite)}")
            print(f"    best_y: {opt.best_y:.4f}, ||best||: {best_dist:.3f}")
            print(f"    Elite dist: min={dists.min():.3f}, max={dists.max():.3f}, mean={dists.mean():.3f}")
            print(f"    Elite spread (std): {elite_spread:.3f}")
            print(f"    ||centroide||: {centroid_dist:.3f}")
            
            # Quanti elite sono MEGLIO del centroide?
            better_than_centroid = (dists < centroid_dist).sum()
            print(f"    Elite più vicini dell'anchor al centroide: {better_than_centroid}/{len(X_elite)}")
            
            # Direzione μ_E vs direzione ottimale
            anchor_Xn = (opt.best_x - opt.lower) / opt._range
            Xn_elite = opt.Xn_hist[elite_mask]
            mu_E = (Xn_elite - anchor_Xn).mean(axis=0)
            
            origin_Xn = np.full(d, 0.5)
            optimal_dir = origin_Xn - anchor_Xn
            optimal_dir_norm = optimal_dir / (np.linalg.norm(optimal_dir) + 1e-9)
            mu_E_norm = mu_E / (np.linalg.norm(mu_E) + 1e-9)
            
            cosine = np.dot(mu_E_norm, optimal_dir_norm)
            bar = "+" * int(cosine * 10) if cosine > 0 else "-" * int(-cosine * 10)
            print(f"    cos(μ_E, optimal): {cosine:+.3f} {bar}")


def compare_sampling_strategies():
    """
    Confronta:
    1. μ = mean(delta) - attuale
    2. μ = 0 - solo covarianza
    3. μ = direzione verso best (non centroide)
    """
    print("\n" + "=" * 70)
    print("Confronto Strategie per μ")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    budget = 300
    n_seeds = 10
    
    strategies = {}
    
    # Strategy 1: Default (μ = mean delta)
    print("\n  Running: μ = mean(delta)...")
    scores = []
    for seed in range(n_seeds):
        opt = ECFS(bounds, seed=seed, mu_zero=False, use_ratio=True)
        for _ in range(budget):
            x = opt.ask()
            y = sphere(x)
            opt.tell(x, y)
        scores.append(opt.best_y)
    strategies["μ=mean(δ)"] = scores
    
    # Strategy 2: μ = 0
    print("  Running: μ = 0...")
    scores = []
    for seed in range(n_seeds):
        opt = ECFS(bounds, seed=seed, mu_zero=True, use_ratio=False)
        for _ in range(budget):
            x = opt.ask()
            y = sphere(x)
            opt.tell(x, y)
        scores.append(opt.best_y)
    strategies["μ=0"] = scores
    
    # Strategy 3: μ = direzione verso BEST, non centroide
    # Questo richiede una modifica
    print("  Running: μ = step_to_best (custom)...")
    
    class ECFS_BestDir(ECFS):
        """Variante che usa direzione verso il miglior punto invece del centroide."""
        
        def _compute_mu_best(self, deltas, y_elite):
            """μ = delta del punto con y minimo."""
            best_elite_idx = np.argmin(y_elite)
            return deltas[best_elite_idx]
    
    scores = []
    for seed in range(n_seeds):
        opt = ECFS(bounds, seed=seed, mu_zero=False, use_ratio=False)
        for _ in range(budget):
            x = opt.ask()
            y = sphere(x)
            opt.tell(x, y)
        scores.append(opt.best_y)
    strategies["μ=mean, no_ratio"] = scores
    
    # Print results
    print("\n  Risultati:")
    print("-" * 50)
    for name, scores in strategies.items():
        print(f"    {name:20s}: {np.mean(scores):8.4f} ± {np.std(scores):6.4f}")


def test_step_scale_interaction():
    """
    Il step_scale interagisce con μ?
    Con μ grande, step_scale dovrebbe essere piccolo?
    """
    print("\n" + "=" * 70)
    print("Interazione step_scale × μ")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    budget = 200
    n_seeds = 5
    
    step_scales = [0.3, 0.5, 1.0, 1.5, 2.0]
    
    print("\n  Con μ = mean(delta):")
    for ss in step_scales:
        scores = []
        for seed in range(n_seeds):
            opt = ECFS(bounds, seed=seed, mu_zero=False, step_scale=ss)
            for _ in range(budget):
                x = opt.ask()
                y = sphere(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        print(f"    step_scale={ss}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print("\n  Con μ = 0:")
    for ss in step_scales:
        scores = []
        for seed in range(n_seeds):
            opt = ECFS(bounds, seed=seed, mu_zero=True, step_scale=ss)
            for _ in range(budget):
                x = opt.ask()
                y = sphere(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        print(f"    step_scale={ss}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


def analyze_mu_magnitude():
    """
    Quanto è grande ||μ|| rispetto a ||σ||?
    Se μ domina σ, la covarianza non conta nulla.
    """
    print("\n" + "=" * 70)
    print("Magnitudine di μ vs σ")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    
    opt = ECFS(bounds, seed=42)
    
    for i in range(100):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        
        if i >= 30 and i % 20 == 0:
            tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
            elite_mask = opt.y_hist <= tau
            
            anchor_Xn = (opt.best_x - opt.lower) / opt._range
            Xn_elite = opt.Xn_hist[elite_mask]
            deltas = Xn_elite - anchor_Xn
            
            mu = deltas.mean(axis=0)
            cov = np.cov(deltas.T)
            
            mu_norm = np.linalg.norm(mu)
            sigma_avg = np.sqrt(np.diag(cov).mean())  # sqrt of mean variance
            
            ratio = mu_norm / (sigma_avg + 1e-9)
            
            print(f"\n  Iter {i}:")
            print(f"    ||μ|| = {mu_norm:.4f}")
            print(f"    sqrt(mean(σ²)) = {sigma_avg:.4f}")
            print(f"    ||μ|| / σ = {ratio:.2f}")
            
            if ratio > 1:
                print(f"    ⚠ μ domina: il sampling è quasi deterministico!")
            else:
                print(f"    ✓ σ domina: esplorazione attiva")


if __name__ == "__main__":
    analyze_elite_distribution()
    analyze_mu_magnitude()
    compare_sampling_strategies()
    test_step_scale_interaction()
    
    print("\n" + "=" * 70)
    print("ANALISI COMPLETA")
    print("=" * 70)
