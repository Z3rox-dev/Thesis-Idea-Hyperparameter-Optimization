#!/usr/bin/env python3
"""
Debug CopulaHPO v2: perché va peggio del random?

Ipotesi da testare:
1. Correlazione troppo shrinkage → perde informazione
2. Marginali non catturano bene la distribuzione
3. Sampling non esplora abbastanza
4. Elite selection non funziona
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from copula_hpo_v2 import CopulaHPO, CopulaHPO_Continuous, HyperparameterSpec
from scipy import stats

np.set_printoptions(precision=3, suppress=True)


def sphere(x):
    return float(np.sum(x**2))


def rosenbrock(x):
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


# =============================================================================
# TEST 1: Cosa succede alle marginali?
# =============================================================================
def test_marginals():
    print("=" * 70)
    print("TEST 1: Analisi Marginali")
    print("=" * 70)
    
    d = 5
    bounds = [(-5, 5)] * d
    opt = CopulaHPO_Continuous(bounds, seed=42)
    
    # Run per un po'
    for i in range(50):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    # Ora analizziamo
    print(f"\n  Dopo 50 iterazioni:")
    print(f"  best_y = {opt.best_y:.4f}")
    print(f"  best_x = {opt.best_x}")
    
    # Elite
    y_arr = np.array(opt.opt.y_hist)
    tau = np.quantile(y_arr, opt.opt.cfg.gamma)
    elite_mask = y_arr <= tau
    n_elite = elite_mask.sum()
    
    print(f"\n  Elite: {n_elite} punti (tau={tau:.2f})")
    
    # Marginali fitted
    elite_arrays, _ = opt.opt._get_elite_arrays()
    
    print(f"\n  Marginali empiriche (elite):")
    for j in range(min(d, 3)):
        vals = elite_arrays[j]
        print(f"    x{j}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"range=[{vals.min():.2f}, {vals.max():.2f}]")
    
    # Problema: le marginali sono molto ristrette?
    print(f"\n  Bounds originali: [-5, 5]")
    print(f"  Range degli elite è molto più piccolo?")


# =============================================================================
# TEST 2: Cosa succede alla correlazione?
# =============================================================================
def test_correlation():
    print("\n" + "=" * 70)
    print("TEST 2: Analisi Correlazione")
    print("=" * 70)
    
    d = 5
    bounds = [(-5, 5)] * d
    opt = CopulaHPO_Continuous(bounds, seed=42)
    
    for i in range(50):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    # Ricrea la pipeline di fit
    elite_arrays, n_elite = opt.opt._get_elite_arrays()
    
    # Fit marginals and transform
    U = np.zeros((n_elite, d))
    for j in range(d):
        opt.opt.marginals[j].fit(elite_arrays[j])
        U[:, j] = opt.opt.marginals[j].to_uniform(elite_arrays[j])
    
    print(f"\n  U (spazio uniforme) dopo transform:")
    print(f"    Shape: {U.shape}")
    print(f"    Min per dim: {U.min(axis=0)}")
    print(f"    Max per dim: {U.max(axis=0)}")
    print(f"    Mean per dim: {U.mean(axis=0)}")
    
    # Gaussianize
    U_clip = np.clip(U, 1e-6, 1-1e-6)
    Z = stats.norm.ppf(U_clip)
    Z = Z - Z.mean(axis=0, keepdims=True)
    
    print(f"\n  Z (spazio Gaussiano):")
    print(f"    Mean per dim: {Z.mean(axis=0)}")
    print(f"    Std per dim: {Z.std(axis=0)}")
    
    # Correlation
    if n_elite > 1:
        cov = np.cov(Z.T)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
    else:
        corr = np.eye(d)
    
    print(f"\n  Matrice Correlazione (prima di shrinkage):")
    print(corr)
    
    # Dopo shrinkage
    alpha = max(0.1, min(0.9, d / max(n_elite, 1)))
    corr_shrunk = (1.0 - alpha) * corr + alpha * np.eye(d)
    
    print(f"\n  Alpha shrinkage: {alpha:.2f}")
    print(f"  Matrice Correlazione (dopo shrinkage):")
    print(corr_shrunk)
    
    # Problema: se alpha è troppo alto, perdiamo tutta l'informazione!
    if alpha > 0.5:
        print(f"\n  ⚠ PROBLEMA: alpha={alpha:.2f} è troppo alto!")
        print(f"    Con n_elite={n_elite} e d={d}, la correlazione è quasi identità!")


# =============================================================================
# TEST 3: Dove campiona?
# =============================================================================
def test_sampling_distribution():
    print("\n" + "=" * 70)
    print("TEST 3: Distribuzione dei Sample")
    print("=" * 70)
    
    d = 5
    bounds = [(-5, 5)] * d
    
    # Run completo e traccia tutti i punti
    opt = CopulaHPO_Continuous(bounds, seed=42)
    
    all_x = []
    for i in range(100):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        all_x.append(x.copy())
    
    all_x = np.array(all_x)
    
    print(f"\n  Distribuzione di tutti i 100 punti campionati:")
    for j in range(min(d, 3)):
        print(f"    x{j}: mean={all_x[:, j].mean():.3f}, std={all_x[:, j].std():.3f}")
    
    # Confronta con random
    rng = np.random.default_rng(42)
    random_x = rng.uniform(-5, 5, (100, d))
    
    print(f"\n  Random search (per confronto):")
    for j in range(min(d, 3)):
        print(f"    x{j}: mean={random_x[:, j].mean():.3f}, std={random_x[:, j].std():.3f}")
    
    # Il problema: CopulaHPO sta campionando troppo vicino agli elite?
    print(f"\n  Distanza media dall'origine:")
    print(f"    CopulaHPO: {np.linalg.norm(all_x, axis=1).mean():.3f}")
    print(f"    Random: {np.linalg.norm(random_x, axis=1).mean():.3f}")
    
    # Primi 20 vs ultimi 20
    print(f"\n  Evoluzione nel tempo:")
    print(f"    Primi 20: dist_mean={np.linalg.norm(all_x[:20], axis=1).mean():.3f}")
    print(f"    Ultimi 20: dist_mean={np.linalg.norm(all_x[-20:], axis=1).mean():.3f}")


# =============================================================================
# TEST 4: Il problema è l'inversione delle marginali?
# =============================================================================
def test_marginal_inversion():
    print("\n" + "=" * 70)
    print("TEST 4: Inversione Marginali")
    print("=" * 70)
    
    from copula_hpo_v2 import ContinuousMarginal
    
    # Simula elite vicini a 0 (come in sphere)
    elite_vals = np.array([-1.0, -0.5, 0.2, 0.8, 1.5, -0.3])
    
    marginal = ContinuousMarginal(-5, 5)
    marginal.fit(elite_vals)
    
    print(f"\n  Elite values: {elite_vals}")
    print(f"  Sorted: {marginal.sorted_vals}")
    
    # Trasforma a uniform
    u = marginal.to_uniform(elite_vals)
    print(f"  To uniform: {u}")
    
    # Ora campiona da vari u
    test_u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x_back = marginal.from_uniform(test_u)
    print(f"\n  Inversione:")
    print(f"    u = {test_u}")
    print(f"    x = {x_back}")
    
    # PROBLEMA: i valori escono solo nel range degli elite!
    print(f"\n  Range elite: [{elite_vals.min():.2f}, {elite_vals.max():.2f}]")
    print(f"  Range output: [{x_back.min():.2f}, {x_back.max():.2f}]")
    print(f"  Range bounds: [-5, 5]")
    
    if x_back.max() < 2:
        print(f"\n  ⚠ PROBLEMA: L'inversione è confinata nel range degli elite!")
        print(f"    CopulaHPO non può esplorare fuori da dove ha già visto!")


# =============================================================================
# TEST 5: Confronto con/senza exploration
# =============================================================================
def test_exploration_effect():
    print("\n" + "=" * 70)
    print("TEST 5: Effetto Exploration (eps_explore)")
    print("=" * 70)
    
    d = 10
    bounds = [(-5, 5)] * d
    budget = 200
    n_seeds = 3
    
    for eps in [0.05, 0.1, 0.2, 0.3]:
        scores = []
        for seed in range(n_seeds):
            opt = CopulaHPO_Continuous(bounds, seed=seed, eps_explore=eps)
            for _ in range(budget):
                x = opt.ask()
                y = sphere(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        
        print(f"  eps_explore={eps:.2f}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


# =============================================================================
# TEST 6: Confronto shrinkage
# =============================================================================
def test_shrinkage_effect():
    print("\n" + "=" * 70)
    print("TEST 6: Effetto Alpha Shrinkage")
    print("=" * 70)
    
    d = 10
    bounds = [(-5, 5)] * d
    budget = 200
    n_seeds = 3
    
    for alpha in [0.0, 0.1, 0.3, 0.5, 0.8]:
        scores = []
        for seed in range(n_seeds):
            opt = CopulaHPO_Continuous(bounds, seed=seed, alpha_corr=alpha)
            for _ in range(budget):
                x = opt.ask()
                y = sphere(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        
        print(f"  alpha_corr={alpha:.1f}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEBUG CopulaHPO v2")
    print("=" * 70)
    
    test_marginals()
    test_correlation()
    test_sampling_distribution()
    test_marginal_inversion()
    test_exploration_effect()
    test_shrinkage_effect()
    
    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
