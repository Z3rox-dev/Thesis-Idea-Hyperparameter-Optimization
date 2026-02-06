#!/usr/bin/env python3
"""
Debug approfondito: Perché μ_E non funziona bene?

Ipotesi da testare:
1. μ_E punta verso il centroide degli elite, non verso l'ottimo
2. μ_E è troppo rumoroso con pochi punti
3. μ_E e la direzione ottimale sono disallineati
4. μ_E cambia troppo tra iterazioni (instabile)
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/nuovo_progetto')

from ecfs import ECFS

np.set_printoptions(precision=4, suppress=True)


def sphere(x):
    """Optimum at origin."""
    return float(np.sum(x**2))


def rosenbrock(x):
    """Optimum at [1,1,...,1]."""
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


# =============================================================================
# TEST A: Dove punta μ_E rispetto alla direzione ottimale?
# =============================================================================
def test_mu_direction():
    """
    Su Sphere, la direzione ottimale da qualsiasi punto è verso l'origine.
    μ_E dovrebbe puntare in quella direzione?
    """
    print("=" * 70)
    print("TEST A: Dove punta μ_E?")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    
    opt = ECFS(bounds, seed=42)
    
    # Warmup con punti random
    for i in range(50):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    # Ora analizziamo μ_E
    print(f"\n  Dopo 50 iterazioni:")
    print(f"    best_y = {opt.best_y:.4f}")
    print(f"    best_x = {opt.best_x}")
    print(f"    ||best_x|| = {np.linalg.norm(opt.best_x):.4f}")
    
    # Calcola τ e trova elite
    tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
    elite_mask = opt.y_hist <= tau
    n_elite = elite_mask.sum()
    
    print(f"    tau = {tau:.4f}")
    print(f"    n_elite = {n_elite}")
    
    # Anchor = best point (normalizzato)
    anchor_Xn = (opt.best_x - opt.lower) / opt._range
    
    # Elite points (normalizzati)
    Xn_elite = opt.Xn_hist[elite_mask]
    
    # Calcola delta
    deltas = Xn_elite - anchor_Xn
    mu_E = deltas.mean(axis=0)
    
    print(f"\n  μ_E (media dei delta):")
    print(f"    μ_E = {mu_E}")
    print(f"    ||μ_E|| = {np.linalg.norm(mu_E):.4f}")
    
    # Direzione ottimale: da anchor verso origine (in spazio normalizzato, origine = 0.5)
    origin_Xn = np.full(d, 0.5)  # centro dello spazio [0,1]^d
    optimal_dir = origin_Xn - anchor_Xn
    optimal_dir_norm = optimal_dir / (np.linalg.norm(optimal_dir) + 1e-9)
    
    # Ma per Sphere, l'ottimo è a (0,0,...,0), che in [0,1] corrisponde a 0.5
    # Però i bounds sono [-5, 5], quindi l'ottimo (0,...,0) corrisponde a 0.5 in [0,1]
    
    # Cosine tra μ_E e direzione ottimale
    mu_E_norm = mu_E / (np.linalg.norm(mu_E) + 1e-9)
    cosine = np.dot(mu_E_norm, optimal_dir_norm)
    
    print(f"\n  Confronto con direzione ottimale:")
    print(f"    Direzione ottimale (verso centro [0,0]) = {optimal_dir_norm}")
    print(f"    μ_E normalizzato = {mu_E_norm}")
    print(f"    Cosine(μ_E, optimal) = {cosine:.4f}")
    
    if cosine > 0.5:
        print("    ✓ μ_E punta grossomodo verso l'ottimo")
    elif cosine > 0:
        print("    ⚠ μ_E punta debolmente verso l'ottimo")
    else:
        print("    ✗ μ_E punta LONTANO dall'ottimo!")
    
    # Alternativa: direzione verso il centroide degli elite
    centroid_elite = Xn_elite.mean(axis=0)
    dir_to_centroid = centroid_elite - anchor_Xn
    dir_to_centroid_norm = dir_to_centroid / (np.linalg.norm(dir_to_centroid) + 1e-9)
    
    cosine_centroid = np.dot(mu_E_norm, dir_to_centroid_norm)
    print(f"\n  μ_E vs Centroide Elite:")
    print(f"    Cosine(μ_E, verso_centroide) = {cosine_centroid:.4f}")
    print(f"    (Se ~1, μ_E punta verso il centroide, non verso l'ottimo)")
    
    print()


# =============================================================================
# TEST B: Evoluzione di μ_E nel tempo
# =============================================================================
def test_mu_stability():
    """
    μ_E cambia molto tra iterazioni? È stabile?
    """
    print("=" * 70)
    print("TEST B: Stabilità di μ_E nel tempo")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    
    opt = ECFS(bounds, seed=42)
    
    mu_history = []
    cosine_with_optimal_history = []
    
    for i in range(100):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        
        # Ogni 10 iterazioni, calcola μ_E
        if i >= 20 and (i+1) % 10 == 0:
            tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
            elite_mask = opt.y_hist <= tau
            
            if elite_mask.sum() >= 3:
                anchor_Xn = (opt.best_x - opt.lower) / opt._range
                Xn_elite = opt.Xn_hist[elite_mask]
                deltas = Xn_elite - anchor_Xn
                mu_E = deltas.mean(axis=0)
                
                mu_history.append(mu_E.copy())
                
                # Direzione ottimale
                origin_Xn = np.full(d, 0.5)
                optimal_dir = origin_Xn - anchor_Xn
                optimal_dir_norm = optimal_dir / (np.linalg.norm(optimal_dir) + 1e-9)
                mu_E_norm = mu_E / (np.linalg.norm(mu_E) + 1e-9)
                cosine = np.dot(mu_E_norm, optimal_dir_norm)
                cosine_with_optimal_history.append(cosine)
    
    print(f"\n  Evoluzione Cosine(μ_E, optimal) ogni 10 iter:")
    for i, cos in enumerate(cosine_with_optimal_history):
        iter_num = 20 + (i+1)*10
        bar = "█" * int(abs(cos) * 20) if cos > 0 else "░" * int(abs(cos) * 20)
        sign = "+" if cos > 0 else "-"
        print(f"    iter {iter_num:3d}: {cos:+.3f} {sign}{bar}")
    
    # Stabilità: quanto cambia μ_E tra iterazioni consecutive?
    if len(mu_history) >= 2:
        changes = []
        for i in range(1, len(mu_history)):
            diff = np.linalg.norm(mu_history[i] - mu_history[i-1])
            changes.append(diff)
        
        print(f"\n  Variazione ||μ_E(t) - μ_E(t-1)||:")
        print(f"    Mean: {np.mean(changes):.4f}")
        print(f"    Std:  {np.std(changes):.4f}")
        print(f"    Max:  {np.max(changes):.4f}")
    
    print()


# =============================================================================
# TEST C: Alternative a μ_E
# =============================================================================
def test_mu_alternatives():
    """
    Confronta μ_E con alternative:
    1. μ_E = mean(delta)  [attuale]
    2. μ_best = direzione verso il best globale
    3. μ_median = median(delta)
    4. μ_weighted = weighted mean by rank
    """
    print("=" * 70)
    print("TEST C: Alternative per μ")
    print("=" * 70)
    
    d = 10
    bounds = [(-5.0, 5.0)] * d
    budget = 200
    n_seeds = 5
    
    def run_with_custom_mu(bounds, budget, seed, mu_strategy):
        """Simula ECFS con una strategia μ custom."""
        opt = ECFS(bounds, seed=seed, mu_zero=True)  # Usiamo mu_zero come base
        
        for i in range(budget):
            # Fase exploration (usa ECFS standard)
            x = opt.ask()
            y = sphere(x)
            opt.tell(x, y)
        
        return opt.best_y
    
    # Test baseline
    print("\n  Baseline (ECFS con diverse configurazioni):")
    
    configs = [
        ("Default (μ=mean)", {"mu_zero": False, "use_ratio": True}),
        ("μ=0 only", {"mu_zero": True, "use_ratio": True}),
        ("μ=0, no ratio", {"mu_zero": True, "use_ratio": False}),
    ]
    
    for name, kwargs in configs:
        scores = []
        for seed in range(n_seeds):
            opt = ECFS(bounds, seed=seed, **kwargs)
            for _ in range(budget):
                x = opt.ask()
                y = sphere(x)
                opt.tell(x, y)
            scores.append(opt.best_y)
        
        print(f"    {name:25s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print()


# =============================================================================
# TEST D: Visualizza la geometria
# =============================================================================
def test_geometry_2d():
    """
    In 2D, visualizza dove sono elite, non-elite, anchor, e μ_E.
    """
    print("=" * 70)
    print("TEST D: Geometria in 2D (visualizzazione testuale)")
    print("=" * 70)
    
    d = 2
    bounds = [(-5.0, 5.0)] * d
    
    opt = ECFS(bounds, seed=42)
    
    # Warmup
    for i in range(40):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    # Analisi
    tau = float(np.quantile(opt.y_hist, opt.cfg.gamma))
    elite_mask = opt.y_hist <= tau
    
    X = np.array([opt.lower + xn * opt._range for xn in opt.Xn_hist])
    X_elite = X[elite_mask]
    X_nonelite = X[~elite_mask]
    
    print(f"\n  Statistiche:")
    print(f"    N elite: {elite_mask.sum()}, N nonelite: {(~elite_mask).sum()}")
    print(f"    Anchor (best): {opt.best_x}")
    print(f"    Optimum reale: [0, 0]")
    
    print(f"\n  Posizioni Elite (x, y, score):")
    for xe, ye in zip(X_elite, opt.y_hist[elite_mask]):
        dist_to_opt = np.linalg.norm(xe)
        print(f"    ({xe[0]:+6.2f}, {xe[1]:+6.2f}) | y={ye:7.2f} | dist_opt={dist_to_opt:.2f}")
    
    # Centroide elite
    centroid = X_elite.mean(axis=0)
    print(f"\n  Centroide Elite: ({centroid[0]:+.2f}, {centroid[1]:+.2f})")
    print(f"  Dist(centroide, ottimo): {np.linalg.norm(centroid):.2f}")
    print(f"  Dist(anchor, ottimo): {np.linalg.norm(opt.best_x):.2f}")
    
    # μ_E
    anchor_Xn = (opt.best_x - opt.lower) / opt._range
    Xn_elite = opt.Xn_hist[elite_mask]
    deltas = Xn_elite - anchor_Xn
    mu_E_Xn = deltas.mean(axis=0)
    mu_E = mu_E_Xn * opt._range  # denormalizza
    
    print(f"\n  μ_E (in coordinate originali): ({mu_E[0]:+.3f}, {mu_E[1]:+.3f})")
    
    # Se partiamo da anchor e aggiungiamo μ_E, dove arriviamo?
    destination = opt.best_x + mu_E
    print(f"  anchor + μ_E = ({destination[0]:+.2f}, {destination[1]:+.2f})")
    print(f"  Dist(anchor + μ_E, ottimo): {np.linalg.norm(destination):.2f}")
    
    # Confronto
    if np.linalg.norm(destination) < np.linalg.norm(opt.best_x):
        print("  ✓ μ_E porta PIÙ VICINO all'ottimo")
    else:
        print("  ✗ μ_E porta PIÙ LONTANO dall'ottimo!")
    
    print()


# =============================================================================
# TEST E: Il problema fondamentale
# =============================================================================
def test_fundamental_issue():
    """
    Dimostra il problema: μ_E = mean(X_elite - anchor) punta verso il CENTROIDE
    degli elite, non verso l'ottimo.
    
    Questo è corretto solo se il centroide degli elite è più vicino all'ottimo
    dell'anchor. Ma non è sempre vero!
    """
    print("=" * 70)
    print("TEST E: Il Problema Fondamentale")
    print("=" * 70)
    
    d = 10
    rng = np.random.default_rng(42)
    
    # Simula una situazione realistica
    n_points = 100
    
    # Punti random in [-5, 5]^d
    X = rng.uniform(-5, 5, (n_points, d))
    y = np.array([sphere(x) for x in X])
    
    # Elite = top 20%
    tau = np.quantile(y, 0.2)
    elite_mask = y <= tau
    X_elite = X[elite_mask]
    y_elite = y[elite_mask]
    
    # Anchor = best point
    best_idx = np.argmin(y)
    anchor = X[best_idx]
    
    print(f"\n  Setup:")
    print(f"    N punti: {n_points}")
    print(f"    N elite: {elite_mask.sum()}")
    print(f"    Anchor (best): ||x|| = {np.linalg.norm(anchor):.4f}, y = {y[best_idx]:.4f}")
    
    # Centroide elite
    centroid = X_elite.mean(axis=0)
    print(f"    Centroide elite: ||x|| = {np.linalg.norm(centroid):.4f}")
    
    # μ_E = mean(X_elite - anchor) = centroid - anchor
    mu_E = X_elite.mean(axis=0) - anchor
    # Quindi mu_E punta da anchor verso centroid!
    
    print(f"\n  Analisi μ_E:")
    print(f"    μ_E = centroid - anchor")
    print(f"    ||μ_E|| = {np.linalg.norm(mu_E):.4f}")
    
    # Dove porta μ_E?
    destination = anchor + mu_E  # = centroid!
    print(f"    anchor + μ_E = centroide (per costruzione)")
    
    # Il problema: centroide è meglio o peggio di anchor?
    print(f"\n  Confronto:")
    print(f"    ||anchor||    = {np.linalg.norm(anchor):.4f}")
    print(f"    ||centroide|| = {np.linalg.norm(centroid):.4f}")
    
    if np.linalg.norm(centroid) < np.linalg.norm(anchor):
        print("    ✓ Centroide più vicino all'ottimo → μ_E aiuta")
    else:
        print("    ✗ Centroide PIÙ LONTANO dall'ottimo → μ_E fa MALE!")
    
    # La probabilità che questo succeda
    print(f"\n  Test statistico (100 simulazioni):")
    helps = 0
    hurts = 0
    
    for seed in range(100):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-5, 5, (100, d))
        y = np.array([sphere(x) for x in X])
        tau = np.quantile(y, 0.2)
        elite_mask = y <= tau
        
        best_idx = np.argmin(y)
        anchor = X[best_idx]
        centroid = X[elite_mask].mean(axis=0)
        
        if np.linalg.norm(centroid) < np.linalg.norm(anchor):
            helps += 1
        else:
            hurts += 1
    
    print(f"    μ_E aiuta: {helps}%")
    print(f"    μ_E fa male: {hurts}%")
    
    print()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEBUG μ_E: Perché la media dei delta non funziona?")
    print("=" * 70 + "\n")
    
    test_mu_direction()
    test_mu_stability()
    test_mu_alternatives()
    test_geometry_2d()
    test_fundamental_issue()
    
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
