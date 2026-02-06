#!/usr/bin/env python3
"""
Analisi specifica della fase LOCAL SEARCH - dove Cov diverge
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

np.set_printoptions(precision=4, suppress=True)

def rosenbrock(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


def analyze_local_search_phase(seed=3):
    """Analizza cosa succede esattamente nella fase local search."""
    print(f"{'='*70}")
    print(f"ANALISI FASE LOCAL SEARCH - SEED {seed}")
    print(f"{'='*70}")
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    # Esegui fino alla fine della fase di esplorazione (70%)
    warmup = int(budget * 0.7)  # 70 iterazioni
    
    # Cov
    opt_cov = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                   local_search_sampler=CovarianceLocalSearchSampler(), use_drilling=False, seed=seed)
    
    # Gauss
    opt_gauss = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                     local_search_sampler=GaussianLocalSearchSampler(), use_drilling=False, seed=seed)
    
    # Fase di esplorazione
    for i in range(warmup):
        x_cov = opt_cov.ask()
        opt_cov.tell(x_cov, rosenbrock(x_cov))
        
        x_gauss = opt_gauss.ask()
        opt_gauss.tell(x_gauss, rosenbrock(x_gauss))
    
    print(f"\n--- Dopo {warmup} iter (fine esplorazione) ---")
    print(f"  Cov:   best_y = {opt_cov.best_y:.2f}, best_x = {opt_cov.best_x}")
    print(f"  Gauss: best_y = {opt_gauss.best_y:.2f}, best_x = {opt_gauss.best_x}")
    
    # Analizza la covarianza a questo punto
    X_all = opt_cov.X_all
    y_all = opt_cov.y_all
    n = len(X_all)
    dim = 3
    k = max(10, int(n * 0.15))
    
    indices = np.argsort(y_all)[-k:][::-1]
    top_X = np.array([X_all[i] for i in indices])
    top_y = np.array([y_all[i] for i in indices])
    
    weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
    weights = weights / np.sum(weights)
    
    mu_w = np.average(top_X, axis=0, weights=weights)
    
    centered = top_X - mu_w
    C = np.dot((centered.T * weights), centered)
    C += 1e-6 * np.eye(dim)
    
    eigvals, eigvecs = np.linalg.eigh(C)
    
    print(f"\n--- Covarianza al momento della local search ---")
    print(f"  mu_w = {mu_w}")
    print(f"  best_x = {opt_cov.best_x}")
    print(f"  Ottimo = [1, 1, 1]")
    print(f"  Distanza mu_w da ottimo: {np.linalg.norm(mu_w - np.array([1,1,1])):.2f}")
    print(f"  Distanza best_x da ottimo: {np.linalg.norm(opt_cov.best_x - np.array([1,1,1])):.2f}")
    
    print(f"\n  Matrice C:\n{C}")
    print(f"\n  Autovalori: {eigvals}")
    print(f"  Condizionamento: {eigvals.max() / eigvals.min():.2f}")
    
    print(f"\n  Direzione principale (eigenvec max): {eigvecs[:, -1]}")
    
    # Campiona alcuni punti dalla covarianza
    print(f"\n--- Campioni dalla distribuzione N(best_x, C * scale) ---")
    rng = np.random.default_rng(42)
    scale = 0.15 * 3.0  # Come nel codice: scale * 3.0
    
    for i in range(10):
        z = rng.multivariate_normal(np.zeros(dim), C)
        x_sample = opt_cov.best_x + z * scale
        x_clipped = np.clip(x_sample, -5, 10)
        y_sample = rosenbrock(x_clipped)
        dist_to_opt = np.linalg.norm(x_clipped - np.array([1,1,1]))
        print(f"  {i+1}. x = {x_clipped}, y = {y_sample:8.2f}, dist = {dist_to_opt:.2f}")
    
    # Confronta con Gaussian sampling
    print(f"\n--- Campioni Gaussian per confronto ---")
    radius = 0.15  # radius_start
    global_widths = np.array([15.0, 15.0, 15.0])  # bounds width
    
    for i in range(10):
        noise = rng.normal(0, radius, dim) * global_widths
        x_sample = opt_gauss.best_x + noise
        x_clipped = np.clip(x_sample, -5, 10)
        y_sample = rosenbrock(x_clipped)
        dist_to_opt = np.linalg.norm(x_clipped - np.array([1,1,1]))
        print(f"  {i+1}. x = {x_clipped}, y = {y_sample:8.2f}, dist = {dist_to_opt:.2f}")


def compare_sampling_strategies():
    """Confronta le strategie di campionamento partendo dallo stesso punto."""
    print(f"\n{'='*70}")
    print(f"CONFRONTO STRATEGIE DI CAMPIONAMENTO")
    print(f"{'='*70}")
    
    # Simula la situazione a iter 70 del seed 3
    best_x = np.array([-1.279, -0.8393, 0.8286])  # best_x a iter 70 per seed 3
    best_y = 622.77
    
    print(f"\nPunto di partenza:")
    print(f"  best_x = {best_x}")
    print(f"  best_y = {best_y}")
    print(f"  Distanza da [1,1,1]: {np.linalg.norm(best_x - np.array([1,1,1])):.2f}")
    
    # Covarianza tipica (appresa da punti sparsi)
    # Usiamo una covarianza "realistica" con direzione dominante
    C = np.array([
        [0.09, -0.01, -0.01],
        [-0.01, 0.05, -0.07],
        [-0.01, -0.07, 0.27]
    ])
    
    rng = np.random.default_rng(42)
    dim = 3
    n_samples = 100
    
    # Sampling con Cov
    cov_samples = []
    scale = 0.15 * 3.0
    for _ in range(n_samples):
        z = rng.multivariate_normal(np.zeros(dim), C)
        x = best_x + z * scale
        x = np.clip(x, -5, 10)
        cov_samples.append({'x': x, 'y': rosenbrock(x)})
    
    # Sampling Gaussian
    gauss_samples = []
    radius = 0.15
    global_widths = np.array([15.0, 15.0, 15.0])
    rng2 = np.random.default_rng(42)  # Same seed for fair comparison
    for _ in range(n_samples):
        noise = rng2.normal(0, radius, dim) * global_widths
        x = best_x + noise
        x = np.clip(x, -5, 10)
        gauss_samples.append({'x': x, 'y': rosenbrock(x)})
    
    # Statistiche
    cov_y = [s['y'] for s in cov_samples]
    gauss_y = [s['y'] for s in gauss_samples]
    
    print(f"\n--- Statistiche 100 campioni ---")
    print(f"  Cov:   mean={np.mean(cov_y):.2f}, std={np.std(cov_y):.2f}, min={np.min(cov_y):.2f}, max={np.max(cov_y):.2f}")
    print(f"  Gauss: mean={np.mean(gauss_y):.2f}, std={np.std(gauss_y):.2f}, min={np.min(gauss_y):.2f}, max={np.max(gauss_y):.2f}")
    
    # Quanti campioni sono migliori del punto di partenza?
    cov_better = sum(1 for y in cov_y if y < best_y)
    gauss_better = sum(1 for y in gauss_y if y < best_y)
    
    print(f"\n  Campioni migliori di {best_y:.2f}:")
    print(f"    Cov:   {cov_better}/100 ({cov_better}%)")
    print(f"    Gauss: {gauss_better}/100 ({gauss_better}%)")
    
    # Distanza dall'ottimo
    opt = np.array([1, 1, 1])
    cov_dist = [np.linalg.norm(s['x'] - opt) for s in cov_samples]
    gauss_dist = [np.linalg.norm(s['x'] - opt) for s in gauss_samples]
    
    print(f"\n  Distanza media da [1,1,1]:")
    print(f"    Cov:   {np.mean(cov_dist):.2f}")
    print(f"    Gauss: {np.mean(gauss_dist):.2f}")
    
    # IL PROBLEMA: La covarianza concentra i campioni in una DIREZIONE
    # che potrebbe non puntare verso l'ottimo
    print(f"\n--- Direzione della covarianza ---")
    eigvals, eigvecs = np.linalg.eigh(C)
    main_dir = eigvecs[:, -1]
    print(f"  Direzione principale: {main_dir}")
    
    # Direzione verso l'ottimo
    dir_to_opt = opt - best_x
    dir_to_opt = dir_to_opt / np.linalg.norm(dir_to_opt)
    print(f"  Direzione verso ottimo: {dir_to_opt}")
    
    # Coseno dell'angolo tra le due direzioni
    cos_angle = np.abs(np.dot(main_dir, dir_to_opt))
    print(f"  Allineamento (|cos|): {cos_angle:.2f}")
    
    if cos_angle < 0.5:
        print(f"\n⚠️ PROBLEMA: La covarianza NON è allineata con la direzione verso l'ottimo!")
        print(f"   I campioni si concentrano in una direzione sbagliata.")
    else:
        print(f"\n✓ La covarianza è ragionevolmente allineata.")


def main():
    analyze_local_search_phase(seed=3)
    compare_sampling_strategies()
    
    # Analizza anche seed 0 e 25
    print("\n\n")
    analyze_local_search_phase(seed=0)
    

if __name__ == "__main__":
    main()
