#!/usr/bin/env python3
"""
Debug: Analisi dettagliata del seed 19 dove Cov fallisce (1047 vs 61)
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


def analyze_failure_case(seed=19):
    """Analizza perché Cov fallisce su questo seed."""
    print(f"="*70)
    print(f"ANALISI SEED {seed}: Cov=1047, Gaussian=61 (17x peggio!)")
    print(f"="*70)
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    # Traccia con Cov
    opt_cov = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=CovarianceLocalSearchSampler(),
        use_drilling=False,
        seed=seed
    )
    
    history = []
    
    for i in range(budget):
        x = opt_cov.ask()
        y = rosenbrock(x)
        opt_cov.tell(x, y)
        
        # Traccia dettagli a ogni step
        history.append({
            'iter': i,
            'x': np.array(x),
            'y': y,
            'best_y': opt_cov.best_y,
            'best_x': opt_cov.best_x.copy()
        })
    
    print("\nCONVERGENZA:")
    for step in [10, 20, 30, 50, 70, 100]:
        if step <= len(history):
            best = min(h['y'] for h in history[:step])
            best_x = history[step-1]['best_x']
            print(f"  Step {step:3d}: best_y={best:10.2f}, best_x={best_x}")
    
    print(f"\nFINALE: best_y={opt_cov.best_y:.2f}, best_x={opt_cov.best_x}")
    print(f"OTTIMO: y=0, x=[1,1,1]")
    print(f"Distanza da ottimo: {np.linalg.norm(opt_cov.best_x - np.array([1,1,1])):.4f}")
    
    # Analizza la covarianza appresa
    print("\n" + "-"*70)
    print("ANALISI COVARIANZA")
    print("-"*70)
    
    X_all = opt_cov.X_all
    y_all = opt_cov.y_all
    n = len(X_all)
    dim = 3
    k = max(10, int(n * 0.15))
    
    indices = np.argsort(y_all)
    top_indices = indices[-k:][::-1]
    
    top_X = np.array([X_all[i] for i in top_indices])
    top_y = np.array([y_all[i] for i in top_indices])
    
    weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
    weights = weights / np.sum(weights)
    
    mu_w = np.average(top_X, axis=0, weights=weights)
    
    print(f"Top-k costi (lower is better): {-top_y[:5]}")
    print(f"mu_w: {mu_w}")
    print(f"Distanza mu_w da [1,1,1]: {np.linalg.norm(mu_w - np.array([1,1,1])):.4f}")
    
    # Matrice di covarianza
    centered = top_X - mu_w
    C = np.dot((centered.T * weights), centered)
    
    print(f"\nMatrice C:\n{C}")
    
    eigvals, eigvecs = np.linalg.eigh(C)
    print(f"\nAutovalori: {eigvals}")
    print(f"Condizionamento: {eigvals.max() / max(eigvals.min(), 1e-10):.2f}")
    print(f"Direzione principale (eigenvec max): {eigvecs[:, -1]}")
    
    # Dove sono i top-k punti?
    print("\n" + "-"*70)
    print("POSIZIONE DEI TOP-k PUNTI")
    print("-"*70)
    
    for i in range(min(5, k)):
        print(f"  {i+1}. x={top_X[i]}, costo={-top_y[i]:.2f}")
    
    # Centro geometrico dei top-k
    center = np.mean(top_X, axis=0)
    print(f"\nCentro geometrico top-k: {center}")
    print(f"Distanza centro da [1,1,1]: {np.linalg.norm(center - np.array([1,1,1])):.4f}")
    
    # Il problema è che il sampler "impara" una zona sbagliata?
    print("\n" + "-"*70)
    print("DIAGNOSI")
    print("-"*70)
    
    # Verifica se Cov sta cercando nella zona giusta
    opt_region = np.array([1, 1, 1])
    
    # Quanti dei top-k sono vicini all'ottimo?
    distances = [np.linalg.norm(x - opt_region) for x in top_X]
    close_to_opt = sum(1 for d in distances if d < 2.0)
    
    print(f"Punti top-k vicini a [1,1,1] (dist < 2): {close_to_opt}/{k}")
    print(f"Distanza media: {np.mean(distances):.2f}")
    print(f"Distanza min: {np.min(distances):.2f}")
    print(f"Distanza max: {np.max(distances):.2f}")
    
    if close_to_opt < k * 0.5:
        print("\n⚠️ PROBLEMA: La covarianza è stata appresa da punti LONTANI dall'ottimo!")
        print("   Questo fa sì che il sampler campioni nella zona sbagliata.")


def compare_with_gaussian(seed=19):
    """Confronta con Gaussian per capire la differenza."""
    print("\n" + "="*70)
    print(f"CONFRONTO CON GAUSSIAN (seed={seed})")
    print("="*70)
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    # Gaussian
    opt_gauss = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=GaussianLocalSearchSampler(),
        use_drilling=False,
        seed=seed
    )
    
    for _ in range(budget):
        x = opt_gauss.ask()
        y = rosenbrock(x)
        opt_gauss.tell(x, y)
    
    print(f"Gaussian: best_y={opt_gauss.best_y:.2f}, best_x={opt_gauss.best_x}")
    print(f"Distanza da [1,1,1]: {np.linalg.norm(opt_gauss.best_x - np.array([1,1,1])):.4f}")
    
    # Dove ha esplorato Gaussian?
    X_all = opt_gauss.X_all
    y_all = opt_gauss.y_all
    
    # Punti vicini all'ottimo
    distances = [np.linalg.norm(x - np.array([1,1,1])) for x in X_all]
    close = sum(1 for d in distances if d < 2.0)
    
    print(f"\nPunti esplorati vicini a [1,1,1] (dist < 2): {close}/{len(X_all)}")
    
    # I migliori punti di Gaussian
    k = 15
    indices = np.argsort(y_all)[-k:][::-1]
    top_X = np.array([X_all[i] for i in indices])
    
    print(f"\nTop-5 punti Gaussian:")
    for i in range(5):
        print(f"  x={top_X[i]}, costo={-y_all[indices[i]]:.2f}")


def main():
    analyze_failure_case(19)
    compare_with_gaussian(19)
    
    # Analizza anche seed 3 e 25
    print("\n\n")
    analyze_failure_case(3)
    compare_with_gaussian(3)


if __name__ == "__main__":
    main()
