#!/usr/bin/env python3
"""
Debug: Perché ALBA_Cov peggiora su Rosenbrock 3D rispetto a ALBA base?

Risultati benchmark:
- ALBA (Gaussian): 45.07
- ALBA_Cov: 89.79 (quasi 2x peggio!)

Ipotesi da verificare:
1. La covarianza impara una direzione sbagliata
2. Il weighted mean (mu_w) si allontana dall'ottimo
3. La covarianza è troppo "stretta" e perde esplorazione
4. Il fallback a Gaussian non scatta quando dovrebbe
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

np.set_printoptions(precision=4, suppress=True)

def rosenbrock(x):
    """Rosenbrock 3D: ottimo a [1,1,1], valore 0"""
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


def debug_single_run(use_cov=True, seed=42, verbose=True):
    """Esegue una run con logging dettagliato."""
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    if use_cov:
        sampler = CovarianceLocalSearchSampler()
        name = "ALBA_Cov"
    else:
        sampler = GaussianLocalSearchSampler()
        name = "ALBA_Gaussian"
    
    optimizer = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=sampler,
        use_drilling=False,
        seed=seed
    )
    
    # Traccia storia
    history = []
    
    for i in range(budget):
        x = optimizer.ask()
        y = rosenbrock(x)
        optimizer.tell(x, y)
        
        history.append({
            'iter': i,
            'x': np.array(x),
            'y': y,
            'best_y': optimizer.best_y if hasattr(optimizer, 'best_y') else y,
        })
    
    best_x, best_y = optimizer.best_x, optimizer.best_y
    
    if verbose:
        print(f"\n{name} (seed={seed}):")
        print(f"  Best y = {best_y:.4f}")
        print(f"  Best x = {best_x}")
        print(f"  Ottimo: x=[1,1,1], y=0")
    
    return best_y, history, optimizer


def analyze_covariance_learning():
    """Analizza cosa impara la covarianza."""
    print("="*70)
    print("ANALISI: Cosa impara CovarianceLocalSearchSampler su Rosenbrock")
    print("="*70)
    
    # Esegui con Cov
    best_y_cov, history_cov, opt_cov = debug_single_run(use_cov=True, seed=42)
    
    # Esegui con Gaussian
    best_y_gauss, history_gauss, opt_gauss = debug_single_run(use_cov=False, seed=42)
    
    print(f"\nConfronto finale:")
    print(f"  Gaussian: {best_y_gauss:.4f}")
    print(f"  Cov:      {best_y_cov:.4f}")
    print(f"  Differenza: {best_y_cov - best_y_gauss:.4f} ({'Cov peggio' if best_y_cov > best_y_gauss else 'Cov meglio'})")
    
    # Analizza convergenza
    print("\n" + "-"*70)
    print("CURVA DI CONVERGENZA")
    print("-"*70)
    
    for step in [10, 30, 50, 70, 100]:
        if step <= len(history_cov):
            y_cov = min(h['y'] for h in history_cov[:step])
            y_gauss = min(h['y'] for h in history_gauss[:step])
            print(f"  Step {step:3d}: Gaussian={y_gauss:10.2f}, Cov={y_cov:10.2f}")
    
    return history_cov, history_gauss


def debug_covariance_matrix():
    """Ispeziona la matrice di covarianza durante l'ottimizzazione."""
    print("\n" + "="*70)
    print("DEBUG: Matrice di Covarianza")
    print("="*70)
    
    # Simula manualmente cosa fa CovarianceLocalSearchSampler
    rng = np.random.default_rng(42)
    dim = 3
    bounds = [(-5.0, 10.0)] * 3
    
    # Genera punti come farebbe ALBA
    # Simula history dopo 50 iterazioni
    n_points = 50
    
    # Punti vicino all'ottimo (caso ideale)
    print("\n1. CASO IDEALE: punti vicino a [1,1,1]")
    X_ideal = [np.array([1,1,1]) + rng.normal(0, 0.5, 3) for _ in range(n_points)]
    y_ideal = [rosenbrock(x) for x in X_ideal]
    
    analyze_cov_from_history(X_ideal, y_ideal, "Ideale")
    
    # Punti lontani dall'ottimo (esplorazione iniziale)
    print("\n2. CASO REALE: punti sparsi nello spazio")
    X_real = [np.array([rng.uniform(-5, 10) for _ in range(3)]) for _ in range(n_points)]
    y_real = [rosenbrock(x) for x in X_real]
    
    analyze_cov_from_history(X_real, y_real, "Reale")
    
    # Punti lungo la valle di Rosenbrock
    print("\n3. CASO VALLE: punti lungo la parabola y=x^2")
    t = np.linspace(0.5, 1.5, n_points)
    X_valley = [np.array([ti, ti**2, ti**4]) + rng.normal(0, 0.1, 3) for ti in t]
    y_valley = [rosenbrock(x) for x in X_valley]
    
    analyze_cov_from_history(X_valley, y_valley, "Valle")


def analyze_cov_from_history(X_history, y_history, name):
    """Analizza la covarianza calcolata da una history."""
    n = len(X_history)
    dim = 3
    k = max(10, int(n * 0.15))
    
    # Stessa logica di CovarianceLocalSearchSampler
    indices = np.argsort(y_history)
    # I MIGLIORI sono quelli con y PIÙ BASSO (minimizzazione)
    # Ma il sampler assume "higher is better"!
    
    # QUESTO È IL BUG?
    # CovarianceLocalSearchSampler fa: indices[-k:] (prende gli ULTIMI = più alti)
    # Ma per minimizzazione dovremmo prendere i PRIMI (più bassi)!
    
    top_indices_wrong = indices[-k:]  # Come fa il sampler (highest)
    top_indices_correct = indices[:k]  # Come dovrebbe fare (lowest)
    
    print(f"\n  {name}:")
    print(f"    Top-k points (come fa il sampler - highest y):")
    top_y_wrong = [y_history[i] for i in top_indices_wrong]
    print(f"      y values: {np.array(top_y_wrong)[:5]}...")
    print(f"      mean y = {np.mean(top_y_wrong):.2f}")
    
    print(f"    Top-k points (corretti - lowest y):")
    top_y_correct = [y_history[i] for i in top_indices_correct]
    print(f"      y values: {np.array(top_y_correct)[:5]}...")
    print(f"      mean y = {np.mean(top_y_correct):.2f}")
    
    # Calcola mu_w per entrambi
    weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
    weights = weights / np.sum(weights)
    
    top_X_wrong = np.array([X_history[i] for i in top_indices_wrong[::-1]])
    top_X_correct = np.array([X_history[i] for i in top_indices_correct])
    
    mu_wrong = np.average(top_X_wrong, axis=0, weights=weights)
    mu_correct = np.average(top_X_correct, axis=0, weights=weights)
    
    print(f"    mu_w (sampler, wrong): {mu_wrong}")
    print(f"    mu_w (correct):        {mu_correct}")
    print(f"    Ottimo Rosenbrock:     [1.0, 1.0, 1.0]")
    
    # Distanza dall'ottimo
    opt = np.array([1.0, 1.0, 1.0])
    dist_wrong = np.linalg.norm(mu_wrong - opt)
    dist_correct = np.linalg.norm(mu_correct - opt)
    print(f"    Distanza da ottimo: wrong={dist_wrong:.2f}, correct={dist_correct:.2f}")


def check_sorting_bug():
    """Verifica se c'è un bug nel sorting."""
    print("\n" + "="*70)
    print("VERIFICA BUG SORTING")
    print("="*70)
    
    # Simula y_history dove LOWER is better (minimizzazione)
    y_history = [100, 50, 10, 5, 1]  # 1 è il migliore
    
    indices = np.argsort(y_history)
    print(f"y_history = {y_history}")
    print(f"argsort = {indices}")
    print(f"indices[-3:] (highest) = {indices[-3:]} → y = {[y_history[i] for i in indices[-3:]]}")
    print(f"indices[:3] (lowest) = {indices[:3]} → y = {[y_history[i] for i in indices[:3]]}")
    
    print("\n⚠️ IL SAMPLER FA indices[-k:] CHE PRENDE I PEGGIORI!")
    print("   Per minimizzazione dovrebbe fare indices[:k]!")


def verify_optimizer_y_storage():
    """Verifica cosa salva optimizer.y_all."""
    print("\n" + "="*70)
    print("VERIFICA: Cosa contiene y_all in ALBA?")
    print("="*70)
    
    bounds = [(-5.0, 10.0)] * 3
    optimizer = ALBA(bounds=bounds, total_budget=10, seed=42)
    
    for i in range(10):
        x = optimizer.ask()
        y = rosenbrock(x)  # y è il COSTO (lower is better)
        optimizer.tell(x, y)
    
    print(f"Primi 5 valori di optimizer.y_all: {optimizer.y_all[:5]}")
    print(f"Best y secondo optimizer: {optimizer.best_y}")
    
    # Controlla se y_all è negato o no
    # X_all contiene np.array, non dict
    raw_y = [rosenbrock(x) for x in optimizer.X_all[:5]]
    print(f"Raw rosenbrock(x) per stessi x: {raw_y[:5]}")
    
    # Verifica la relazione
    print(f"\ny_all[0] = {optimizer.y_all[0]}")
    print(f"-raw_y[0] = {-raw_y[0]}")
    
    if abs(optimizer.y_all[0] - (-raw_y[0])) < 1e-6:
        print("\n✓ y_all contiene -y (fitness, higher is better)")
        print("  Quindi indices[-k:] è CORRETTO per prendere i migliori!")
    elif abs(optimizer.y_all[0] - raw_y[0]) < 1e-6:
        print("\n⚠️ BUG: y_all contiene y raw (cost, lower is better)")
        print("  indices[-k:] prende i PEGGIORI!")
    else:
        print("\n??? Relazione non chiara")


def main():
    verify_optimizer_y_storage()
    trace_cov_vs_gaussian_detailed()
    
    print("\n" + "="*70)
    print("CONCLUSIONI")
    print("="*70)


def trace_cov_vs_gaussian_detailed():
    """Traccia dettagliata step-by-step."""
    print("\n" + "="*70)
    print("TRACCIA DETTAGLIATA: Cov vs Gaussian su Rosenbrock 3D")
    print("="*70)
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    seed = 123
    
    # Cov
    sampler_cov = CovarianceLocalSearchSampler()
    opt_cov = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=sampler_cov,
        use_drilling=False,
        seed=seed
    )
    
    # Gaussian
    sampler_gauss = GaussianLocalSearchSampler()
    opt_gauss = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=sampler_gauss,
        use_drilling=False,
        seed=seed
    )
    
    # Storia dettagliata
    history_cov = []
    history_gauss = []
    
    for i in range(budget):
        # Cov
        x_cov = opt_cov.ask()
        y_cov = rosenbrock(x_cov)
        opt_cov.tell(x_cov, y_cov)
        
        # Gauss
        x_gauss = opt_gauss.ask()
        y_gauss = rosenbrock(x_gauss)
        opt_gauss.tell(x_gauss, y_gauss)
        
        history_cov.append({'iter': i, 'x': np.array(x_cov), 'y': y_cov, 'best_y': opt_cov.best_y})
        history_gauss.append({'iter': i, 'x': np.array(x_gauss), 'y': y_gauss, 'best_y': opt_gauss.best_y})
    
    print("\nSTORIA CONVERGENZA (best_y):")
    print(f"{'Iter':>5} | {'Gauss best':>12} | {'Cov best':>12} | {'Diff':>10}")
    print("-" * 50)
    
    for step in [10, 20, 30, 50, 70, 100]:
        if step <= len(history_cov):
            best_cov = min(h['best_y'] for h in history_cov[:step])
            best_gauss = min(h['best_y'] for h in history_gauss[:step])
            diff = best_cov - best_gauss
            marker = "← Cov peggio" if diff > 0 else ("← Cov meglio" if diff < 0 else "")
            print(f"{step:5d} | {best_gauss:12.2f} | {best_cov:12.2f} | {diff:+10.2f} {marker}")
    
    print(f"\nFinale: Gaussian={opt_gauss.best_y:.4f}, Cov={opt_cov.best_y:.4f}")
    print(f"Best x Gaussian: {opt_gauss.best_x}")
    print(f"Best x Cov:      {opt_cov.best_x}")
    
    # Analisi: dove divergono?
    print("\n" + "-"*50)
    print("Dove iniziano a divergere?")
    print("-"*50)
    
    for i in range(min(budget, 50)):
        x_c = history_cov[i]['x']
        x_g = history_gauss[i]['x']
        dist = np.linalg.norm(x_c - x_g)
        if dist > 0.01:
            print(f"Iter {i}: dist={dist:.4f}")
            print(f"  x_cov={x_c}")
            print(f"  x_gauss={x_g}")
            print(f"  y_cov={history_cov[i]['y']:.2f}, y_gauss={history_gauss[i]['y']:.2f}")
            break
    
    # Analisi del mu_w calcolato dal Cov sampler
    print("\n" + "-"*50)
    print("Analisi mu_w (weighted mean) del CovarianceLocalSearchSampler")
    print("-"*50)
    
    # Ricostruisci cosa calcolerebbe il sampler
    X_all = opt_cov.X_all
    y_all = opt_cov.y_all
    n = len(X_all)
    dim = 3
    k = max(10, int(n * 0.15))
    
    print(f"Punti totali: {n}")
    print(f"Top-k usato: {k}")
    
    indices = np.argsort(y_all)
    top_indices = indices[-k:][::-1]
    
    top_X = np.array([X_all[i] for i in top_indices])
    top_y = [y_all[i] for i in top_indices]
    
    print(f"\nTop-k y_all (internal fitness, higher=better): {np.array(top_y[:5])}")
    print(f"Corrispondenti costi raw: {[-yi for yi in top_y[:5]]}")
    
    weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
    weights = weights / np.sum(weights)
    
    mu_w = np.average(top_X, axis=0, weights=weights)
    
    print(f"\nmu_w (weighted mean): {mu_w}")
    print(f"best_x (singolo migliore): {opt_cov.best_x}")
    print(f"Ottimo Rosenbrock: [1, 1, 1]")
    
    print(f"\nDistanza mu_w da ottimo: {np.linalg.norm(mu_w - np.array([1,1,1])):.4f}")
    print(f"Distanza best_x da ottimo: {np.linalg.norm(opt_cov.best_x - np.array([1,1,1])):.4f}")
    
    # Matrice di covarianza
    centered = top_X - mu_w
    C = np.dot((centered.T * weights), centered)
    
    print(f"\nMatrice di covarianza C:")
    print(C)
    
    eigvals = np.linalg.eigvalsh(C)
    print(f"\nAutovalori di C: {eigvals}")
    print(f"Condizionamento: {eigvals.max() / eigvals.min():.2f}")
    
    # Campiona qualche punto dalla covarianza
    rng = np.random.default_rng(42)
    print("\nCampioni dalla distribuzione N(mu_w, C):")
    for _ in range(5):
        z = rng.multivariate_normal(np.zeros(dim), C)
        x_sample = mu_w + z * 0.9
        x_clipped = np.clip(x_sample, -5, 10)
        y_sample = rosenbrock(x_clipped)
        print(f"  x={x_clipped}, y={y_sample:.2f}")


if __name__ == "__main__":
    main()
