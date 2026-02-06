#!/usr/bin/env python3
"""
Verifica ipotesi: Cov ha varianza troppo bassa e intrappola la ricerca

Il problema:
- Cov campiona in una regione ristretta (bassa varianza)
- Se quella regione NON contiene l'ottimo, non può uscirne
- Gaussian ha alta varianza, occasionalmente trova regioni migliori

Soluzione possibile:
- Aumentare la scala quando non ci sono miglioramenti
- Mixare Cov con Gaussian (esplorazione)
- Fallback a Gaussian se stagnazione
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

np.set_printoptions(precision=4, suppress=True)

def rosenbrock(x):
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


def compare_exploration_radius():
    """Confronta la 'copertura' di Cov vs Gaussian."""
    print("="*70)
    print("ANALISI: Copertura spaziale Cov vs Gaussian")
    print("="*70)
    
    # Situazione tipica: best_x lontano da ottimo
    best_x = np.array([-1.279, -0.8393, 0.8286])
    opt = np.array([1, 1, 1])
    
    # Covarianza tipica dal seed 3
    C = np.array([
        [1.66, -0.78, 1.38],
        [-0.78, 0.64, -0.57],
        [1.38, -0.57, 4.31]
    ])
    
    rng = np.random.default_rng(42)
    n_samples = 1000
    
    # Scale come nel codice
    cov_scale = 0.15 * 3.0  # = 0.45
    gauss_radius = 0.15
    global_widths = np.array([15.0, 15.0, 15.0])
    
    # Campioni Cov
    cov_samples = []
    for _ in range(n_samples):
        z = rng.multivariate_normal(np.zeros(3), C)
        x = best_x + z * cov_scale
        x = np.clip(x, -5, 10)
        cov_samples.append(x)
    cov_samples = np.array(cov_samples)
    
    # Campioni Gaussian
    rng2 = np.random.default_rng(42)
    gauss_samples = []
    for _ in range(n_samples):
        noise = rng2.normal(0, gauss_radius, 3) * global_widths
        x = best_x + noise
        x = np.clip(x, -5, 10)
        gauss_samples.append(x)
    gauss_samples = np.array(gauss_samples)
    
    # Analisi copertura
    print("\n--- Volume esplorato ---")
    cov_std = np.std(cov_samples, axis=0)
    gauss_std = np.std(gauss_samples, axis=0)
    
    print(f"Cov:   std per dim = {cov_std}")
    print(f"Gauss: std per dim = {gauss_std}")
    print(f"Cov volume (prod std): {np.prod(cov_std):.4f}")
    print(f"Gauss volume (prod std): {np.prod(gauss_std):.4f}")
    print(f"Ratio Gauss/Cov: {np.prod(gauss_std)/np.prod(cov_std):.1f}x")
    
    # Quanti campioni coprono la regione dell'ottimo?
    print("\n--- Campioni vicini all'ottimo (dist < 1.5) ---")
    cov_close = np.sum(np.linalg.norm(cov_samples - opt, axis=1) < 1.5)
    gauss_close = np.sum(np.linalg.norm(gauss_samples - opt, axis=1) < 1.5)
    
    print(f"Cov:   {cov_close}/1000 ({cov_close/10:.1f}%)")
    print(f"Gauss: {gauss_close}/1000 ({gauss_close/10:.1f}%)")
    
    # Distribuzione delle distanze
    cov_dists = np.linalg.norm(cov_samples - opt, axis=1)
    gauss_dists = np.linalg.norm(gauss_samples - opt, axis=1)
    
    print("\n--- Distribuzione distanze da ottimo ---")
    print(f"Cov:   min={cov_dists.min():.2f}, mean={cov_dists.mean():.2f}, max={cov_dists.max():.2f}")
    print(f"Gauss: min={gauss_dists.min():.2f}, mean={gauss_dists.mean():.2f}, max={gauss_dists.max():.2f}")
    
    return cov_samples, gauss_samples


def test_stagnation_detection():
    """Simula cosa succede con stagnazione."""
    print("\n" + "="*70)
    print("SIMULAZIONE STAGNAZIONE")
    print("="*70)
    
    best_x = np.array([-1.279, -0.8393, 0.8286])
    best_y = 622.77
    
    C = np.array([
        [1.66, -0.78, 1.38],
        [-0.78, 0.64, -0.57],
        [1.38, -0.57, 4.31]
    ])
    
    rng = np.random.default_rng(42)
    cov_scale = 0.15 * 3.0
    
    # Simula 30 iterazioni di local search con Cov
    print("\n--- 30 campioni Cov consecutivi ---")
    improvements = 0
    for i in range(30):
        z = rng.multivariate_normal(np.zeros(3), C)
        x = best_x + z * cov_scale
        x = np.clip(x, -5, 10)
        y = rosenbrock(x)
        
        if y < best_y:
            improvements += 1
            best_y = y
            best_x = x
            print(f"  Iter {i+1}: IMPROVEMENT! y = {y:.2f}")
    
    print(f"\nMiglioramenti: {improvements}/30")
    print(f"Finale: y = {best_y:.2f}, x = {best_x}")
    
    # Ora simula con Gaussian
    print("\n--- 30 campioni Gaussian consecutivi ---")
    best_x = np.array([-1.279, -0.8393, 0.8286])
    best_y = 622.77
    
    rng2 = np.random.default_rng(42)
    gauss_radius = 0.15
    global_widths = np.array([15.0, 15.0, 15.0])
    
    improvements = 0
    for i in range(30):
        noise = rng2.normal(0, gauss_radius, 3) * global_widths
        x = best_x + noise
        x = np.clip(x, -5, 10)
        y = rosenbrock(x)
        
        if y < best_y:
            improvements += 1
            best_y = y
            best_x = x
            print(f"  Iter {i+1}: IMPROVEMENT! y = {y:.2f}")
    
    print(f"\nMiglioramenti: {improvements}/30")
    print(f"Finale: y = {best_y:.2f}, x = {best_x}")


def analyze_why_cov_fails_to_escape():
    """Analizza perché Cov non riesce a "scappare" dalla regione."""
    print("\n" + "="*70)
    print("PERCHÉ COV NON RIESCE A SCAPPARE?")
    print("="*70)
    
    # La covarianza è calcolata sui TOP-K punti
    # Se i top-k sono tutti in una regione sbagliata,
    # la covarianza "impara" quella regione
    
    # Simuliamo: top-k punti tutti lontani da [1,1,1]
    rng = np.random.default_rng(42)
    
    # Centro dei top-k (lontano da ottimo)
    center = np.array([-0.5, -0.5, 1.5])
    opt = np.array([1, 1, 1])
    
    # Genera top-k punti attorno al centro
    k = 15
    top_X = center + rng.normal(0, 0.3, (k, 3))
    
    # Calcola covarianza
    weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
    weights = weights / np.sum(weights)
    
    mu_w = np.average(top_X, axis=0, weights=weights)
    centered = top_X - mu_w
    C = np.dot((centered.T * weights), centered)
    C += 1e-6 * np.eye(3)
    
    print(f"Centro top-k: {center}")
    print(f"mu_w: {mu_w}")
    print(f"Ottimo: {opt}")
    print(f"Distanza mu_w da ottimo: {np.linalg.norm(mu_w - opt):.2f}")
    
    eigvals, eigvecs = np.linalg.eigh(C)
    print(f"\nAutovalori: {eigvals}")
    print(f"Direzione principale: {eigvecs[:, -1]}")
    
    # La direzione verso l'ottimo
    dir_to_opt = opt - mu_w
    dir_to_opt = dir_to_opt / np.linalg.norm(dir_to_opt)
    print(f"Direzione verso ottimo: {dir_to_opt}")
    
    # Quanto della varianza è nella direzione verso l'ottimo?
    # Proietta C sulla direzione verso l'ottimo
    var_in_opt_dir = np.dot(dir_to_opt, np.dot(C, dir_to_opt))
    total_var = np.trace(C)
    
    print(f"\nVarianza totale: {total_var:.4f}")
    print(f"Varianza verso ottimo: {var_in_opt_dir:.4f}")
    print(f"Frazione: {var_in_opt_dir/total_var*100:.1f}%")
    
    if var_in_opt_dir / total_var < 0.3:
        print(f"\n⚠️ PROBLEMA: Solo {var_in_opt_dir/total_var*100:.1f}% della varianza punta verso l'ottimo!")
        print("   La covarianza 'distrae' la ricerca in direzioni inutili.")


def main():
    compare_exploration_radius()
    test_stagnation_detection()
    analyze_why_cov_fails_to_escape()
    
    print("\n" + "="*70)
    print("CONCLUSIONI")
    print("="*70)
    print("""
Il problema fondamentale:

1. COV HA BASSA VARIANZA
   - Tutti i campioni sono in una regione ristretta
   - Se quella regione non contiene l'ottimo, è intrappolata

2. LA COVARIANZA PUNTA NELLA DIREZIONE SBAGLIATA  
   - È calcolata sui top-k punti storici
   - Se quei punti sono lontani dall'ottimo, la direzione è sbagliata
   - Solo 14% della varianza punta verso l'ottimo!

3. GAUSSIAN HA ALTA VARIANZA
   - Campioni sparsi, può "saltare" in regioni migliori
   - Trova outlier molto migliori (1.83 vs 261)

SOLUZIONI POSSIBILI:
a) Aumentare la scala di Cov (ma perde il vantaggio della forma)
b) Mixare Cov + Gaussian (50/50)
c) Fallback a Gaussian dopo N iterazioni senza miglioramenti
d) Ridurre top_k_fraction per concentrarsi sui migliori
e) Usare solo la forma, non il centro (già fatto, ma scala troppo bassa)
""")


if __name__ == "__main__":
    main()
