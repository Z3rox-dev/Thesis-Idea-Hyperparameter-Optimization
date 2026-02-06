#!/usr/bin/env python3
"""
Debug e verifica delle matrici di covarianza CMA-like nel framework ALBA.

Due componenti usano covarianza:
1. DrillingOptimizer - (1+1)-CMA-ES per drilling locale
2. CovarianceLocalSearchSampler - sampling con weighted covariance

Questo script verifica:
- La matrice C resta positiva definita
- La matrice C si adatta alla geometria del problema
- I campioni generati sono coerenti con la covarianza
- Non ci sono NaN/Inf che corrompono lo stato
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.drilling import DrillingOptimizer
from alba_framework_potential.local_search import CovarianceLocalSearchSampler

np.set_printoptions(precision=4, suppress=True)


def check_positive_definite(C: np.ndarray, name: str = "C") -> dict:
    """Verifica che C sia positiva definita."""
    result = {
        "name": name,
        "shape": C.shape,
        "symmetric": np.allclose(C, C.T),
        "has_nan": np.any(np.isnan(C)),
        "has_inf": np.any(np.isinf(C)),
        "eigenvalues": None,
        "is_pd": False,
        "condition_number": None,
    }
    
    if result["has_nan"] or result["has_inf"]:
        return result
    
    try:
        eigvals = np.linalg.eigvalsh(C)
        result["eigenvalues"] = eigvals
        result["is_pd"] = np.all(eigvals > 0)
        result["min_eig"] = eigvals.min()
        result["max_eig"] = eigvals.max()
        result["condition_number"] = eigvals.max() / max(eigvals.min(), 1e-15)
    except Exception as e:
        result["error"] = str(e)
    
    return result


def print_cov_status(result: dict):
    """Stampa lo stato della matrice."""
    name = result["name"]
    print(f"  {name}:")
    print(f"    Shape: {result['shape']}")
    print(f"    Symmetric: {result['symmetric']}")
    print(f"    Has NaN: {result['has_nan']}")
    print(f"    Has Inf: {result['has_inf']}")
    
    if result.get("eigenvalues") is not None:
        print(f"    Eigenvalues: min={result['min_eig']:.2e}, max={result['max_eig']:.2e}")
        print(f"    Positive Definite: {result['is_pd']}")
        print(f"    Condition Number: {result['condition_number']:.2e}")
    
    if result.get("error"):
        print(f"    ERROR: {result['error']}")


def test_drilling_covariance_evolution():
    """
    Test 1: Verifica che DrillingOptimizer aggiorni C correttamente.
    
    Scenari:
    - Funzione sferica: C dovrebbe restare ~ identità
    - Funzione ellittica: C dovrebbe allungarsi lungo l'asse dominante
    """
    print("\n" + "="*70)
    print("TEST 1: DrillingOptimizer - Evoluzione della Covarianza")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    # === TEST 1.1: Funzione sferica ===
    print("\n1.1 Funzione Sferica (f = sum(x^2))")
    print("    Atteso: C ~ I (isotropica)")
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    start_x = np.array([0.6] * dim)
    start_y = sphere(start_x)
    
    drill = DrillingOptimizer(start_x, start_y, initial_sigma=0.1, bounds=bounds)
    
    print(f"    C iniziale:")
    print_cov_status(check_positive_definite(drill.C, "C_init"))
    
    # Esegui drilling
    n_success = 0
    n_fail = 0
    for i in range(30):
        x_new = drill.ask(rng)
        y_new = sphere(x_new)
        should_continue = drill.tell(x_new, y_new)
        
        if y_new < start_y:
            n_success += 1
        else:
            n_fail += 1
        
        if not should_continue:
            break
    
    print(f"    Dopo {i+1} iterazioni (success={n_success}, fail={n_fail}):")
    print_cov_status(check_positive_definite(drill.C, "C_final"))
    
    # Verifica isotropia per sfera
    eigvals = np.linalg.eigvalsh(drill.C)
    anisotropy = eigvals.max() / max(eigvals.min(), 1e-10)
    print(f"    Anisotropia (max/min eig): {anisotropy:.2f}")
    
    if anisotropy < 10:
        print("    ✓ C resta ragionevolmente isotropica per sfera")
    else:
        print("    ⚠ C è diventata anisotropa (potrebbe essere OK se convergenza rapida)")
    
    # === TEST 1.2: Funzione ellittica ===
    print("\n1.2 Funzione Ellittica (f = sum(i * x_i^2))")
    print("    Atteso: C si allunga lungo le direzioni 'facili'")
    
    def elliptic(x):
        weights = np.arange(1, dim + 1)
        return np.sum(weights * (x - 0.5)**2)
    
    start_x = np.array([0.6] * dim)
    start_y = elliptic(start_x)
    
    drill = DrillingOptimizer(start_x, start_y, initial_sigma=0.1, bounds=bounds)
    
    for i in range(50):
        x_new = drill.ask(rng)
        y_new = elliptic(x_new)
        should_continue = drill.tell(x_new, y_new)
        if not should_continue:
            break
    
    print(f"    Dopo {i+1} iterazioni:")
    print_cov_status(check_positive_definite(drill.C, "C_final"))
    print(f"    Diagonale di C: {np.diag(drill.C)}")
    print("    (Atteso: valori più grandi per indici piccoli, dove il gradiente è minore)")
    
    # === TEST 1.3: Stress test - molti successi consecutivi ===
    print("\n1.3 Stress Test: 100 successi consecutivi")
    print("    Rischio: C esplode, sigma esplode")
    
    drill = DrillingOptimizer(np.array([0.5]*dim), 1.0, initial_sigma=0.1, bounds=bounds)
    
    # Forza successi artificiali
    for i in range(100):
        x_new = drill.ask(rng)
        # Forza successo: y sempre migliore
        y_new = drill.best_y - 0.01
        should_continue = drill.tell(x_new, y_new)
        if not should_continue:
            break
    
    print(f"    Dopo {i+1} successi forzati:")
    print(f"    sigma = {drill.sigma:.2e}")
    result = check_positive_definite(drill.C, "C_exploded?")
    print_cov_status(result)
    
    if drill.sigma > 1e10:
        print("    ⚠ BUG: sigma è esploso!")
    else:
        print("    ✓ sigma sotto controllo")
    
    if result["condition_number"] and result["condition_number"] > 1e10:
        print("    ⚠ BUG: C mal condizionata!")
    else:
        print("    ✓ C ben condizionata")


def test_covariance_local_search_geometry():
    """
    Test 2: Verifica che CovarianceLocalSearchSampler impari la geometria.
    
    Scenari:
    - Cloud sferico: campioni sferici
    - Cloud ellittico: campioni ellittici
    - Cloud con correlazione: campioni correlati
    """
    print("\n" + "="*70)
    print("TEST 2: CovarianceLocalSearchSampler - Geometria Appresa")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    global_widths = np.ones(dim)
    
    sampler = CovarianceLocalSearchSampler()
    
    # === TEST 2.1: Cloud sferico ===
    print("\n2.1 Cloud Sferico (punti uniformi intorno al centro)")
    print("    Atteso: covarianza ~ I")
    
    n_points = 50
    center = np.array([0.5] * dim)
    X_history = [center + rng.normal(0, 0.1, dim) for _ in range(n_points)]
    y_history = [-np.sum((x - center)**2) for x in X_history]  # Higher is better
    
    # Calcola covarianza interna (simulando cosa fa il sampler)
    k = max(10, int(n_points * 0.15))
    indices = np.argsort(y_history)[-k:][::-1]
    top_X = np.array([X_history[i] for i in indices])
    weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
    weights = weights / np.sum(weights)
    mu_w = np.average(top_X, axis=0, weights=weights)
    centered = top_X - mu_w
    C = np.dot((centered.T * weights), centered)
    C += 1e-6 * np.eye(dim)
    
    print_cov_status(check_positive_definite(C, "C_interno"))
    
    # Genera campioni
    samples = [sampler.sample(center, bounds, global_widths, 0.5, rng, X_history, y_history) 
               for _ in range(100)]
    samples = np.array(samples)
    
    sample_cov = np.cov(samples.T)
    print_cov_status(check_positive_definite(sample_cov, "Cov(samples)"))
    
    # Verifica isotropia
    eigvals = np.linalg.eigvalsh(sample_cov)
    anisotropy = eigvals.max() / max(eigvals.min(), 1e-10)
    print(f"    Anisotropia campioni: {anisotropy:.2f}")
    
    # === TEST 2.2: Cloud ellittico ===
    print("\n2.2 Cloud Ellittico (stretching lungo dim 0)")
    print("    Atteso: covarianza allungata lungo dim 0")
    
    # Crea cloud allungato
    X_history = []
    for _ in range(n_points):
        x = center.copy()
        x[0] += rng.normal(0, 0.3)  # Stretching lungo dim 0
        x[1:] += rng.normal(0, 0.05, dim-1)
        X_history.append(x)
    y_history = [-np.sum((x - center)**2) for x in X_history]
    
    # Genera campioni
    samples = [sampler.sample(center, bounds, global_widths, 0.5, rng, X_history, y_history) 
               for _ in range(100)]
    samples = np.array(samples)
    
    sample_cov = np.cov(samples.T)
    print(f"    Diagonale Cov(samples): {np.diag(sample_cov)}")
    print(f"    Atteso: dim[0] >> dim[1:4]")
    
    if sample_cov[0,0] > 2 * np.mean(np.diag(sample_cov)[1:]):
        print("    ✓ Sampler ha imparato la geometria ellittica!")
    else:
        print("    ⚠ Sampler non ha catturato lo stretching")
    
    # === TEST 2.3: Cloud con correlazione ===
    print("\n2.3 Cloud con Correlazione (dim0 ~ dim1)")
    print("    Atteso: covarianza con off-diagonal non nulla")
    
    X_history = []
    for _ in range(n_points):
        t = rng.normal(0, 0.2)
        x = center.copy()
        x[0] += t
        x[1] += 0.8 * t + rng.normal(0, 0.02)  # Correlato con dim0
        x[2:] += rng.normal(0, 0.05, dim-2)
        X_history.append(x)
    y_history = [-np.sum((x - center)**2) for x in X_history]
    
    samples = [sampler.sample(center, bounds, global_widths, 0.5, rng, X_history, y_history) 
               for _ in range(100)]
    samples = np.array(samples)
    
    sample_cov = np.cov(samples.T)
    correlation = sample_cov[0,1] / np.sqrt(sample_cov[0,0] * sample_cov[1,1])
    print(f"    Correlazione campioni[0,1]: {correlation:.3f}")
    print(f"    Atteso: ~0.8 (alta correlazione positiva)")
    
    if correlation > 0.5:
        print("    ✓ Sampler ha catturato la correlazione!")
    elif correlation > 0.2:
        print("    ~ Sampler ha catturato parzialmente la correlazione")
    else:
        print("    ⚠ Sampler non ha catturato la correlazione")


def test_edge_cases():
    """
    Test 3: Edge cases che potrebbero rompere la covarianza.
    """
    print("\n" + "="*70)
    print("TEST 3: Edge Cases")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    # === TEST 3.1: Punti collineari ===
    print("\n3.1 Punti Collineari (covarianza rank-deficient)")
    
    sampler = CovarianceLocalSearchSampler()
    center = np.array([0.5] * dim)
    global_widths = np.ones(dim)
    
    # Punti tutti sulla stessa linea
    X_history = [center + t * np.array([1,0,0,0,0]) for t in np.linspace(-0.2, 0.2, 20)]
    y_history = [-abs(t) for t in np.linspace(-0.2, 0.2, 20)]  # Best at center
    
    try:
        samples = [sampler.sample(center, bounds, global_widths, 0.5, rng, X_history, y_history) 
                   for _ in range(10)]
        samples = np.array(samples)
        
        if np.any(np.isnan(samples)):
            print("    ⚠ BUG: NaN nei campioni con punti collineari!")
        else:
            print("    ✓ Gestisce punti collineari senza crash")
            print(f"    Varianza campioni: {np.var(samples, axis=0)}")
    except Exception as e:
        print(f"    ⚠ CRASH: {e}")
    
    # === TEST 3.2: Punti identici ===
    print("\n3.2 Punti Identici (covarianza zero)")
    
    X_history = [center.copy() for _ in range(20)]
    y_history = [1.0] * 20
    
    try:
        samples = [sampler.sample(center, bounds, global_widths, 0.5, rng, X_history, y_history) 
                   for _ in range(10)]
        samples = np.array(samples)
        
        if np.any(np.isnan(samples)):
            print("    ⚠ BUG: NaN nei campioni con punti identici!")
        else:
            print("    ✓ Gestisce punti identici senza crash")
    except Exception as e:
        print(f"    ⚠ CRASH: {e}")
    
    # === TEST 3.3: NaN in history ===
    print("\n3.3 NaN in X_history")
    
    X_history = [center + rng.normal(0, 0.1, dim) for _ in range(20)]
    X_history[5] = np.array([np.nan] * dim)  # Corrupt one point
    y_history = [1.0] * 20
    
    try:
        samples = [sampler.sample(center, bounds, global_widths, 0.5, rng, X_history, y_history) 
                   for _ in range(10)]
        samples = np.array(samples)
        
        if np.any(np.isnan(samples)):
            print("    ⚠ BUG: NaN propagato ai campioni!")
        else:
            print("    ✓ NaN in history non propaga")
    except Exception as e:
        print(f"    ⚠ CRASH: {e}")
    
    # === TEST 3.4: Drilling con start NaN ===
    print("\n3.4 DrillingOptimizer con start_x contenente NaN")
    
    try:
        drill = DrillingOptimizer(
            start_x=np.array([0.5, np.nan, 0.5, 0.5, 0.5]),
            start_y=1.0,
            initial_sigma=0.1,
            bounds=bounds
        )
        
        if np.any(np.isnan(drill.mu)):
            print("    ⚠ BUG: NaN in mu dopo init!")
        else:
            print(f"    ✓ NaN sanitizzato: mu = {drill.mu}")
    except Exception as e:
        print(f"    ⚠ CRASH: {e}")


def test_cma_update_correctness():
    """
    Test 4: Verifica matematica dell'update CMA.
    
    L'update corretto per (1+1)-CMA-ES è:
    C <- (1 - c_cov) * C + c_cov * y * y^T
    
    dove y = (x_new - x_old) / sigma
    """
    print("\n" + "="*70)
    print("TEST 4: Correttezza Update CMA")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 3
    bounds = [(0, 1)] * dim
    
    # Setup
    start_x = np.array([0.5, 0.5, 0.5])
    start_y = 1.0
    
    drill = DrillingOptimizer(start_x, start_y, initial_sigma=0.1, bounds=bounds)
    
    print(f"C iniziale:\n{drill.C}")
    print(f"c_cov = {drill.c_cov:.4f}")
    
    # Simula un successo manuale
    C_old = drill.C.copy()
    sigma_old = drill.sigma
    
    x_new = drill.ask(rng)
    y_mutation_stored = drill.last_y_mutation.copy()
    
    # Forza successo
    y_new = start_y - 0.1
    drill.tell(x_new, y_new)
    
    C_new = drill.C
    
    # Verifica update
    # C_new dovrebbe essere: (1 - c_cov) * C_old + c_cov * outer(y, y)
    C_expected = (1 - drill.c_cov) * C_old + drill.c_cov * np.outer(y_mutation_stored, y_mutation_stored)
    
    print(f"\nDopo successo:")
    print(f"y_mutation = {y_mutation_stored}")
    print(f"C atteso:\n{C_expected}")
    print(f"C ottenuto:\n{C_new}")
    
    if np.allclose(C_expected, C_new, rtol=1e-5):
        print("\n✓ Update CMA matematicamente corretto!")
    else:
        diff = np.abs(C_expected - C_new).max()
        print(f"\n⚠ Discrepanza: max_diff = {diff:.2e}")
        print("  Possibile bug nell'update della covarianza")


def test_sampling_distribution():
    """
    Test 5: Verifica che i campioni seguano la distribuzione attesa.
    
    Se C è la covarianza, i campioni x ~ N(mu, sigma^2 * C) dovrebbero
    avere covarianza empirica ≈ sigma^2 * C.
    """
    print("\n" + "="*70)
    print("TEST 5: Distribuzione dei Campioni")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 3
    bounds = [(-10, 10)] * dim  # Bounds larghi per non clippare
    
    # Setup con C non-identità
    drill = DrillingOptimizer(
        start_x=np.zeros(dim),
        start_y=1.0,
        initial_sigma=1.0,
        bounds=bounds
    )
    
    # Imposta C manualmente per test
    drill.C = np.array([
        [2.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ])
    
    print(f"C impostata:\n{drill.C}")
    print(f"sigma = {drill.sigma}")
    
    # Genera molti campioni
    n_samples = 5000
    samples = []
    for _ in range(n_samples):
        # Reset drill per ogni sample (vogliamo solo testare ask())
        drill.mu = np.zeros(dim)
        x = drill.ask(rng)
        samples.append(x)
    
    samples = np.array(samples)
    
    # Covarianza empirica
    emp_cov = np.cov(samples.T)
    
    # Covarianza attesa: sigma^2 * C
    expected_cov = drill.sigma**2 * drill.C
    
    print(f"\nCovarianza attesa (sigma^2 * C):\n{expected_cov}")
    print(f"Covarianza empirica:\n{emp_cov}")
    
    # Test: rapporto dovrebbe essere ~1 per ogni elemento
    ratio = emp_cov / (expected_cov + 1e-10)
    print(f"\nRapporto (empirica/attesa):\n{ratio}")
    
    # Verifica che il rapporto sia vicino a 1 (tolleranza Monte Carlo)
    diagonal_ok = np.allclose(np.diag(ratio), 1.0, rtol=0.1)
    offdiag_ok = np.allclose(ratio[0,1], 1.0, rtol=0.2)  # Off-diagonal ha più varianza
    
    if diagonal_ok:
        print("✓ Diagonale OK (varianze corrette)")
    else:
        print("⚠ Diagonale non corretta")
    
    if offdiag_ok:
        print("✓ Off-diagonal OK (correlazioni corrette)")
    else:
        print("⚠ Off-diagonal non corretta")


def main():
    print("="*70)
    print("DEBUG MATRICI DI COVARIANZA CMA-LIKE")
    print("="*70)
    
    bugs_found = []
    
    test_drilling_covariance_evolution()
    test_covariance_local_search_geometry()
    test_edge_cases()
    test_cma_update_correctness()
    test_sampling_distribution()
    
    print("\n" + "="*70)
    print("RIEPILOGO")
    print("="*70)
    print("Test completati. Verifica i ⚠ sopra per potenziali problemi.")


if __name__ == "__main__":
    main()
