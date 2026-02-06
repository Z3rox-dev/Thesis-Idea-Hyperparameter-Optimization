#!/usr/bin/env python3
"""
Deep Investigation of Coherence Module - Mathematical Stability Analysis

Il modulo Coherence è un'invenzione originale per evitare path dependency.
L'intuizione: per un campo gradiente conservativo, l'integrale dipende solo 
dai punti estremi, non dal percorso.

IDEA MATEMATICA:
- Se f è smooth, allora: f(c_j) - f(c_i) ≈ ∇f(c_i)·(c_j - c_i) (Taylor 1° ordine)
- Se abbiamo gradienti locali g_i in ogni foglia, possiamo "ricostruire" un 
  potenziale globale u risolvendo: u_j - u_i = g_i·(c_j - c_i) per ogni edge (i,j)
- Sistema sovradeterminato → least squares
- La qualità del fit (residui) indica quanto il campo è "coerente"

POTENZIALI PROBLEMI:
1. Path dependency: l'integrale dipende dal percorso se il campo NON è conservativo
2. Inconsistenza locale: gradienti locali potrebbero essere inconsistenti tra loro
3. Sparsità del grafo: in high-D il kNN graph diventa troppo sparse
4. Gauge freedom: il potenziale è definito a meno di una costante
5. Anchoring: come scegliamo il punto di riferimento?

Questo script testa tutti questi aspetti.
"""

import sys
sys.path.insert(0, '/mnt/workspace')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import List, Tuple
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def create_mock_leaves(n_leaves: int, dim: int, seed: int = 42):
    """Crea foglie mock con gradienti sintetici."""
    from alba_framework_potential.cube import Cube
    
    rng = np.random.default_rng(seed)
    
    leaves = []
    for i in range(n_leaves):
        # Crea bounds random per il cubo
        lo = rng.uniform(-4, 2, dim)
        hi = lo + rng.uniform(0.5, 2, dim)
        hi = np.minimum(hi, 5.0)
        
        # Cube usa bounds come lista di tuple
        cube_bounds = [(float(lo[d]), float(hi[d])) for d in range(dim)]
        
        cube = Cube(bounds=cube_bounds)
        cube.n_good = int(rng.integers(3, 20))
        cube.n_bad = int(rng.integers(1, 10))
        cube.n_trials = cube.n_good + cube.n_bad
        
        leaves.append(cube)
    
    return leaves


def assign_consistent_gradients(leaves, optimum=None):
    """
    Assegna gradienti CONSISTENTI che puntano verso l'ottimo.
    Questo dovrebbe dare alta coerenza.
    """
    if optimum is None:
        optimum = np.zeros(len(leaves[0].bounds))
    
    for leaf in leaves:
        center = leaf.center()
        # Gradiente = direzione verso l'ottimo (per minimizzazione)
        # Ma matematicamente ∇f punta nella direzione di MASSIMA crescita
        # Quindi per una funzione con minimo in 0: ∇f ∝ x (punta LONTANO dall'ottimo)
        grad = center - optimum  # Questo è corretto per Sphere-like
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad = grad / grad_norm
        else:
            grad = np.zeros_like(center)
        
        leaf.lgs_model = {
            "grad": grad,
            "intercept": 0.0,
            "all_pts": [center],
        }


def assign_inconsistent_gradients(leaves, noise_level=1.0, seed=123):
    """
    Assegna gradienti INCONSISTENTI (rumore casuale).
    Questo dovrebbe dare bassa coerenza.
    """
    rng = np.random.default_rng(seed)
    dim = len(leaves[0].bounds)
    
    for leaf in leaves:
        grad = rng.standard_normal(dim) * noise_level
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad = grad / grad_norm
        else:
            grad = np.zeros(dim)
        
        leaf.lgs_model = {
            "grad": grad,
            "intercept": 0.0,
            "all_pts": [leaf.center()],
        }


def assign_partially_consistent_gradients(leaves, consistency_ratio=0.5, seed=42):
    """
    Assegna gradienti dove una frazione è consistente e il resto è rumore.
    """
    rng = np.random.default_rng(seed)
    dim = len(leaves[0].bounds)
    optimum = np.zeros(dim)
    
    n_consistent = int(len(leaves) * consistency_ratio)
    
    for i, leaf in enumerate(leaves):
        center = leaf.center()
        if i < n_consistent:
            # Gradiente consistente
            grad = center - optimum
        else:
            # Gradiente casuale
            grad = rng.standard_normal(dim)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad = grad / grad_norm
        else:
            grad = np.zeros(dim)
        
        leaf.lgs_model = {
            "grad": grad,
            "intercept": 0.0,
            "all_pts": [center],
        }


def test_path_independence():
    """
    TEST 1: Path Independence
    
    Per un campo conservativo, il potenziale calcolato NON dovrebbe dipendere
    dal percorso. Testiamo questo creando un grafo con cicli e verificando
    che il potenziale sia consistente.
    """
    print("\n" + "="*70)
    print("TEST 1: Path Independence")
    print("="*70)
    
    from alba_framework_potential.coherence import (
        _build_knn_graph,
        _compute_predicted_drops,
        _solve_potential_least_squares,
    )
    
    # Crea foglie con gradienti consistenti
    leaves = create_mock_leaves(n_leaves=20, dim=5, seed=42)
    assign_consistent_gradients(leaves, optimum=np.zeros(5))
    
    # Costruisci grafo kNN
    edges = _build_knn_graph(leaves, k=6)
    d_lm, alignment, valid_edges = _compute_predicted_drops(leaves, edges)
    
    if len(valid_edges) == 0:
        print("  ⚠️ No valid edges - cannot test")
        return
    
    # Risolvi per il potenziale
    u = _solve_potential_least_squares(len(leaves), valid_edges, d_lm)
    
    # Calcola i residui: quanto le previsioni differiscono dalla realtà?
    residuals = []
    for e, (i, j) in enumerate(valid_edges):
        predicted_drop = d_lm[e]
        actual_drop = u[j] - u[i]
        residuals.append(abs(predicted_drop - actual_drop))
    
    mean_residual = np.mean(residuals)
    max_residual = np.max(residuals)
    
    print(f"  Consistent gradients (should be low residuals):")
    print(f"    Mean residual: {mean_residual:.4f}")
    print(f"    Max residual:  {max_residual:.4f}")
    
    # Ora con gradienti inconsistenti
    leaves2 = create_mock_leaves(n_leaves=20, dim=5, seed=42)
    assign_inconsistent_gradients(leaves2)
    
    edges2 = _build_knn_graph(leaves2, k=6)
    d_lm2, _, valid_edges2 = _compute_predicted_drops(leaves2, edges2)
    u2 = _solve_potential_least_squares(len(leaves2), valid_edges2, d_lm2)
    
    residuals2 = []
    for e, (i, j) in enumerate(valid_edges2):
        predicted_drop = d_lm2[e]
        actual_drop = u2[j] - u2[i]
        residuals2.append(abs(predicted_drop - actual_drop))
    
    mean_residual2 = np.mean(residuals2)
    
    print(f"\n  Inconsistent gradients (should be high residuals):")
    print(f"    Mean residual: {mean_residual2:.4f}")
    
    if mean_residual < mean_residual2 * 0.5:
        print("\n  ✅ PASS: Consistent gradients have lower residuals")
    else:
        print("\n  ⚠️ WARN: Residual difference is not significant")


def test_potential_vs_distance():
    """
    TEST 2: Potential vs Distance from Optimum
    
    Con gradienti consistenti, il potenziale dovrebbe correlare con la 
    distanza dall'ottimo (foglie più lontane = potenziale più alto).
    """
    print("\n" + "="*70)
    print("TEST 2: Potential vs Distance from Optimum")
    print("="*70)
    
    from alba_framework_potential.coherence import compute_coherence_scores
    
    # Crea foglie con gradienti consistenti
    leaves = create_mock_leaves(n_leaves=30, dim=5, seed=42)
    assign_consistent_gradients(leaves, optimum=np.zeros(5))
    
    scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves)
    
    # Calcola distanze dall'ottimo
    distances = [np.linalg.norm(leaf.center()) for leaf in leaves]
    potential_values = [potentials.get(i, 0.5) for i in range(len(leaves))]
    
    corr, pval = spearmanr(distances, potential_values)
    
    print(f"  Correlation (potential vs distance): {corr:.3f} (p={pval:.4f})")
    print(f"  Global coherence: {global_coh:.3f}")
    
    if corr > 0.3:
        print("  ✅ PASS: Potential correlates positively with distance")
    elif corr < -0.3:
        print("  ⚠️ WARN: Potential is INVERTED (negative correlation)")
    else:
        print("  ⚠️ WARN: Weak or no correlation")


def test_coherence_discriminates():
    """
    TEST 3: Coherence Score Discrimination
    
    I punteggi di coerenza dovrebbero essere ALTI quando i gradienti sono
    consistenti e BASSI quando sono inconsistenti.
    """
    print("\n" + "="*70)
    print("TEST 3: Coherence Score Discrimination")
    print("="*70)
    
    from alba_framework_potential.coherence import compute_coherence_scores
    
    results = []
    
    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        leaves = create_mock_leaves(n_leaves=25, dim=5, seed=42)
        assign_partially_consistent_gradients(leaves, consistency_ratio=ratio)
        
        scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves)
        mean_score = np.mean(list(scores.values()))
        
        results.append((ratio, global_coh, mean_score))
        print(f"  Consistency {ratio*100:5.0f}%: global_coh={global_coh:.3f}, mean_score={mean_score:.3f}")
    
    # Verifica monotonia
    global_cohs = [r[1] for r in results]
    is_monotonic = all(global_cohs[i] <= global_cohs[i+1] for i in range(len(global_cohs)-1))
    
    if is_monotonic:
        print("\n  ✅ PASS: Coherence increases monotonically with consistency")
    else:
        print("\n  ⚠️ WARN: Coherence is NOT monotonic with consistency")


def test_high_dimensional_stability():
    """
    TEST 4: High-Dimensional Stability
    
    In alta dimensione, il grafo kNN diventa più sparse.
    Testiamo che il modulo non collassi.
    """
    print("\n" + "="*70)
    print("TEST 4: High-Dimensional Stability")
    print("="*70)
    
    from alba_framework_potential.coherence import compute_coherence_scores
    
    for dim in [5, 10, 20, 50]:
        leaves = create_mock_leaves(n_leaves=30, dim=dim, seed=42)
        assign_consistent_gradients(leaves, optimum=np.zeros(dim))
        
        try:
            scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves)
            
            # Verifica che i valori siano sensati
            score_values = list(scores.values())
            potential_values = list(potentials.values())
            
            has_nan = np.any(np.isnan(score_values)) or np.any(np.isnan(potential_values))
            in_range = all(0 <= s <= 1 for s in score_values) and all(0 <= p <= 1 for p in potential_values)
            
            status = "✅" if (not has_nan and in_range) else "⚠️"
            print(f"  {dim:2d}D: global_coh={global_coh:.3f}, scores in [0,1]: {in_range}, NaN: {has_nan} {status}")
        except Exception as e:
            print(f"  {dim:2d}D: ❌ ERROR: {e}")


def test_gauge_invariance():
    """
    TEST 5: Gauge Invariance
    
    Il potenziale è definito a meno di una costante. Dopo l'anchoring,
    il potenziale minimo dovrebbe essere 0 e il massimo 1.
    """
    print("\n" + "="*70)
    print("TEST 5: Gauge Invariance (Anchoring)")
    print("="*70)
    
    from alba_framework_potential.coherence import compute_coherence_scores
    
    for seed in [42, 123, 456]:
        leaves = create_mock_leaves(n_leaves=25, dim=5, seed=seed)
        assign_consistent_gradients(leaves, optimum=np.zeros(5))
        
        scores, potentials, global_coh, q60, q80 = compute_coherence_scores(leaves)
        
        potential_values = list(potentials.values())
        p_min = min(potential_values)
        p_max = max(potential_values)
        
        is_normalized = (abs(p_min) < 0.01 or abs(p_min - 0.5) < 0.01) and (abs(p_max - 1.0) < 0.01 or abs(p_max - 0.5) < 0.01)
        
        print(f"  Seed {seed}: potential range [{p_min:.3f}, {p_max:.3f}]")
    
    print("  Note: After normalization, potentials should be in [0, 1]")


def test_curl_free_violation():
    """
    TEST 6: Curl-Free Violation
    
    Il teorema di Stokes dice che per un campo conservativo, l'integrale
    lungo un ciclo chiuso è zero. Se i gradienti NON sono derivati da
    un potenziale scalare, questo può fallire.
    
    Creiamo un campo con "rotazione" per vedere come si comporta.
    """
    print("\n" + "="*70)
    print("TEST 6: Non-Conservative Field (Curl ≠ 0)")
    print("="*70)
    
    from alba_framework_potential.coherence import (
        _build_knn_graph,
        _compute_predicted_drops,
        _solve_potential_least_squares,
    )
    
    # Crea un campo con rotazione (non conservativo)
    leaves = create_mock_leaves(n_leaves=20, dim=2, seed=42)
    
    for leaf in leaves:
        center = leaf.center()
        x, y = center[0], center[1]
        # Campo con rotazione: g = (-y, x) / r
        r = np.sqrt(x*x + y*y) + 0.1
        grad = np.array([-y/r, x/r])
        grad = grad / (np.linalg.norm(grad) + 1e-9)
        
        leaf.lgs_model = {
            "grad": grad,
            "intercept": 0.0,
            "all_pts": [center],
        }
    
    edges = _build_knn_graph(leaves, k=6)
    d_lm, alignment, valid_edges = _compute_predicted_drops(leaves, edges)
    
    if len(valid_edges) == 0:
        print("  ⚠️ No valid edges")
        return
    
    u = _solve_potential_least_squares(len(leaves), valid_edges, d_lm)
    
    # Calcola residui
    residuals = []
    for e, (i, j) in enumerate(valid_edges):
        predicted_drop = d_lm[e]
        actual_drop = u[j] - u[i]
        residuals.append(abs(predicted_drop - actual_drop))
    
    mean_residual = np.mean(residuals)
    
    print(f"  Mean residual for rotational field: {mean_residual:.4f}")
    print(f"  (High residual expected since field is non-conservative)")
    
    if mean_residual > 0.3:
        print("  ✅ PASS: High residual correctly detects non-conservative field")
    else:
        print("  ⚠️ WARN: Residual is surprisingly low")


def test_real_optimization_scenario():
    """
    TEST 7: Real Optimization Scenario
    
    Esegui una vera ottimizzazione e verifica che le coerenze 
    siano sensate.
    """
    print("\n" + "="*70)
    print("TEST 7: Real Optimization Scenario")
    print("="*70)
    
    from alba_framework_potential.optimizer import ALBA
    
    dim = 5
    bounds = [(-5.0, 5.0)] * dim
    
    def sphere(x):
        return float(np.sum(np.array(x)**2))
    
    # Esegui ottimizzazione
    opt = ALBA(bounds=bounds, total_budget=100, use_potential_field=True, seed=42)
    
    # Hook per catturare le coerenze durante l'ottimizzazione
    coherences_over_time = []
    
    original_tell = opt.tell
    def tell_wrapper(x, y):
        result = original_tell(x, y)
        if opt._coherence_tracker and opt._coherence_tracker._cache:
            coherences_over_time.append(opt._coherence_tracker._cache.global_coherence)
        return result
    
    opt.tell = tell_wrapper
    opt.optimize(sphere, 100)
    
    if coherences_over_time:
        print(f"  Coherence over time:")
        print(f"    Start: {coherences_over_time[0]:.3f}")
        print(f"    End:   {coherences_over_time[-1]:.3f}")
        print(f"    Mean:  {np.mean(coherences_over_time):.3f}")
        print(f"    Std:   {np.std(coherences_over_time):.3f}")
        
        # La coerenza dovrebbe stabilizzarsi
        early = np.mean(coherences_over_time[:10]) if len(coherences_over_time) >= 10 else coherences_over_time[0]
        late = np.mean(coherences_over_time[-10:]) if len(coherences_over_time) >= 10 else coherences_over_time[-1]
        
        if late > early * 0.9:
            print("  ✅ PASS: Coherence remains stable during optimization")
        else:
            print("  ⚠️ WARN: Coherence dropped significantly during optimization")
    else:
        print("  ⚠️ Could not capture coherences")


def main():
    print("\n" + "="*70)
    print("COHERENCE MODULE - DEEP STABILITY ANALYSIS")
    print("="*70)
    print("Testing mathematical foundations and edge cases")
    
    test_path_independence()
    test_potential_vs_distance()
    test_coherence_discriminates()
    test_high_dimensional_stability()
    test_gauge_invariance()
    test_curl_free_violation()
    test_real_optimization_scenario()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
