#!/usr/bin/env python3
"""
ANALISI COMPLETA DEL FLUSSO COHERENCE + POTENTIAL FIELD

Questo script traccia numericamente tutto ciò che succede nel modulo Coherence:
1. Costruzione grafo kNN
2. Calcolo gradienti LGS e predicted drops
3. Risoluzione del campo potenziale (least squares)
4. Inversione e normalizzazione
5. Come influenza la selezione delle foglie

Confronto su funzione SMOOTH vs DISCRETIZED per capire perché PF aiuta su quest'ultima.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import ALBA components
from alba_framework_potential.cube import Cube
from alba_framework_potential.coherence import (
    _build_knn_graph,
    _compute_predicted_drops,
    _solve_potential_least_squares,
    compute_coherence_scores,
    CoherenceTracker,
)
from alba_framework_potential.optimizer import ALBA


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def sphere_discretized(x: np.ndarray, n_bins: int = 15) -> float:
    """Sphere discretizzata come RF surrogate."""
    y = sphere(x)
    y_range = (0, 200)
    bin_size = (y_range[1] - y_range[0]) / n_bins
    bin_idx = int(np.clip(y, y_range[0], y_range[1] - 0.01) / bin_size)
    return y_range[0] + (bin_idx + 0.5) * bin_size


# ============================================================================
# DIAGNOSTIC OPTIMIZER
# ============================================================================

class DiagnosticALBA(ALBA):
    """ALBA con diagnostica estesa per tracciare tutto il flusso Coherence."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = []
        self._diag_iteration = 0
    
    def ask(self):
        """Ask con logging diagnostico."""
        result = super().ask()
        
        # Raccogli diagnostica ogni 10 iterazioni
        if self._diag_iteration % 10 == 0:
            self._collect_diagnostics()
        
        self._diag_iteration += 1
        return result
    
    def _collect_diagnostics(self):
        """Raccoglie informazioni diagnostiche dettagliate."""
        if not hasattr(self, '_root') or self._root is None:
            return
        
        leaves = self._root.leaves()
        n_leaves = len(leaves)
        
        diag = {
            'iteration': self._diag_iteration,
            'n_leaves': n_leaves,
            'total_trials': sum(c.n_trials for c in leaves),
        }
        
        if self._coherence_tracker is not None and n_leaves >= 5:
            # Forza update coherence
            self._coherence_tracker.update(leaves, self._diag_iteration, force=True)
            
            # Raccogli statistiche
            stats = self._coherence_tracker.get_statistics()
            diag.update(stats)
            
            # Analisi dettagliata per foglie top e bottom
            leaf_data = []
            for i, leaf in enumerate(leaves):
                coh = self._coherence_tracker.get_coherence(leaf, leaves)
                pot = self._coherence_tracker.get_potential(leaf, leaves)
                leaf_data.append({
                    'idx': i,
                    'n_trials': leaf.n_trials,
                    'n_good': leaf.n_good,
                    'good_ratio': leaf.good_ratio(),
                    'coherence': coh,
                    'potential': pot,
                    'best_y': leaf.best_y if hasattr(leaf, 'best_y') else None,
                    'center': leaf.center().tolist(),
                    'has_gradient': leaf.lgs_model is not None and leaf.lgs_model.get('gradient') is not None,
                })
            
            diag['leaf_data'] = leaf_data
            
            # Ordina per potential (lower = better)
            sorted_by_potential = sorted(leaf_data, key=lambda x: x['potential'])
            diag['best_by_potential'] = sorted_by_potential[:3]
            diag['worst_by_potential'] = sorted_by_potential[-3:]
            
            # Ordina per coherence
            sorted_by_coherence = sorted(leaf_data, key=lambda x: x['coherence'], reverse=True)
            diag['best_by_coherence'] = sorted_by_coherence[:3]
            
        self.diagnostics.append(diag)


def run_diagnostic_comparison(func, func_name: str, dim: int, budget: int, seed: int):
    """Esegue ALBA con diagnostica e stampa report dettagliato."""
    
    print(f"\n{'='*80}")
    print(f"  DIAGNOSTICA: {func_name} (dim={dim}, budget={budget}, seed={seed})")
    print(f"{'='*80}")
    
    bounds = [(-5.0, 5.0)] * dim
    
    # Esegui COV-only (PF=False)
    print("\n--- Esecuzione COV-only (PF=False) ---")
    opt_cov = DiagnosticALBA(
        bounds=bounds,
        seed=seed,
        maximize=False,
        total_budget=budget,
        use_potential_field=False,
        use_coherence_gating=True,
    )
    
    best_y_cov = np.inf
    for i in range(budget):
        x = opt_cov.ask()
        if isinstance(x, dict):
            x_arr = np.array(list(x.values()))
        else:
            x_arr = np.array(x)
        y = func(x_arr)
        opt_cov.tell(x, y)
        if y < best_y_cov:
            best_y_cov = y
    
    # Esegui COV+PF (PF=True)
    print("--- Esecuzione COV+PF (PF=True) ---")
    opt_pf = DiagnosticALBA(
        bounds=bounds,
        seed=seed,
        maximize=False,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
    )
    
    best_y_pf = np.inf
    for i in range(budget):
        x = opt_pf.ask()
        if isinstance(x, dict):
            x_arr = np.array(list(x.values()))
        else:
            x_arr = np.array(x)
        y = func(x_arr)
        opt_pf.tell(x, y)
        if y < best_y_pf:
            best_y_pf = y
    
    print(f"\n{'='*80}")
    print(f"  RISULTATI FINALI: {func_name}")
    print(f"{'='*80}")
    print(f"  COV-only best: {best_y_cov:.6f}")
    print(f"  COV+PF best:   {best_y_pf:.6f}")
    print(f"  Winner:        {'PF' if best_y_pf < best_y_cov else 'COV'}")
    
    return {
        'func_name': func_name,
        'best_cov': best_y_cov,
        'best_pf': best_y_pf,
        'diagnostics_cov': opt_cov.diagnostics,
        'diagnostics_pf': opt_pf.diagnostics,
    }


def analyze_coherence_flow(diagnostics: List[Dict], label: str):
    """Analizza il flusso di coherence nel tempo."""
    
    print(f"\n{'='*80}")
    print(f"  ANALISI FLUSSO COHERENCE: {label}")
    print(f"{'='*80}")
    
    print(f"\n{'Iter':<6} {'Leaves':<8} {'GlobCoh':<10} {'Q60':<8} {'Q80':<8} {'PotRange':<12}")
    print("-" * 60)
    
    for d in diagnostics:
        if 'global_coherence' in d:
            # Calcola range potenziali
            pot_range = "N/A"
            if 'leaf_data' in d:
                pots = [ld['potential'] for ld in d['leaf_data']]
                pot_range = f"{min(pots):.3f}-{max(pots):.3f}"
            
            print(f"{d['iteration']:<6} {d['n_leaves']:<8} "
                  f"{d['global_coherence']:<10.4f} {d['q60_threshold']:<8.4f} "
                  f"{d.get('q80_threshold', 0):<8.4f} {pot_range:<12}")
    
    # Analisi finale
    if diagnostics and 'leaf_data' in diagnostics[-1]:
        final = diagnostics[-1]
        print(f"\n  STATO FINALE (iter {final['iteration']}):")
        print(f"  - Foglie totali: {final['n_leaves']}")
        print(f"  - Global coherence: {final['global_coherence']:.4f}")
        
        print(f"\n  TOP 3 foglie per POTENTIAL (lower = better):")
        for ld in final.get('best_by_potential', []):
            print(f"    Leaf {ld['idx']}: pot={ld['potential']:.4f}, coh={ld['coherence']:.4f}, "
                  f"trials={ld['n_trials']}, good_ratio={ld['good_ratio']:.3f}")
        
        print(f"\n  WORST 3 foglie per POTENTIAL:")
        for ld in final.get('worst_by_potential', []):
            print(f"    Leaf {ld['idx']}: pot={ld['potential']:.4f}, coh={ld['coherence']:.4f}, "
                  f"trials={ld['n_trials']}, good_ratio={ld['good_ratio']:.3f}")
        
        print(f"\n  TOP 3 foglie per COHERENCE:")
        for ld in final.get('best_by_coherence', []):
            print(f"    Leaf {ld['idx']}: coh={ld['coherence']:.4f}, pot={ld['potential']:.4f}, "
                  f"trials={ld['n_trials']}, has_grad={ld['has_gradient']}")


def deep_coherence_analysis():
    """
    Analisi numerica profonda del modulo Coherence.
    
    Simula manualmente i passaggi per capire i valori numerici.
    """
    
    print("\n" + "=" * 80)
    print("  ANALISI NUMERICA PROFONDA DEL MODULO COHERENCE")
    print("=" * 80)
    
    # Creiamo uno scenario semplificato con 6 foglie
    # Simuliamo una funzione 2D: f(x,y) = x^2 + y^2
    # Il minimo è in (0, 0)
    
    print("\n  SCENARIO: 6 foglie in spazio 2D, funzione f(x,y) = x² + y²")
    print("  Minimo globale in (0, 0)")
    
    # Centri delle foglie
    centers = np.array([
        [0.1, 0.1],    # Leaf 0: vicino al minimo
        [1.0, 0.5],    # Leaf 1: a destra
        [-0.5, 1.0],   # Leaf 2: in alto a sinistra
        [0.3, -0.8],   # Leaf 3: in basso
        [2.0, 2.0],    # Leaf 4: lontano (angolo)
        [-1.5, -1.0],  # Leaf 5: a sinistra
    ])
    
    # Valori della funzione e gradienti analitici
    # ∇f = (2x, 2y)
    values = np.sum(centers**2, axis=1)
    gradients = 2 * centers  # Gradiente analitico
    
    # NOTA: ALBA usa y_internal = -f(x) per minimizzazione
    # Quindi LGS gradient = -∇f (punta VERSO il minimo)
    lgs_gradients = -gradients
    
    print("\n  Dati delle foglie:")
    print(f"  {'Leaf':<6} {'Center':<20} {'f(x)':<10} {'∇f':<25} {'LGS grad':<25}")
    print("-" * 90)
    
    for i in range(6):
        print(f"  {i:<6} ({centers[i,0]:>6.2f}, {centers[i,1]:>6.2f})      "
              f"{values[i]:<10.4f} ({gradients[i,0]:>6.2f}, {gradients[i,1]:>6.2f})        "
              f"({lgs_gradients[i,0]:>6.2f}, {lgs_gradients[i,1]:>6.2f})")
    
    print("\n" + "-" * 80)
    print("  STEP 1: Costruzione grafo kNN (k=3)")
    print("-" * 80)
    
    # Costruisci manualmente il grafo kNN
    n = len(centers)
    k = 3
    edges = []
    
    for i in range(n):
        dists = np.sum((centers - centers[i])**2, axis=1)
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            edges.append((i, j))
    
    print(f"  Edges: {edges[:12]}...")  # Solo primi 12
    
    print("\n" + "-" * 80)
    print("  STEP 2: Calcolo predicted drops d_lm = g_l · (c_m - c_l)")
    print("-" * 80)
    
    print(f"\n  {'Edge':<10} {'Δc = c_m - c_l':<25} {'g_l (LGS)':<25} {'d_lm':<12} {'Interpretazione'}")
    print("-" * 90)
    
    predicted_drops = []
    alignments = []
    
    for i, j in edges[:8]:  # Solo primi 8 per leggibilità
        delta_c = centers[j] - centers[i]
        g_l = lgs_gradients[i]
        
        # Predicted drop
        d_lm = np.dot(g_l, delta_c)
        predicted_drops.append(d_lm)
        
        # Alignment (cosine similarity)
        g_norm = np.linalg.norm(g_l)
        delta_norm = np.linalg.norm(delta_c)
        if g_norm > 1e-9 and delta_norm > 1e-9:
            alignment = np.dot(g_l, delta_c) / (g_norm * delta_norm)
        else:
            alignment = 0.0
        alignments.append(alignment)
        
        # Interpretazione
        if d_lm > 0:
            interp = "u_j > u_i (j peggiore)"
        else:
            interp = "u_j < u_i (j migliore)"
        
        print(f"  ({i},{j})      ({delta_c[0]:>6.2f}, {delta_c[1]:>6.2f})       "
              f"({g_l[0]:>6.2f}, {g_l[1]:>6.2f})        {d_lm:>+8.4f}    {interp}")
    
    print("\n" + "-" * 80)
    print("  STEP 3: Soluzione Least Squares per potenziale u")
    print("-" * 80)
    
    print("\n  Minimizziamo: Σ (u_j - u_i - d_ij)²")
    print("  Vincolo: u_0 = 0 (gauge fixing)")
    
    # Calcola tutti i predicted drops per tutti gli edge
    all_d_lm = []
    for i, j in edges:
        delta_c = centers[j] - centers[i]
        g_l = lgs_gradients[i]
        d_lm = np.dot(g_l, delta_c)
        all_d_lm.append(d_lm)
    
    # Risolvi sistema lineare
    # u_j - u_i = d_ij  =>  A @ u = d
    # Con u_0 = 0: riduciamo a n-1 variabili
    
    n_edges = len(edges)
    A = np.zeros((n_edges, n - 1))
    b = np.array(all_d_lm)
    
    for e, (i, j) in enumerate(edges):
        if i > 0:
            A[e, i - 1] = -1
        if j > 0:
            A[e, j - 1] = +1
    
    # Risolvi via pseudoinversa
    u_reduced, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    u = np.zeros(n)
    u[1:] = u_reduced
    
    print("\n  Potenziale raw (prima inversione):")
    for i in range(n):
        print(f"    u[{i}] = {u[i]:>+8.4f}  (f={values[i]:.4f})")
    
    print("\n" + "-" * 80)
    print("  STEP 4: Inversione del potenziale (u_inverted = -u)")
    print("-" * 80)
    
    print("\n  PROBLEMA: LGS gradient = -∇f (punta verso minimo)")
    print("  Quando integriamo: camminando VERSO minimo, g·Δ > 0")
    print("  Quindi u CRESCE verso il minimo → SBAGLIATO!")
    print("  Dobbiamo invertire: u_inverted = -u")
    
    u_inverted = -u
    
    print("\n  Potenziale invertito:")
    for i in range(n):
        print(f"    u_inv[{i}] = {u_inverted[i]:>+8.4f}  (f={values[i]:.4f}) "
              f"{'← MINIMO' if values[i] < 0.5 else ''}")
    
    print("\n" + "-" * 80)
    print("  STEP 5: Normalizzazione a [0, 1]")
    print("-" * 80)
    
    # Ancora al minimo
    best_idx = np.argmin(u_inverted)
    u_anchored = u_inverted - u_inverted[best_idx]
    
    # Normalizza
    u_max = np.max(u_anchored)
    u_norm = u_anchored / u_max if u_max > 1e-9 else np.full(n, 0.5)
    u_norm = np.clip(u_norm, 0, 1)
    
    print("\n  Potenziale normalizzato [0, 1]:")
    print(f"  {'Leaf':<6} {'u_norm':<10} {'f(x)':<10} {'Distanza da origine':<20}")
    print("-" * 50)
    
    for i in range(n):
        dist = np.linalg.norm(centers[i])
        print(f"  {i:<6} {u_norm[i]:<10.4f} {values[i]:<10.4f} {dist:<20.4f}")
    
    # Verifica correlazione
    corr = np.corrcoef(u_norm, values)[0, 1]
    print(f"\n  Correlazione (u_norm, f): {corr:.4f}")
    print(f"  {'✅ BUONA correlazione!' if corr > 0.8 else '⚠️ Correlazione debole'}")
    
    print("\n" + "-" * 80)
    print("  STEP 6: Coherence-scaling del peso PF")
    print("-" * 80)
    
    # Simula coherence scores (mean alignment per nodo)
    node_alignments = {i: [] for i in range(n)}
    for e, (i, j) in enumerate(edges):
        delta_c = centers[j] - centers[i]
        g_l = lgs_gradients[i]
        g_norm = np.linalg.norm(g_l)
        delta_norm = np.linalg.norm(delta_c)
        if g_norm > 1e-9 and delta_norm > 1e-9:
            align = np.dot(g_l, delta_c) / (g_norm * delta_norm)
            node_alignments[i].append(align)
            node_alignments[j].append(align)
    
    coherence_scores = []
    for i in range(n):
        if node_alignments[i]:
            mean_align = np.mean(node_alignments[i])
            coh = (mean_align + 1) / 2  # Map [-1,1] to [0,1]
        else:
            coh = 0.5
        coherence_scores.append(coh)
    
    global_coherence = np.median(coherence_scores)
    
    print(f"\n  Coherence scores per foglia:")
    for i in range(n):
        print(f"    Leaf {i}: coherence = {coherence_scores[i]:.4f}")
    
    print(f"\n  Global coherence (median): {global_coherence:.4f}")
    
    # Calcola scaling factor
    coherence_scale = max(0.0, min(1.0, (global_coherence - 0.5) * 3.33))
    
    print(f"\n  Formula scaling: coherence_scale = max(0, min(1, (global_coh - 0.5) * 3.33))")
    print(f"  → coherence_scale = {coherence_scale:.4f}")
    
    if coherence_scale < 0.3:
        print(f"  ⚠️ Coherence bassa: PF peso ridotto del {(1-coherence_scale)*100:.0f}%")
    else:
        print(f"  ✅ Coherence buona: PF usa {coherence_scale*100:.0f}% del peso nominale")
    
    print("\n" + "-" * 80)
    print("  STEP 7: Effetto sulla selezione foglie")
    print("-" * 80)
    
    print("\n  Formula score foglia (PotentialAwareLeafSelector):")
    print("    score = good_ratio + exploration + model_bonus + potential_bonus")
    print("    potential_bonus = weight * coherence_scale * (1 - u_norm)")
    print()
    
    potential_weight = 0.5
    
    print(f"  {'Leaf':<6} {'good_ratio':<12} {'pot_bonus':<12} {'Effetto'}")
    print("-" * 50)
    
    for i in range(n):
        pot_bonus = potential_weight * coherence_scale * (1 - u_norm[i])
        effect = "↑ favorita" if pot_bonus > 0.1 else "→ neutro" if pot_bonus > 0.05 else "↓ penalizzata"
        print(f"  {i:<6} {0.5:<12.4f} {pot_bonus:<12.4f} {effect}")


def compare_smooth_vs_discretized_coherence():
    """
    Confronta il comportamento di Coherence su funzione smooth vs discretized.
    """
    
    print("\n" + "=" * 80)
    print("  CONFRONTO COHERENCE: SMOOTH vs DISCRETIZED")
    print("=" * 80)
    
    DIM = 4
    BUDGET = 100
    SEED = 42
    
    # Test su SMOOTH
    result_smooth = run_diagnostic_comparison(
        sphere, "SPHERE_SMOOTH", DIM, BUDGET, SEED
    )
    
    # Test su DISCRETIZED
    result_disc = run_diagnostic_comparison(
        sphere_discretized, "SPHERE_DISCRETIZED", DIM, BUDGET, SEED
    )
    
    # Analisi dettagliata
    analyze_coherence_flow(result_smooth['diagnostics_pf'], "SMOOTH + PF")
    analyze_coherence_flow(result_disc['diagnostics_pf'], "DISCRETIZED + PF")
    
    # Confronto finale
    print("\n" + "=" * 80)
    print("  CONFRONTO FINALE")
    print("=" * 80)
    
    print(f"\n  {'Metrica':<30} {'SMOOTH':<15} {'DISCRETIZED':<15}")
    print("-" * 60)
    
    # Global coherence finale
    smooth_coh = result_smooth['diagnostics_pf'][-1].get('global_coherence', 0.5) if result_smooth['diagnostics_pf'] else 0.5
    disc_coh = result_disc['diagnostics_pf'][-1].get('global_coherence', 0.5) if result_disc['diagnostics_pf'] else 0.5
    
    print(f"  {'Final global coherence':<30} {smooth_coh:<15.4f} {disc_coh:<15.4f}")
    print(f"  {'Best y (COV-only)':<30} {result_smooth['best_cov']:<15.6f} {result_disc['best_cov']:<15.6f}")
    print(f"  {'Best y (COV+PF)':<30} {result_smooth['best_pf']:<15.6f} {result_disc['best_pf']:<15.6f}")
    
    # Delta PF
    delta_smooth = (result_smooth['best_cov'] - result_smooth['best_pf']) / result_smooth['best_cov'] * 100
    delta_disc = (result_disc['best_cov'] - result_disc['best_pf']) / result_disc['best_cov'] * 100
    
    print(f"  {'PF improvement %':<30} {delta_smooth:<+15.1f} {delta_disc:<+15.1f}")
    
    print("\n" + "=" * 80)
    if delta_disc > delta_smooth:
        print("  🔥 PF aiuta DI PIÙ su DISCRETIZED!")
        print("  Possibili motivi:")
        print("    1. Su discretized, i gradienti locali sono 'spezzati'")
        print("    2. Il campo potenziale globale fornisce direzione quando gradiente locale è nullo")
        print("    3. PF stabilizza evitando oscillazioni tra plateau vicini")
    else:
        print("  ➖ PF aiuta ugualmente (o meno) su DISCRETIZED")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("  ANALISI COMPLETA MODULO COHERENCE")
    print("=" * 80)
    
    # 1. Analisi numerica profonda su scenario semplificato
    deep_coherence_analysis()
    
    # 2. Confronto smooth vs discretized
    compare_smooth_vs_discretized_coherence()


if __name__ == "__main__":
    main()
