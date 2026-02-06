"""
ALBA Coherence Analysis on Rastrigin

Analisi approfondita di come la coherence tracker rileva e gestisce
i gradienti conflittuali su funzioni multimodali come Rastrigin.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/workspace/thesis')
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.coherence import (
    CoherenceTracker,
    compute_coherence_scores,
    _build_knn_graph,
    _compute_predicted_drops,
)


def trace_separator(title: str):
    print("\n" + "="*80)
    print(f"ANALYSIS: {title}")
    print("="*80)


def analyze_coherence_rastrigin():
    """
    Analisi dettagliata della coherence su Rastrigin.
    """
    trace_separator("COHERENCE ON RASTRIGIN")
    
    dim = 5
    bounds = [(0, 1)] * dim
    
    def rastrigin(x):
        x_shifted = (x - 0.5) * 5.12 * 2
        A = 10
        return A * dim + np.sum(x_shifted**2 - A * np.cos(2 * np.pi * x_shifted))
    
    # ALBA con coherence tracking attivo
    opt = ALBA(
        bounds=bounds, 
        seed=42, 
        total_budget=150,
        maximize=False,
        use_coherence_gating=True,
        use_potential_field=True,
        coherence_update_interval=5,
    )
    
    print("\n[SETUP]")
    print(f"  dim = {dim}")
    print(f"  use_coherence_gating = True")
    print(f"  use_potential_field = True")
    print(f"  coherence_update_interval = 5")
    
    # Run e traccia coherence
    coherence_history = []
    
    print("\n[EXECUTION]")
    for i in range(100):
        x = opt.ask()
        y = rastrigin(x)
        opt.tell(x, y)
        
        # Ogni 10 iterazioni, dump coherence state
        if i % 10 == 9:
            tracker = opt._coherence_tracker
            if tracker is not None and tracker._cache is not None:
                cache = tracker._cache
                
                entry = {
                    'iter': i + 1,
                    'n_leaves': len(opt.leaves),
                    'global_coherence': cache.global_coherence,
                    'q60': cache.q60_threshold,
                    'q80': cache.q80_threshold,
                    'scores': dict(cache.scores),
                    'potentials': dict(cache.potentials),
                    'best_y': opt.best_y,
                }
                coherence_history.append(entry)
                
                print(f"\n  [Iter {i+1}]")
                print(f"    n_leaves = {len(opt.leaves)}")
                print(f"    global_coherence = {cache.global_coherence:.4f}")
                print(f"    best_y = {opt.best_y:.4f}")
    
    print(f"\n[COHERENCE EVOLUTION]")
    for entry in coherence_history:
        print(f"  Iter {entry['iter']:3d}: global_coh = {entry['global_coherence']:.4f}, "
              f"leaves = {entry['n_leaves']}, best_y = {entry['best_y']:.4f}")
    
    # Analisi dettagliata dello stato finale
    print(f"\n[FINAL STATE ANALYSIS]")
    
    if not coherence_history:
        print("  No coherence data collected")
        return
    
    final_state = coherence_history[-1]
    scores = final_state['scores']
    potentials = final_state['potentials']
    
    if scores:
        score_values = list(scores.values())
        print(f"\n  Coherence scores per leaf:")
        print(f"    min = {min(score_values):.4f}")
        print(f"    max = {max(score_values):.4f}")
        print(f"    mean = {np.mean(score_values):.4f}")
        print(f"    std = {np.std(score_values):.4f}")
        
        # Distribuzione
        n_high = sum(1 for s in score_values if s > 0.7)
        n_medium = sum(1 for s in score_values if 0.3 <= s <= 0.7)
        n_low = sum(1 for s in score_values if s < 0.3)
        
        print(f"\n    High coherence (>0.7): {n_high} leaves")
        print(f"    Medium (0.3-0.7): {n_medium} leaves")
        print(f"    Low (<0.3): {n_low} leaves")
    
    if potentials:
        pot_values = list(potentials.values())
        print(f"\n  Potential field per leaf:")
        print(f"    min = {min(pot_values):.4f}")
        print(f"    max = {max(pot_values):.4f}")
        print(f"    spread = {max(pot_values) - min(pot_values):.4f}")
    
    # Analisi gradienti vs coherence
    print(f"\n[GRADIENT ALIGNMENT VS COHERENCE]")
    
    leaves_with_model = [l for l in opt.leaves if l.lgs_model is not None]
    print(f"  Leaves with LGS model: {len(leaves_with_model)}/{len(opt.leaves)}")
    
    if len(leaves_with_model) >= 2:
        # Costruisci grafo KNN
        edges = _build_knn_graph(leaves_with_model, k=4)
        
        # Calcola alignment tra vicini
        alignments = []
        for i, j in edges:
            leaf_i = leaves_with_model[i]
            leaf_j = leaves_with_model[j]
            
            grad_i = leaf_i.lgs_model.get('gradient_dir')
            grad_j = leaf_j.lgs_model.get('gradient_dir')
            
            if grad_i is not None and grad_j is not None:
                align = np.dot(grad_i, grad_j)
                alignments.append({
                    'i': i,
                    'j': j,
                    'alignment': align,
                    'center_i': leaf_i.center(),
                    'center_j': leaf_j.center(),
                })
        
        if alignments:
            align_values = [a['alignment'] for a in alignments]
            
            print(f"\n  Gradient alignment tra leaves vicine:")
            print(f"    Edges analyzed: {len(alignments)}")
            print(f"    min alignment = {min(align_values):.4f}")
            print(f"    max alignment = {max(align_values):.4f}")
            print(f"    mean alignment = {np.mean(align_values):.4f}")
            
            # Conta allineamenti positivi vs negativi
            n_aligned = sum(1 for a in align_values if a > 0.5)
            n_orthogonal = sum(1 for a in align_values if -0.5 <= a <= 0.5)
            n_opposite = sum(1 for a in align_values if a < -0.5)
            
            print(f"\n    Well aligned (>0.5): {n_aligned} edges ({100*n_aligned/len(alignments):.0f}%)")
            print(f"    Orthogonal (-0.5 to 0.5): {n_orthogonal} edges ({100*n_orthogonal/len(alignments):.0f}%)")
            print(f"    Opposite (<-0.5): {n_opposite} edges ({100*n_opposite/len(alignments):.0f}%)")
            
            # Mostra esempi di gradienti opposti
            opposite_examples = [a for a in alignments if a['alignment'] < -0.3]
            if opposite_examples:
                print(f"\n  Esempi di gradienti OPPOSTI:")
                for ex in opposite_examples[:3]:
                    print(f"    Leaf {ex['i']} vs Leaf {ex['j']}: alignment = {ex['alignment']:.4f}")
    
    # Come reagisce la coherence ai gradienti conflittuali?
    print(f"\n[COHERENCE REACTION TO CONFLICTS]")
    
    global_coh = final_state['global_coherence']
    print(f"  Global coherence: {global_coh:.4f}")
    
    if global_coh < 0.5:
        print(f"  → LOW COHERENCE DETECTED!")
        print(f"    La coherence ha rilevato che i gradienti sono inconsistenti.")
        print(f"    Questo dovrebbe aumentare l'esplorazione.")
    elif global_coh < 0.7:
        print(f"  → MEDIUM COHERENCE")
        print(f"    Gradienti parzialmente consistenti.")
    else:
        print(f"  → HIGH COHERENCE")
        print(f"    Gradienti consistenti (sorprendente per Rastrigin!)")
    
    # Verifica il gating behavior
    print(f"\n[GATING BEHAVIOR]")
    
    tracker = opt._coherence_tracker
    if tracker is not None:
        # Test il gating su alcune leaves
        for leaf_id, leaf in enumerate(opt.leaves[:3]):
            should_exploit = tracker.should_exploit(leaf, opt.rng)
            print(f"  Leaf {leaf_id}: should_exploit = {should_exploit}")
    
    return coherence_history


def compare_with_sphere():
    """
    Confronta la coherence su Rastrigin vs Sphere (funzione unimodale).
    """
    trace_separator("COMPARISON: RASTRIGIN vs SPHERE")
    
    dim = 5
    bounds = [(0, 1)] * dim
    
    def rastrigin(x):
        x_shifted = (x - 0.5) * 5.12 * 2
        A = 10
        return A * dim + np.sum(x_shifted**2 - A * np.cos(2 * np.pi * x_shifted))
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    results = {}
    
    for name, func in [("Sphere", sphere), ("Rastrigin", rastrigin)]:
        opt = ALBA(
            bounds=bounds, 
            seed=42, 
            total_budget=80,
            maximize=False,
            use_coherence_gating=True,
            use_potential_field=True,
        )
        
        for i in range(60):
            x = opt.ask()
            y = func(x)
            opt.tell(x, y)
        
        tracker = opt._coherence_tracker
        if tracker is not None and tracker._cache is not None:
            cache = tracker._cache
            results[name] = {
                'global_coherence': cache.global_coherence,
                'n_leaves': len(opt.leaves),
                'best_y': opt.best_y,
                'scores': list(cache.scores.values()) if cache.scores else [],
            }
    
    print("\n[COMPARISON]")
    print(f"{'Metric':<25} {'Sphere':>15} {'Rastrigin':>15}")
    print("-" * 55)
    
    for key in ['global_coherence', 'n_leaves', 'best_y']:
        val_sphere = results.get('Sphere', {}).get(key, 'N/A')
        val_rast = results.get('Rastrigin', {}).get(key, 'N/A')
        
        if isinstance(val_sphere, float):
            print(f"{key:<25} {val_sphere:>15.4f} {val_rast:>15.4f}")
        else:
            print(f"{key:<25} {val_sphere:>15} {val_rast:>15}")
    
    # Distribuzione scores
    print("\n[COHERENCE SCORE DISTRIBUTION]")
    for name in ['Sphere', 'Rastrigin']:
        scores = results.get(name, {}).get('scores', [])
        if scores:
            print(f"  {name}:")
            print(f"    mean = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")
            print(f"    min = {min(scores):.4f}, max = {max(scores):.4f}")
    
    # Interpretation
    print("\n[INTERPRETATION]")
    coh_sphere = results.get('Sphere', {}).get('global_coherence', 0)
    coh_rast = results.get('Rastrigin', {}).get('global_coherence', 0)
    
    if coh_sphere > coh_rast:
        diff = coh_sphere - coh_rast
        print(f"  Sphere ha coherence {diff:.4f} più alta di Rastrigin.")
        print(f"  ✅ La coherence RILEVA CORRETTAMENTE che Rastrigin è più caotico!")
    else:
        print(f"  ⚠️ Coherence simile o più alta su Rastrigin?")
        print(f"     Potrebbe essere che le leaves sono molto piccole e localmente coerenti.")


def analyze_exploit_explore_ratio():
    """
    Conta quante volte ALBA decide di exploit vs explore su Rastrigin.
    """
    trace_separator("EXPLOIT/EXPLORE RATIO")
    
    dim = 5
    bounds = [(0, 1)] * dim
    
    def rastrigin(x):
        x_shifted = (x - 0.5) * 5.12 * 2
        A = 10
        return A * dim + np.sum(x_shifted**2 - A * np.cos(2 * np.pi * x_shifted))
    
    # Patch temporaneo per tracciare le decisioni
    exploit_count = 0
    explore_count = 0
    
    opt = ALBA(
        bounds=bounds, 
        seed=42, 
        total_budget=100,
        maximize=False,
        use_coherence_gating=True,
        use_potential_field=True,
    )
    
    # Modifica il tracker per tracciare
    original_should_exploit = None
    if opt._coherence_tracker is not None:
        original_should_exploit = opt._coherence_tracker.should_exploit
        
        def tracked_should_exploit(cube, rng):
            nonlocal exploit_count, explore_count
            result = original_should_exploit(cube, rng)
            if result:
                exploit_count += 1
            else:
                explore_count += 1
            return result
        
        opt._coherence_tracker.should_exploit = tracked_should_exploit
    
    print("\n[EXECUTION]")
    for i in range(80):
        x = opt.ask()
        y = rastrigin(x)
        opt.tell(x, y)
    
    total = exploit_count + explore_count
    if total > 0:
        print(f"\n[EXPLOIT/EXPLORE DECISIONS]")
        print(f"  Exploit: {exploit_count} ({100*exploit_count/total:.1f}%)")
        print(f"  Explore: {explore_count} ({100*explore_count/total:.1f}%)")
        
        if explore_count > exploit_count:
            print(f"\n  ✅ Più esplorazione che sfruttamento - appropriato per Rastrigin!")
        else:
            print(f"\n  ⚠️ Più sfruttamento - potrebbe rimanere bloccato in minimi locali")
    else:
        print("  Nessuna decisione tracciata (potrebbe essere che il tracker non è attivo)")
    
    print(f"\n  Final best_y = {opt.best_y:.4f}")
    print(f"  (Optimum = 0.0, tipico locale ~10-30)")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALBA COHERENCE ANALYSIS ON RASTRIGIN")
    print("="*80)
    
    analyze_coherence_rastrigin()
    compare_with_sphere()
    analyze_exploit_explore_ratio()
