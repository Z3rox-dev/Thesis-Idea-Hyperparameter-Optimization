#!/usr/bin/env python3
"""
Deep dive sul Potential Field che non correla bene con la distanza dall'ottimo.

Ipotesi:
1. Il potential field si basa su gradient alignment, non sulla posizione
2. La density-based re-anchoring sovrascrive il gradient signal
3. Con poche foglie il segnale è rumoroso
4. La normalizzazione [0,1] distorce le differenze
"""

import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

import numpy as np
from scipy.stats import spearmanr
from alba_framework_potential.optimizer import ALBA


def make_sphere(dim):
    return lambda x: float(np.sum(np.array(x)**2))


def analyze_potential_vs_quality():
    """Analizza se il potential correla con la qualità della foglia"""
    print("=" * 70)
    print("ANALYSIS: Potential vs Leaf Quality")
    print("=" * 70)
    
    dim = 10
    budget = 300
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        use_coherence_gating=True,
        seed=42
    )
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
    
    if opt._coherence_tracker is None or len(opt.leaves) == 0:
        print("No coherence tracker or leaves!")
        return
    
    tracker = opt._coherence_tracker
    
    # Raccogli dati per ogni foglia
    data = []
    for i, leaf in enumerate(opt.leaves):
        center = leaf.center()
        dist_from_origin = np.linalg.norm(center)
        potential = tracker._cache.potentials.get(i, 0.5)
        coherence = tracker._cache.scores.get(i, 0.5)
        n_good = leaf.n_good
        n_total = leaf.n_trials
        good_ratio = n_good / max(1, n_total)
        volume = leaf.volume()
        density = n_good / max(1e-12, volume)
        
        # Valore medio della funzione nel centro
        func_at_center = func(center)
        
        data.append({
            "idx": i,
            "dist": dist_from_origin,
            "potential": potential,
            "coherence": coherence,
            "n_good": n_good,
            "n_total": n_total,
            "good_ratio": good_ratio,
            "density": density,
            "func_at_center": func_at_center,
        })
    
    # Calcola correlazioni
    dists = [d["dist"] for d in data]
    pots = [d["potential"] for d in data]
    cohs = [d["coherence"] for d in data]
    funcs = [d["func_at_center"] for d in data]
    densities = [d["density"] for d in data]
    good_ratios = [d["good_ratio"] for d in data]
    
    print(f"\nN leaves: {len(data)}")
    
    # Correlazioni con Potential
    print("\n--- Correlations with POTENTIAL ---")
    corr_dist, p_dist = spearmanr(dists, pots)
    corr_func, p_func = spearmanr(funcs, pots)
    corr_dens, p_dens = spearmanr(densities, pots)
    corr_gr, p_gr = spearmanr(good_ratios, pots)
    
    print(f"  Distance from origin: r={corr_dist:+.3f} (p={p_dist:.3f})")
    print(f"  Function value:       r={corr_func:+.3f} (p={p_func:.3f})")
    print(f"  Density:              r={corr_dens:+.3f} (p={p_dens:.3f})")
    print(f"  Good ratio:           r={corr_gr:+.3f} (p={p_gr:.3f})")
    
    # Correlazioni con Coherence
    print("\n--- Correlations with COHERENCE ---")
    corr_dist_c, _ = spearmanr(dists, cohs)
    corr_func_c, _ = spearmanr(funcs, cohs)
    
    print(f"  Distance from origin: r={corr_dist_c:+.3f}")
    print(f"  Function value:       r={corr_func_c:+.3f}")
    
    # Mostra le foglie ordinate per potential
    print("\n--- Leaves sorted by POTENTIAL (low=good) ---")
    sorted_data = sorted(data, key=lambda x: x["potential"])
    for d in sorted_data[:5]:
        print(f"  Leaf {d['idx']:2d}: pot={d['potential']:.3f}, dist={d['dist']:.2f}, "
              f"f(c)={d['func_at_center']:.1f}, dens={d['density']:.2e}")
    print("  ...")
    for d in sorted_data[-3:]:
        print(f"  Leaf {d['idx']:2d}: pot={d['potential']:.3f}, dist={d['dist']:.2f}, "
              f"f(c)={d['func_at_center']:.1f}, dens={d['density']:.2e}")


def analyze_gradient_vs_function():
    """Verifica che i gradienti puntino verso l'ottimo"""
    print("\n" + "=" * 70)
    print("ANALYSIS: Gradient Direction vs Optimal Direction")
    print("=" * 70)
    
    dim = 5
    budget = 200
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    opt = ALBA(
        bounds=bounds,
        total_budget=budget,
        use_potential_field=True,
        seed=42
    )
    
    for i in range(budget):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
    
    # Per ogni foglia, verifica se il gradiente punta verso l'origine
    print(f"\nN leaves: {len(opt.leaves)}")
    
    alignments = []
    for i, leaf in enumerate(opt.leaves):
        if leaf.lgs_model is None:
            continue
        
        grad = leaf.lgs_model.get("grad")
        if grad is None:
            continue
        
        center = leaf.center()
        
        # Per Sphere, il gradiente dovrebbe puntare LONTANO dall'origine
        # (perché f aumenta allontanandosi dall'origine)
        # Quindi per MINIMIZZARE, dobbiamo andare OPPOSTO al gradiente
        
        # Direzione ottimale: verso l'origine = -center/|center|
        optimal_dir = -center / (np.linalg.norm(center) + 1e-9)
        
        # Direzione anti-gradiente (discesa)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-9:
            continue
        descent_dir = -grad / grad_norm
        
        # Allineamento
        alignment = np.dot(descent_dir, optimal_dir)
        alignments.append(alignment)
        
        if i < 5:
            print(f"  Leaf {i}: center_norm={np.linalg.norm(center):.2f}, "
                  f"grad_norm={grad_norm:.2f}, alignment={alignment:+.3f}")
    
    if alignments:
        mean_align = np.mean(alignments)
        print(f"\nMean alignment (descent vs optimal): {mean_align:+.3f}")
        
        if mean_align > 0.3:
            print("✅ GOOD: Gradients mostly point toward optimum")
        elif mean_align > 0:
            print("⚠️  WEAK: Gradients weakly point toward optimum")
        else:
            print("⚠️  BAD: Gradients don't point toward optimum!")


def analyze_potential_usage():
    """Analizza come il potential viene effettivamente usato"""
    print("\n" + "=" * 70)
    print("ANALYSIS: How Potential Field is Used in Leaf Selection")
    print("=" * 70)
    
    # Leggi il codice di leaf_selection per capire
    from alba_framework_potential import leaf_selection
    
    # Check if potential is used in selection
    import inspect
    source = inspect.getsource(leaf_selection)
    
    if "potential" in source.lower():
        print("✅ Potential field IS referenced in leaf_selection.py")
        
        # Count occurrences
        count = source.lower().count("potential")
        print(f"   Found {count} references to 'potential'")
    else:
        print("⚠️  Potential field is NOT used in leaf_selection!")


def test_potential_with_more_budget():
    """Test con budget maggiore per vedere se la correlazione migliora"""
    print("\n" + "=" * 70)
    print("ANALYSIS: Potential Correlation vs Budget")
    print("=" * 70)
    
    dim = 10
    bounds = [(-5.0, 5.0)] * dim
    func = make_sphere(dim)
    
    for budget in [100, 300, 500]:
        opt = ALBA(
            bounds=bounds,
            total_budget=budget,
            use_potential_field=True,
            use_coherence_gating=True,
            seed=42
        )
        
        for i in range(budget):
            x = opt.ask()
            y = func(x)
            opt.tell(x, y)
        
        if opt._coherence_tracker is None or len(opt.leaves) == 0:
            continue
        
        tracker = opt._coherence_tracker
        
        dists = []
        pots = []
        for i, leaf in enumerate(opt.leaves):
            dists.append(np.linalg.norm(leaf.center()))
            pots.append(tracker._cache.potentials.get(i, 0.5))
        
        corr, _ = spearmanr(dists, pots)
        print(f"  Budget {budget:4d}: {len(opt.leaves):2d} leaves, dist-pot corr = {corr:+.3f}")


if __name__ == "__main__":
    analyze_potential_vs_quality()
    analyze_gradient_vs_function()
    analyze_potential_usage()
    test_potential_with_more_budget()
    
    print("\n" + "=" * 70)
    print("DEEP DIVE COMPLETE")
    print("=" * 70)
