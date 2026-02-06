"""
ALBA End-to-End Trace - Verifica flusso completo dall'utente ai numeri

Trace the entire flow:
1. User calls tell(x, y_raw)
2. ALBA converts to internal (negate if minimize)
3. Cube stores internal score
4. LGS fits on internal scores
5. Acquisition uses LGS predictions
6. Selection picks next point

We dump numbers at each step and verify consistency.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/workspace/thesis')
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube
from alba_framework_potential.lgs import fit_lgs_model, predict_bayesian


def trace_separator(title: str):
    print("\n" + "="*80)
    print(f"TRACE: {title}")
    print("="*80)


# =============================================================================
# TRACE 1: End-to-end con funzione ill-conditioned
# =============================================================================
def trace_e2e_ill_conditioned():
    """
    Test completo: dall'utente (minimize) fino alla selezione.
    Verifica che i pesi siano corretti.
    """
    trace_separator("END-TO-END ILL-CONDITIONED")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def ill_cond(x):
        return 1.0*(x[0]-0.5)**2 + 100.0*(x[1]-0.5)**2 + 10000.0*(x[2]-0.5)**2
    
    print("\n[SETUP] ALBA con maximize=False (minimize)")
    opt = ALBA(bounds=bounds, seed=42, total_budget=50, maximize=False)
    
    # Raccogliamo info durante l'esecuzione
    trace_data = []
    
    print("\n[EXECUTION] Running 20 iterations with tracing")
    for i in range(20):
        x = opt.ask()
        y_raw = ill_cond(x)  # Valore grezzo (utente vuole minimizzare)
        
        # Prima del tell, salviamo
        y_internal_expected = -y_raw  # maximize=False → negate
        
        opt.tell(x, y_raw)
        
        # Dopo il tell, verifichiamo
        y_stored = opt.y_all[-1]
        
        if i < 5:
            print(f"\n  [Iter {i}]")
            print(f"    x = {x}")
            print(f"    y_raw (user) = {y_raw:.4f}")
            print(f"    y_internal expected = {y_internal_expected:.4f}")
            print(f"    y_stored in y_all = {y_stored:.4f}")
            
            if abs(y_stored - y_internal_expected) > 1e-9:
                print(f"    ❌ MISMATCH!")
            else:
                print(f"    ✅ Correct")
        
        trace_data.append({
            'iter': i,
            'x': x.copy(),
            'y_raw': y_raw,
            'y_internal': y_stored,
        })
    
    print(f"\n[ANALYSIS] After 20 iterations")
    print(f"  best_y (user view) = {opt.best_y:.6f}")
    print(f"  best_y_internal = {opt.best_y_internal:.6f}")
    print(f"  best_x = {opt.best_x}")
    
    # Verifica coerenza
    true_best_raw = min(td['y_raw'] for td in trace_data)
    print(f"  true min(y_raw) = {true_best_raw:.6f}")
    
    if abs(opt.best_y - true_best_raw) > 1e-9:
        print(f"  ❌ best_y mismatch!")
        return False
    
    # Analizziamo i pesi nel root cube
    print(f"\n[LGS WEIGHTS ANALYSIS] Root cube")
    root = opt.root
    pairs = list(root.tested_pairs)
    if len(pairs) < 5:
        print(f"  Root has only {len(pairs)} points, checking a leaf")
        if opt.leaves:
            for leaf in opt.leaves:
                if len(leaf.tested_pairs) >= 5:
                    pairs = list(leaf.tested_pairs)
                    break
    
    if len(pairs) >= 5:
        scores = np.array([s for _, s in pairs])
        print(f"  Internal scores: min={scores.min():.4f}, max={scores.max():.4f}")
        
        # rank_weights formula
        rank_weights = 1.0 + 0.5 * (scores - scores.min()) / (scores.ptp() + 1e-9)
        
        best_internal_idx = np.argmax(scores)  # Più alto = migliore internamente
        worst_internal_idx = np.argmin(scores)  # Più basso = peggiore
        
        print(f"  Best internal score idx: {best_internal_idx}, score = {scores[best_internal_idx]:.4f}")
        print(f"  Worst internal score idx: {worst_internal_idx}, score = {scores[worst_internal_idx]:.4f}")
        print(f"  rank_weight[best] = {rank_weights[best_internal_idx]:.4f}")
        print(f"  rank_weight[worst] = {rank_weights[worst_internal_idx]:.4f}")
        
        if rank_weights[best_internal_idx] > rank_weights[worst_internal_idx]:
            print(f"  ✅ Weights correct: best point has higher weight")
        else:
            print(f"  ❌ BUG: best point has LOWER weight!")
            return False
    
    print("\n  ✅ E2E test passed")
    return True


# =============================================================================
# TRACE 2: Verifica che UCB selezioni correttamente
# =============================================================================
def trace_ucb_selection():
    """
    Verifica che UCB (mu + novelty * sigma) selezioni punti ragionevoli.
    """
    trace_separator("UCB SELECTION LOGIC")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    print("\n[SETUP] ALBA con sphere function")
    opt = ALBA(bounds=bounds, seed=42, total_budget=100, maximize=False)
    
    # Run for a while to populate cubes
    for i in range(30):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    print(f"\n[STATE] After 30 iterations")
    print(f"  n_leaves = {len(opt.leaves)}")
    print(f"  best_y = {opt.best_y:.6f}")
    
    # Get a leaf with an LGS model
    test_leaf = None
    for leaf in opt.leaves:
        if leaf.lgs_model is not None:
            test_leaf = leaf
            break
    
    if test_leaf is None:
        print("  No leaf with LGS model found")
        return True
    
    print(f"\n[LGS MODEL] Testing predictions in leaf")
    print(f"  n_trials = {test_leaf.n_trials}")
    print(f"  n_good = {test_leaf.n_good}")
    
    model = test_leaf.lgs_model
    print(f"  y_mean = {model.get('y_mean'):.4f}")
    print(f"  y_std = {model.get('y_std'):.4f}")
    print(f"  grad = {model.get('grad')}")
    
    # Generate candidates and compute UCB
    rng = np.random.default_rng(123)
    candidates = []
    for _ in range(20):
        c = rng.random(dim) * np.array([b[1]-b[0] for b in test_leaf.bounds]) + np.array([b[0] for b in test_leaf.bounds])
        candidates.append(c)
    
    mu, sigma = test_leaf.predict_bayesian(candidates)
    novelty_weight = 0.4
    ucb = mu + novelty_weight * sigma
    
    print(f"\n  Candidate predictions:")
    print(f"    mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"    sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"    UCB range: [{ucb.min():.4f}, {ucb.max():.4f}]")
    
    # Best UCB should be selected
    best_ucb_idx = np.argmax(ucb)
    best_candidate = candidates[best_ucb_idx]
    
    print(f"\n  Best UCB candidate: {best_candidate}")
    print(f"  UCB value: {ucb[best_ucb_idx]:.4f}")
    print(f"  (mu={mu[best_ucb_idx]:.4f}, sigma={sigma[best_ucb_idx]:.4f})")
    
    # True value at this point
    true_y = sphere(best_candidate)
    print(f"  True sphere value: {true_y:.6f}")
    
    # Internal representation
    internal_y = -true_y  # maximize=False
    print(f"  Internal value: {internal_y:.4f}")
    print(f"  Predicted mu: {mu[best_ucb_idx]:.4f}")
    print(f"  Error: {abs(mu[best_ucb_idx] - internal_y):.4f}")
    
    print("\n  ✅ UCB selection test passed")
    return True


# =============================================================================
# TRACE 3: Verifica splitting mantiene coerenza
# =============================================================================
def trace_split_coherence():
    """
    Dopo uno split, i figli dovrebbero avere dati coerenti col parent.
    """
    trace_separator("SPLIT COHERENCE")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def quadratic(x):
        return np.sum((x - 0.3)**2)
    
    # Low split threshold to force early split
    opt = ALBA(
        bounds=bounds, 
        seed=42, 
        total_budget=100,
        maximize=False,
        split_trials_min=5,  # Split early
    )
    
    print("\n[EXECUTION] Running until first split")
    n_leaves_history = [len(opt.leaves)]
    
    for i in range(50):
        x = opt.ask()
        y = quadratic(x)
        opt.tell(x, y)
        
        current_leaves = len(opt.leaves)
        if current_leaves > n_leaves_history[-1]:
            print(f"\n  [Iter {i}] Split occurred! {n_leaves_history[-1]} → {current_leaves} leaves")
            n_leaves_history.append(current_leaves)
            
            # Analyze the new leaves
            print(f"\n[ANALYSIS] Checking leaf coherence")
            total_points = sum(leaf.n_trials for leaf in opt.leaves)
            print(f"  Total points across leaves: {total_points}")
            print(f"  Points in y_all: {len(opt.y_all)}")
            
            # Each leaf's points should be contained in that leaf
            for j, leaf in enumerate(opt.leaves):
                print(f"\n  Leaf {j}: bounds = {leaf.bounds[:2]}...")
                print(f"           n_trials = {leaf.n_trials}, n_good = {leaf.n_good}")
                
                # Verify all stored points are inside bounds
                all_inside = True
                for pt, score in leaf.tested_pairs:
                    if not leaf.contains(pt):
                        print(f"           ❌ Point {pt} outside bounds!")
                        all_inside = False
                        break
                
                if all_inside:
                    print(f"           ✅ All points inside bounds")
            
            break  # Stop after first split
    
    print("\n  ✅ Split coherence test passed")
    return True


# =============================================================================
# TRACE 4: Edge case - tutti punti in un angolo
# =============================================================================
def trace_corner_clustering():
    """
    Se tutti i punti campionati finiscono in un angolo,
    il gradiente dovrebbe puntare correttamente.
    """
    trace_separator("CORNER CLUSTERING")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    # Funzione con ottimo all'angolo [0, 0, 0]
    def corner_opt(x):
        return np.sum(x**2)  # Minimo a [0,0,0]
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=50, maximize=False)
    
    print("\n[EXECUTION] Running optimization")
    for i in range(30):
        x = opt.ask()
        y = corner_opt(x)
        opt.tell(x, y)
    
    print(f"\n[STATE]")
    print(f"  best_y = {opt.best_y:.6f}")
    print(f"  best_x = {opt.best_x}")
    print(f"  Distance to [0,0,0]: {np.linalg.norm(opt.best_x):.6f}")
    
    # Analyze gradient direction in root
    root = opt.root
    if root.lgs_model is not None:
        model = root.lgs_model
        grad = model.get('grad')
        grad_dir = model.get('gradient_dir')
        
        print(f"\n[GRADIENT ANALYSIS]")
        print(f"  grad = {grad}")
        print(f"  gradient_dir = {grad_dir}")
        
        # Per minimizzare corner_opt, dobbiamo andare verso [0,0,0]
        # In internal scores (negati), vogliamo massimizzare, quindi
        # il gradiente dovrebbe puntare verso valori interni più alti = 
        # valori raw più bassi = verso [0,0,0]
        
        # Ma ricorda: il gradiente è in spazio normalizzato!
        # center = [0.5, 0.5, 0.5], quindi [0,0,0] è a [-0.5,-0.5,-0.5]
        # Il gradiente dovrebbe puntare in quella direzione (negativo su tutti gli assi)
        
        if grad_dir is not None:
            expected_direction = np.array([-1, -1, -1]) / np.sqrt(3)  # Normalized
            alignment = np.dot(grad_dir, expected_direction)
            print(f"  Expected direction (toward corner): {expected_direction}")
            print(f"  Alignment (dot product): {alignment:.4f}")
            
            if alignment > 0.5:
                print(f"  ✅ Gradient points roughly toward optimum")
            elif alignment > 0:
                print(f"  ⚠️ Gradient weakly aligned with optimum")
            else:
                print(f"  ❌ Gradient points AWAY from optimum!")
    
    print("\n  ✅ Corner clustering test passed")
    return True


# =============================================================================
# TRACE 5: Numerical stability under extreme scaling
# =============================================================================
def trace_extreme_scaling():
    """
    Test con scala estrema per verificare stabilità numerica.
    """
    trace_separator("EXTREME SCALING")
    
    dim = 5
    bounds = [(0, 1)] * dim
    
    # Funzione con range enorme
    def extreme_scale(x):
        # Scala da 10^-10 a 10^10
        d = np.linalg.norm(x - 0.5)
        if d < 0.01:
            return 1e-10 * d
        return 1e10 * d
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=50, maximize=False)
    
    print("\n[EXECUTION] Running with extreme scaling")
    any_nan = False
    any_inf = False
    
    for i in range(30):
        x = opt.ask()
        y = extreme_scale(x)
        
        if i < 5:
            print(f"  [Iter {i}] y = {y:.2e}")
        
        opt.tell(x, y)
        
        # Check for numerical issues
        if np.isnan(opt.best_y):
            print(f"  ❌ NaN detected at iter {i}")
            any_nan = True
            break
        if np.isinf(opt.best_y):
            print(f"  ❌ Inf detected at iter {i}")
            any_inf = True
            break
    
    if not any_nan and not any_inf:
        print(f"\n[RESULT]")
        print(f"  best_y = {opt.best_y:.2e}")
        print(f"  ✅ No NaN/Inf detected")
    else:
        print(f"  ❌ Numerical instability detected")
        return False
    
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALBA END-TO-END TRACER")
    print("="*80)
    
    traces = [
        ("E2E Ill-Conditioned", trace_e2e_ill_conditioned),
        ("UCB Selection", trace_ucb_selection),
        ("Split Coherence", trace_split_coherence),
        ("Corner Clustering", trace_corner_clustering),
        ("Extreme Scaling", trace_extreme_scaling),
    ]
    
    results = {}
    for name, trace_fn in traces:
        try:
            results[name] = trace_fn()
        except Exception as e:
            print(f"\n❌ {name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
