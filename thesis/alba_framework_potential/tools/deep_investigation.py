"""
ALBA Deep Investigation - Analisi dei casi problematici trovati

Focus su:
1. Perché tiny cubes non hanno modello LGS?
2. Perché Rastrigin si blocca?
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/workspace/thesis')
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube
from alba_framework_potential.lgs import fit_lgs_model


def trace_separator(title: str):
    print("\n" + "="*80)
    print(f"INVESTIGATION: {title}")
    print("="*80)


# =============================================================================
# INV1: Perché tiny cubes non hanno modello?
# =============================================================================
def investigate_tiny_cube_no_model():
    """
    I tiny cubes hanno model=None. Perché?
    Probabilmente perché hanno pochi punti.
    """
    trace_separator("TINY CUBE - NO MODEL")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    opt = ALBA(
        bounds=bounds, 
        seed=42, 
        total_budget=300,
        maximize=False,
        split_trials_min=5,
        split_trials_factor=1.0,
        split_depth_max=20,
    )
    
    print("\n[EXECUTION] Running with aggressive splitting")
    for i in range(200):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    print(f"\n[ANALYSIS] Analyzing all leaves")
    
    stats = {
        'has_model': 0,
        'no_model': 0,
        'no_model_reasons': [],
    }
    
    for i, leaf in enumerate(opt.leaves):
        has_model = leaf.lgs_model is not None
        
        if has_model:
            stats['has_model'] += 1
        else:
            stats['no_model'] += 1
            
            # Diagnose why
            n_pts = leaf.n_trials
            reason = f"n_trials={n_pts}"
            
            # Check minimum requirement: dim + 2
            if n_pts < dim + 2:
                reason += f" (< {dim+2} = dim+2)"
            
            # Check parent backfill
            if leaf.parent:
                parent_pts = len(getattr(leaf.parent, '_tested_pairs', []))
                reason += f", parent_pts={parent_pts}"
            
            stats['no_model_reasons'].append(reason)
            
            if len(stats['no_model_reasons']) <= 5:
                print(f"\n  Leaf {i}: NO MODEL")
                print(f"    n_trials = {n_pts}")
                print(f"    depth = {leaf.depth}")
                print(f"    volume = {leaf.volume():.2e}")
    
    print(f"\n[SUMMARY]")
    print(f"  Total leaves: {len(opt.leaves)}")
    print(f"  With model: {stats['has_model']}")
    print(f"  Without model: {stats['no_model']}")
    
    if stats['no_model_reasons']:
        print(f"\n  Reasons for no model:")
        for reason in stats['no_model_reasons'][:5]:
            print(f"    - {reason}")
    
    # The issue: when model=None, predict_bayesian returns (0, 1)
    # This means mu=0 and sigma=1, which doesn't use any learned info!
    print(f"\n[IMPLICATION]")
    print(f"  When model=None, predict_bayesian returns (mu=0, sigma=1)")
    print(f"  This means NO gradient info, NO uncertainty from data!")
    print(f"  The UCB becomes: 0 + 0.4 * 1 = 0.4 for all points")
    print(f"  → Selection is essentially random in these cubes")
    
    return True


# =============================================================================
# INV2: Cosa succede esattamente su Rastrigin?
# =============================================================================
def investigate_rastrigin():
    """
    Trace dettagliato del comportamento su Rastrigin.
    """
    trace_separator("RASTRIGIN DETAILED TRACE")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def rastrigin(x):
        x_shifted = (x - 0.5) * 5.12 * 2
        A = 10
        return A * dim + np.sum(x_shifted**2 - A * np.cos(2 * np.pi * x_shifted))
    
    print("\n[LANDSCAPE ANALYSIS]")
    # Campiona la funzione per capire il landscape
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(1000):
        x = rng.random(dim)
        y = rastrigin(x)
        samples.append((x, y))
    
    samples.sort(key=lambda t: t[1])
    
    print(f"  Best sample: y = {samples[0][1]:.4f} at {samples[0][0]}")
    print(f"  10th best: y = {samples[9][1]:.4f}")
    print(f"  Median: y = {samples[500][1]:.4f}")
    print(f"  Worst: y = {samples[-1][1]:.4f}")
    
    # Quanti minimi locali ci sono?
    local_minima = []
    for x, y in samples[:50]:  # Top 50
        is_local_min = True
        for x2, y2 in samples[:50]:
            if np.linalg.norm(x - x2) < 0.1 and y2 < y - 0.1:
                is_local_min = False
                break
        if is_local_min:
            local_minima.append((x, y))
    
    print(f"\n  Approximate local minima in top 50: {len(local_minima)}")
    for x, y in local_minima[:5]:
        dist_to_opt = np.linalg.norm(x - 0.5)
        print(f"    y = {y:.4f}, dist to [0.5] = {dist_to_opt:.4f}")
    
    # Run ALBA with tracing
    print("\n[ALBA EXECUTION]")
    opt = ALBA(bounds=bounds, seed=42, total_budget=100, maximize=False)
    
    best_history = []
    
    for i in range(80):
        x = opt.ask()
        y = rastrigin(x)
        opt.tell(x, y)
        
        if i == 0 or opt.best_y < best_history[-1]:
            best_history.append(opt.best_y)
            if len(best_history) <= 10:
                print(f"  Iter {i:2d}: New best = {opt.best_y:.4f}")
        else:
            best_history.append(best_history[-1])
    
    print(f"\n[CONVERGENCE]")
    print(f"  Initial best: {best_history[0]:.4f}")
    print(f"  Final best: {best_history[-1]:.4f}")
    print(f"  Improvement: {best_history[0] - best_history[-1]:.4f}")
    
    # Analyze gradient direction
    print(f"\n[GRADIENT ANALYSIS]")
    for leaf in opt.leaves:
        if leaf.lgs_model is not None:
            model = leaf.lgs_model
            grad = model.get('grad')
            grad_dir = model.get('gradient_dir')
            
            if grad_dir is not None:
                center = leaf.center()
                
                # La direzione corretta verso l'ottimo [0.5, 0.5, 0.5]
                optimal_dir = (0.5 - center) / (np.linalg.norm(0.5 - center) + 1e-9)
                
                # Normalizza in spazio cube
                widths = leaf.widths()
                optimal_dir_normalized = optimal_dir / widths
                optimal_dir_normalized = optimal_dir_normalized / (np.linalg.norm(optimal_dir_normalized) + 1e-9)
                
                alignment = np.dot(grad_dir, optimal_dir_normalized)
                
                print(f"\n  Leaf at {center}:")
                print(f"    grad_dir = {grad_dir}")
                print(f"    optimal_dir (norm) = {optimal_dir_normalized}")
                print(f"    alignment = {alignment:.4f}")
                
                if alignment > 0.5:
                    print(f"    ✅ Good alignment")
                elif alignment > 0:
                    print(f"    ⚠️ Weak alignment")
                else:
                    print(f"    ❌ Wrong direction!")
    
    return True


# =============================================================================
# INV3: Verifica la matematica del gradiente
# =============================================================================
def investigate_gradient_math():
    """
    Verifica che il gradiente calcolato sia matematicamente corretto.
    Confrontiamo con il gradiente numerico della funzione.
    """
    trace_separator("GRADIENT MATH VERIFICATION")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def quadratic(x):
        # Semplice funzione quadratica dove conosciamo il gradiente esatto
        # f(x) = (x - c)^T A (x - c) dove A = diag([1, 4, 9])
        c = np.array([0.3, 0.6, 0.4])
        A = np.diag([1, 4, 9])
        d = x - c
        return d @ A @ d
    
    def grad_quadratic(x):
        # ∇f = 2 A (x - c)
        c = np.array([0.3, 0.6, 0.4])
        A = np.diag([1, 4, 9])
        return 2 * A @ (x - c)
    
    # Crea un cube e riempilo di punti
    cube = Cube(bounds=list(bounds))
    rng = np.random.default_rng(42)
    
    for _ in range(30):
        x = rng.random(dim)
        # IMPORTANTE: dobbiamo passare lo score INTERNO (negato per minimize)
        y_internal = -quadratic(x)
        cube.add_observation(x, y_internal, gamma=-10.0)  # gamma negativo per internal
    
    print("\n[SETUP]")
    print(f"  Function: f(x) = (x - c)^T A (x - c)")
    print(f"  c = [0.3, 0.6, 0.4]")
    print(f"  A = diag([1, 4, 9])")
    
    # Fit LGS
    cube.fit_lgs_model(gamma=-10.0, dim=dim, rng=rng)
    model = cube.lgs_model
    
    if model is None:
        print("  ❌ No model fitted!")
        return False
    
    grad_lgs = model.get('grad')
    grad_dir = model.get('gradient_dir')
    
    print(f"\n[LGS GRADIENT]")
    print(f"  grad = {grad_lgs}")
    print(f"  grad_dir = {grad_dir}")
    
    # Il gradiente LGS è in spazio normalizzato.
    # Per confrontarlo col gradiente vero, dobbiamo:
    # 1. Valutare il gradiente vero al centro del cube
    # 2. Convertirlo in spazio normalizzato
    
    center = cube.center()
    widths = cube.widths()
    
    # Gradiente vero al centro (in spazio originale)
    true_grad_raw = grad_quadratic(center)
    
    # Ma stiamo minimizzando, quindi internamente massimizziamo -f
    # Il gradiente di -f è -∇f
    true_grad_internal = -true_grad_raw
    
    # Converti in spazio normalizzato
    # Se X_norm = (X - center) / widths, allora
    # ∂f/∂X_norm = widths * ∂f/∂X
    # Ma no, in realtà il gradiente in spazio normalizzato è:
    # ∂f_norm/∂X_norm dove f_norm = (f - f_mean) / f_std
    
    # Per LGS: grad = inv_cov @ X_norm.T @ W @ y_centered
    # Questo è diverso dal gradiente analitico!
    
    print(f"\n[TRUE GRADIENT at center]")
    print(f"  center = {center}")
    print(f"  ∇f(center) = {true_grad_raw}")
    print(f"  -∇f(center) (internal) = {true_grad_internal}")
    
    # Il gradiente LGS non è lo stesso del gradiente della funzione!
    # È il gradiente della superficie di regressione pesata.
    
    print(f"\n[COMPARISON]")
    
    # Normalizziamo entrambi e confrontiamo la direzione
    if np.linalg.norm(grad_lgs) > 1e-9:
        lgs_dir = grad_lgs / np.linalg.norm(grad_lgs)
    else:
        lgs_dir = np.zeros(dim)
    
    if np.linalg.norm(true_grad_internal) > 1e-9:
        # Convertiamo in spazio normalizzato
        # ∂f_internal/∂X_norm = ∂f_internal/∂X * ∂X/∂X_norm = true_grad_internal * widths
        true_grad_normalized = true_grad_internal * widths  # Jacobian
        true_dir = true_grad_normalized / np.linalg.norm(true_grad_normalized)
    else:
        true_dir = np.zeros(dim)
    
    alignment = np.dot(lgs_dir, true_dir)
    
    print(f"  LGS direction (normalized): {lgs_dir}")
    print(f"  True direction (normalized): {true_dir}")
    print(f"  Alignment: {alignment:.4f}")
    
    if alignment > 0.8:
        print(f"  ✅ LGS gradient well-aligned with true gradient")
    elif alignment > 0.5:
        print(f"  ⚠️ LGS gradient moderately aligned")
    elif alignment > 0:
        print(f"  ⚠️ LGS gradient weakly aligned")
    else:
        print(f"  ❌ LGS gradient OPPOSITE to true gradient!")
    
    return True


# =============================================================================
# INV4: Split decision bug?
# =============================================================================
def investigate_split_decision():
    """
    Verifica che la decisione di split sia corretta.
    """
    trace_separator("SPLIT DECISION LOGIC")
    
    dim = 5
    bounds = [(0, 1)] * dim
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    opt = ALBA(
        bounds=bounds, 
        seed=42, 
        total_budget=100,
        maximize=False,
    )
    
    print(f"\n[CONFIG]")
    print(f"  split_trials_min = {opt._split_trials_min}")
    print(f"  split_trials_factor = {opt._split_trials_factor}")
    print(f"  split_trials_offset = {opt._split_trials_offset}")
    print(f"  split_depth_max = {opt._split_depth_max}")
    
    threshold = opt._split_trials_factor * dim + opt._split_trials_offset
    print(f"\n  Split threshold = {opt._split_trials_factor} * {dim} + {opt._split_trials_offset} = {threshold:.1f}")
    
    split_events = []
    
    for i in range(80):
        n_leaves_before = len(opt.leaves)
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        n_leaves_after = len(opt.leaves)
        
        if n_leaves_after > n_leaves_before:
            split_events.append({
                'iter': i,
                'before': n_leaves_before,
                'after': n_leaves_after
            })
    
    print(f"\n[SPLIT EVENTS]")
    for ev in split_events:
        print(f"  Iter {ev['iter']:2d}: {ev['before']} → {ev['after']} leaves")
    
    print(f"\n[FINAL STATE]")
    print(f"  n_leaves = {len(opt.leaves)}")
    
    # Analyze leaf sizes
    leaf_trials = [leaf.n_trials for leaf in opt.leaves]
    print(f"  Leaf n_trials: min={min(leaf_trials)}, max={max(leaf_trials)}, mean={np.mean(leaf_trials):.1f}")
    
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALBA DEEP INVESTIGATION")
    print("="*80)
    
    investigations = [
        ("INV1: Tiny Cube No Model", investigate_tiny_cube_no_model),
        ("INV2: Rastrigin Behavior", investigate_rastrigin),
        ("INV3: Gradient Math", investigate_gradient_math),
        ("INV4: Split Decision", investigate_split_decision),
    ]
    
    results = {}
    for name, inv_fn in investigations:
        try:
            results[name] = inv_fn()
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
