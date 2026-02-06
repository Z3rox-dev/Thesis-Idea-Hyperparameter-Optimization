"""
ALBA Counter-Examples - Costruzione di casi che rompono le assunzioni

Dopo aver verificato che il flusso base funziona, cerchiamo edge cases
che potrebbero rompere il sistema.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/mnt/workspace/thesis')
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube


def trace_separator(title: str):
    print("\n" + "="*80)
    print(f"COUNTER-EXAMPLE: {title}")
    print("="*80)


# =============================================================================
# CE1: Gradiente completamente sbagliato
# =============================================================================
def ce_gradient_wrong_direction():
    """
    Caso: funzione con struttura ingannevole che porta
    il gradiente nella direzione sbagliata.
    
    Rastrigin: ha molti minimi locali che confondono il gradiente.
    """
    trace_separator("GRADIENT WRONG DIRECTION (Rastrigin)")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def rastrigin(x):
        # Ottimo globale a [0.5, 0.5, 0.5] (shiftato per stare in [0,1]^d)
        x_shifted = (x - 0.5) * 5.12 * 2  # Map to [-5.12, 5.12]
        A = 10
        return A * dim + np.sum(x_shifted**2 - A * np.cos(2 * np.pi * x_shifted))
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=100, maximize=False)
    
    print("\n[EXECUTION] Running on Rastrigin")
    for i in range(50):
        x = opt.ask()
        y = rastrigin(x)
        opt.tell(x, y)
    
    print(f"\n[RESULT]")
    print(f"  best_y = {opt.best_y:.4f}")
    print(f"  best_x = {opt.best_x}")
    print(f"  True optimum at [0.5, 0.5, 0.5] = 0.0")
    print(f"  Distance to optimum: {np.linalg.norm(opt.best_x - 0.5):.4f}")
    
    # Analizza gradiente nel root
    root = opt.root
    if root.lgs_model is not None:
        grad_dir = root.lgs_model.get('gradient_dir')
        if grad_dir is not None:
            # Ottimo a 0.5, quindi da qualsiasi punto il gradiente 
            # dovrebbe puntare verso 0.5
            center = root.center()
            expected_dir = (0.5 - center) / (np.linalg.norm(0.5 - center) + 1e-9)
            
            # Ma in spazio normalizzato, center = [0.5] sempre per root
            # quindi expected_dir = 0... Il test ha senso solo per leaves
            
            print(f"\n  Root gradient_dir = {grad_dir}")
            print(f"  ||gradient_dir|| = {np.linalg.norm(grad_dir):.4f}")
    
    # Verifichiamo se ALBA sta convergendo
    if opt.best_y < 10:
        print(f"\n  ⚠️ Good progress (y < 10)")
    else:
        print(f"\n  ❌ Stuck in local minimum (y >= 10)")
    
    return True


# =============================================================================
# CE2: Sigma collapse - exploration dies
# =============================================================================
def ce_sigma_collapse():
    """
    Caso: dopo molte iterazioni, sigma potrebbe collassare a ~0,
    eliminando l'esplorazione.
    """
    trace_separator("SIGMA COLLAPSE")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=200, maximize=False)
    
    sigma_history = []
    
    print("\n[EXECUTION] Running 150 iterations, tracking sigma")
    for i in range(150):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
        
        # Campiona sigma da alcune leaves
        if i % 10 == 0 and opt.leaves:
            sigmas = []
            for leaf in opt.leaves:
                if leaf.lgs_model is not None:
                    rng = np.random.default_rng(i)
                    test_pt = rng.random(dim)
                    _, sigma = leaf.predict_bayesian([test_pt])
                    sigmas.append(sigma[0])
            
            if sigmas:
                mean_sigma = np.mean(sigmas)
                min_sigma = np.min(sigmas)
                sigma_history.append({
                    'iter': i,
                    'mean_sigma': mean_sigma,
                    'min_sigma': min_sigma,
                    'n_leaves': len(opt.leaves)
                })
    
    print(f"\n[SIGMA EVOLUTION]")
    for sh in sigma_history[::3]:  # Every 3rd entry
        print(f"  Iter {sh['iter']:3d}: mean_σ = {sh['mean_sigma']:.4f}, min_σ = {sh['min_sigma']:.4f}, leaves = {sh['n_leaves']}")
    
    # Check for collapse
    if sigma_history:
        final_min_sigma = sigma_history[-1]['min_sigma']
        initial_mean_sigma = sigma_history[0]['mean_sigma']
        
        if final_min_sigma < 1e-6:
            print(f"\n  ❌ Sigma collapsed to {final_min_sigma:.2e}!")
            print(f"      This kills exploration.")
            return False
        elif final_min_sigma < initial_mean_sigma * 0.01:
            print(f"\n  ⚠️ Sigma decreased 100x ({initial_mean_sigma:.4f} → {final_min_sigma:.4f})")
        else:
            print(f"\n  ✅ Sigma maintained reasonable values")
    
    return True


# =============================================================================
# CE3: Cube too small - numerical issues
# =============================================================================
def ce_tiny_cube_prediction():
    """
    Caso: dopo molti split, i cubes diventano minuscoli.
    La predizione potrebbe avere problemi numerici.
    """
    trace_separator("TINY CUBE PREDICTION")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    # Force many splits
    opt = ALBA(
        bounds=bounds, 
        seed=42, 
        total_budget=300,
        maximize=False,
        split_trials_min=5,
        split_trials_factor=1.0,  # Very aggressive splitting
        split_depth_max=20,
    )
    
    print("\n[EXECUTION] Running with aggressive splitting")
    for i in range(200):
        x = opt.ask()
        y = sphere(x)
        opt.tell(x, y)
    
    print(f"\n[STATE]")
    print(f"  n_leaves = {len(opt.leaves)}")
    print(f"  best_y = {opt.best_y:.6f}")
    
    # Analyze smallest cube
    min_volume = float('inf')
    smallest_leaf = None
    for leaf in opt.leaves:
        vol = leaf.volume()
        if vol < min_volume:
            min_volume = vol
            smallest_leaf = leaf
    
    if smallest_leaf is not None:
        print(f"\n[SMALLEST CUBE]")
        print(f"  Volume = {min_volume:.2e}")
        widths = smallest_leaf.widths()
        print(f"  Widths = {widths}")
        print(f"  Min width = {widths.min():.2e}")
        
        # Try prediction in this cube
        if smallest_leaf.lgs_model is not None:
            rng = np.random.default_rng(123)
            test_pt = rng.random(dim) * widths + np.array([b[0] for b in smallest_leaf.bounds])
            mu, sigma = smallest_leaf.predict_bayesian([test_pt])
            
            print(f"\n  Test prediction:")
            print(f"    mu = {mu[0]:.4e}")
            print(f"    sigma = {sigma[0]:.4e}")
            
            if np.isnan(mu[0]) or np.isnan(sigma[0]):
                print(f"    ❌ NaN in prediction!")
                return False
            if np.isinf(mu[0]) or np.isinf(sigma[0]):
                print(f"    ❌ Inf in prediction!")
                return False
            
            print(f"    ✅ Prediction valid")
    
    return True


# =============================================================================
# CE4: Deceptive function - gradient ascent trap
# =============================================================================
def ce_deceptive_function():
    """
    Funzione deceptive: il gradiente punta verso un massimo locale,
    non verso il minimo globale.
    """
    trace_separator("DECEPTIVE FUNCTION")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def deceptive(x):
        # Due bacini: uno falso vicino a [0.2] e uno vero a [0.8]
        dist_false = np.linalg.norm(x - 0.2)
        dist_true = np.linalg.norm(x - 0.8)
        
        # Il falso minimo è più grande ma ha gradiente più forte
        # all'inizio
        false_basin = -10 * np.exp(-dist_false**2 / 0.01)
        true_basin = -0.1 - 20 * np.exp(-dist_true**2 / 0.01)
        
        return false_basin + true_basin + 1.0
    
    print("\n[FUNCTION ANALYSIS]")
    print(f"  deceptive([0.2, 0.2, 0.2]) = {deceptive(np.array([0.2]*dim)):.4f}")
    print(f"  deceptive([0.8, 0.8, 0.8]) = {deceptive(np.array([0.8]*dim)):.4f} (true minimum)")
    print(f"  deceptive([0.5, 0.5, 0.5]) = {deceptive(np.array([0.5]*dim)):.4f}")
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=100, maximize=False)
    
    print("\n[EXECUTION] Running optimization")
    for i in range(80):
        x = opt.ask()
        y = deceptive(x)
        opt.tell(x, y)
    
    print(f"\n[RESULT]")
    print(f"  best_y = {opt.best_y:.4f}")
    print(f"  best_x = {opt.best_x}")
    
    dist_to_false = np.linalg.norm(opt.best_x - 0.2)
    dist_to_true = np.linalg.norm(opt.best_x - 0.8)
    
    print(f"  Distance to FALSE minimum [0.2]: {dist_to_false:.4f}")
    print(f"  Distance to TRUE minimum [0.8]: {dist_to_true:.4f}")
    
    if dist_to_true < dist_to_false:
        print(f"\n  ✅ Found TRUE minimum!")
    else:
        print(f"\n  ⚠️ Stuck at FALSE minimum (deceptive trap)")
    
    return True


# =============================================================================
# CE5: High dimensionality - curse of dimensionality
# =============================================================================
def ce_high_dim():
    """
    In alta dimensionalità, il volume dei cubes cresce esponenzialmente
    e la copertura diventa sparsa.
    """
    trace_separator("HIGH DIMENSIONALITY (d=20)")
    
    dim = 20
    bounds = [(0, 1)] * dim
    
    def sphere_hd(x):
        return np.sum((x - 0.5)**2)
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=500, maximize=False)
    
    print(f"\n[SETUP]")
    print(f"  dim = {dim}")
    print(f"  budget = 500")
    print(f"  Points per dimension: {500/dim:.1f}")
    
    print("\n[EXECUTION]")
    for i in range(500):
        x = opt.ask()
        y = sphere_hd(x)
        opt.tell(x, y)
        
        if i % 100 == 99:
            print(f"  Iter {i+1}: best_y = {opt.best_y:.6f}, n_leaves = {len(opt.leaves)}")
    
    print(f"\n[RESULT]")
    print(f"  Final best_y = {opt.best_y:.6f}")
    print(f"  Distance to optimum: {np.linalg.norm(opt.best_x - 0.5):.4f}")
    print(f"  Random baseline (expected): ~{dim * (1/12):.4f}")  # E[U(0,1)^2] = 1/12 per dim
    
    if opt.best_y < dim * 0.05:
        print(f"\n  ✅ Better than random ({opt.best_y:.4f} < {dim*0.05:.4f})")
    else:
        print(f"\n  ⚠️ Not much better than random")
    
    return True


# =============================================================================
# CE6: Discontinuous function
# =============================================================================
def ce_discontinuous():
    """
    Funzione discontinua - il gradiente LGS non ha senso.
    """
    trace_separator("DISCONTINUOUS FUNCTION")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def step_function(x):
        # Funzione a gradini
        region = int(x[0] * 4) + int(x[1] * 4) * 4 + int(x[2] * 4) * 16
        return float(region % 7)  # Values 0-6
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=100, maximize=False)
    
    print("\n[EXECUTION] Running on step function")
    for i in range(80):
        x = opt.ask()
        y = step_function(x)
        opt.tell(x, y)
    
    print(f"\n[RESULT]")
    print(f"  best_y = {opt.best_y:.4f}")
    
    # Il minimo è 0.0
    if opt.best_y == 0.0:
        print(f"  ✅ Found global minimum (0)")
    else:
        print(f"  ⚠️ Didn't find global minimum")
    
    # Analyze gradient
    root = opt.root
    if root.lgs_model is not None:
        grad = root.lgs_model.get('grad')
        print(f"\n  Root gradient = {grad}")
        print(f"  ||grad|| = {np.linalg.norm(grad):.4f}")
        print(f"  (Gradient is meaningless for discontinuous function)")
    
    return True


# =============================================================================
# CE7: All observations identical
# =============================================================================
def ce_identical_observations():
    """
    Edge case: cosa succede se per caso tutti i punti hanno lo stesso y?
    """
    trace_separator("IDENTICAL OBSERVATIONS")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def constant_42(x):
        return 42.0
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=50, maximize=False)
    
    print("\n[EXECUTION] Running with constant function")
    for i in range(30):
        x = opt.ask()
        y = constant_42(x)
        opt.tell(x, y)
    
    print(f"\n[STATE]")
    print(f"  best_y = {opt.best_y:.4f}")
    print(f"  gamma = {opt.gamma:.4f}")
    print(f"  n_leaves = {len(opt.leaves)}")
    
    # Analyze LGS
    root = opt.root
    if root.lgs_model is not None:
        grad = root.lgs_model.get('grad')
        y_std = root.lgs_model.get('y_std')
        print(f"\n  y_std = {y_std:.2e}")
        print(f"  grad = {grad}")
        print(f"  ||grad|| = {np.linalg.norm(grad):.2e}")
        
        if np.linalg.norm(grad) < 1e-6:
            print(f"  ✅ Gradient correctly ~0 for constant function")
        else:
            print(f"  ⚠️ Non-zero gradient for constant function!")
    
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALBA COUNTER-EXAMPLES")
    print("="*80)
    
    tests = [
        ("CE1: Gradient Wrong Direction", ce_gradient_wrong_direction),
        ("CE2: Sigma Collapse", ce_sigma_collapse),
        ("CE3: Tiny Cube Prediction", ce_tiny_cube_prediction),
        ("CE4: Deceptive Function", ce_deceptive_function),
        ("CE5: High Dimensionality", ce_high_dim),
        ("CE6: Discontinuous Function", ce_discontinuous),
        ("CE7: Identical Observations", ce_identical_observations),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
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
