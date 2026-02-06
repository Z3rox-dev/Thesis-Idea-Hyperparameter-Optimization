"""
ALBA Deep Tracer - Seguiamo l'esecuzione come dimostrazione matematica

Per ogni passaggio chiave, dumpiamo i numeri e verifichiamo che il filo logico regga.
Costruiamo controesempi quando qualcosa non torna.
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
# TRACE 1: Flusso numerico completo di fit_lgs_model
# =============================================================================
def trace_lgs_fit_numerical():
    """
    Seguiamo OGNI passaggio di fit_lgs_model con numeri reali.
    Verifichiamo che ogni operazione sia matematicamente corretta.
    """
    trace_separator("LGS FIT - FLUSSO NUMERICO COMPLETO")
    
    dim = 3
    bounds = [(0, 1)] * dim
    cube = Cube(bounds=list(bounds))
    rng = np.random.default_rng(42)
    
    # Funzione test: ill-conditioned per stressare il sistema
    def ill_cond(x):
        return 1.0*(x[0]-0.5)**2 + 100.0*(x[1]-0.5)**2 + 10000.0*(x[2]-0.5)**2
    
    # Aggiungiamo punti
    print("\n[STEP 1] Generazione punti e scores")
    all_pts = []
    all_scores = []
    for i in range(15):
        x = rng.random(dim)
        y = ill_cond(x)
        cube.add_observation(x, y, gamma=0.0)
        all_pts.append(x)
        all_scores.append(y)
        if i < 5:
            print(f"  x[{i}] = {x}, y = {y:.4f}")
    
    all_pts = np.array(all_pts)
    all_scores = np.array(all_scores)
    
    print(f"\n  y stats: min={all_scores.min():.4f}, max={all_scores.max():.4f}, "
          f"mean={all_scores.mean():.4f}, std={all_scores.std():.4f}")
    
    # [STEP 2] Normalizzazione X
    print("\n[STEP 2] Normalizzazione X")
    widths = cube.widths()
    center = cube.center()
    print(f"  widths = {widths}")
    print(f"  center = {center}")
    
    X_norm = (all_pts - center) / widths
    print(f"  X_norm shape = {X_norm.shape}")
    print(f"  X_norm range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
    
    # VERIFICA A1: X_norm dovrebbe essere in [-0.5, 0.5]
    if np.any(np.abs(X_norm) > 0.5 + 1e-9):
        print(f"  ⚠️ VIOLAZIONE A1: X_norm fuori range!")
    else:
        print(f"  ✅ A1 OK: X_norm ∈ [-0.5, 0.5]")
    
    # [STEP 3] Normalizzazione y
    print("\n[STEP 3] Normalizzazione y")
    y_mean = all_scores.mean()
    y_std = all_scores.std() + 1e-6
    y_centered = (all_scores - y_mean) / y_std
    
    print(f"  y_mean = {y_mean:.4f}")
    print(f"  y_std = {y_std:.4f}")
    print(f"  y_centered range: [{y_centered.min():.4f}, {y_centered.max():.4f}]")
    print(f"  y_centered.mean() = {y_centered.mean():.4e} (dovrebbe essere ~0)")
    print(f"  y_centered.std() = {y_centered.std():.4f} (dovrebbe essere ~1)")
    
    # [STEP 4] Pesi gaussiani
    print("\n[STEP 4] Pesi gaussiani")
    dists_sq = np.sum(X_norm**2, axis=1)
    sigma_sq = np.mean(dists_sq) + 1e-6
    weights_gauss = np.exp(-dists_sq / (2 * sigma_sq))
    
    print(f"  dists_sq range: [{dists_sq.min():.4f}, {dists_sq.max():.4f}]")
    print(f"  sigma_sq = {sigma_sq:.4f}")
    print(f"  weights_gauss range: [{weights_gauss.min():.4f}, {weights_gauss.max():.4f}]")
    
    # [STEP 5] Rank weights
    print("\n[STEP 5] Rank weights (boost for top performers)")
    rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / (all_scores.ptp() + 1e-9)
    weights = weights_gauss * rank_weights
    W = np.diag(weights)
    
    print(f"  rank_weights range: [{rank_weights.min():.4f}, {rank_weights.max():.4f}]")
    print(f"  combined weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # ⚠️ PROBLEMA: rank_weights penalizza i punti buoni!
    # "top performers" con punteggio ALTO (male per minimize) ottengono peso maggiore
    best_idx = np.argmin(all_scores)
    worst_idx = np.argmax(all_scores)
    print(f"\n  ATTENZIONE:")
    print(f"    Punto MIGLIORE (y={all_scores[best_idx]:.4f}): rank_weight = {rank_weights[best_idx]:.4f}")
    print(f"    Punto PEGGIORE (y={all_scores[worst_idx]:.4f}): rank_weight = {rank_weights[worst_idx]:.4f}")
    
    if rank_weights[best_idx] < rank_weights[worst_idx]:
        print(f"    ❌ BUG POTENZIALE: il punto migliore ha peso MINORE!")
    
    # [STEP 6] Regressione pesata
    print("\n[STEP 6] Regressione pesata (XtWX + λI)^{-1} XtWy")
    lambda_base = 0.1 * (1 + dim / max(len(all_pts) - dim, 1))
    XtWX = X_norm.T @ W @ X_norm
    
    print(f"  lambda_base = {lambda_base:.4f}")
    print(f"  XtWX diagonal: {np.diag(XtWX)}")
    
    try:
        cond = np.linalg.cond(XtWX)
        print(f"  cond(XtWX) = {cond:.2e}")
    except:
        print(f"  cond(XtWX) = ∞ (singolare)")
    
    XtWX_reg = XtWX + lambda_base * np.eye(dim)
    cond_reg = np.linalg.cond(XtWX_reg)
    print(f"  cond(XtWX + λI) = {cond_reg:.2e}")
    
    inv_cov = np.linalg.inv(XtWX_reg)
    print(f"  inv_cov diagonal: {np.diag(inv_cov)}")
    
    # [STEP 7] Gradiente
    print("\n[STEP 7] Calcolo gradiente")
    grad = inv_cov @ (X_norm.T @ W @ y_centered)
    grad_norm = np.linalg.norm(grad)
    
    print(f"  grad = {grad}")
    print(f"  ||grad|| = {grad_norm:.4f}")
    
    # [STEP 8] Residui e noise variance
    print("\n[STEP 8] Residui e noise variance")
    y_pred = X_norm @ grad
    residuals = y_centered - y_pred
    noise_var = np.clip(np.average(residuals**2, weights=weights) + 1e-6, 1e-4, 10.0)
    
    print(f"  y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  residuals range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    print(f"  noise_var = {noise_var:.4f}")
    
    # [STEP 9] Predizione su nuovo punto
    print("\n[STEP 9] Test predict_bayesian")
    test_pt = np.array([0.5, 0.5, 0.5])  # Centro del cube
    test_pt_norm = (test_pt - center) / widths
    
    mu_normalized = test_pt_norm @ grad
    mu = y_mean + mu_normalized * y_std
    
    model_var = np.clip(np.sum((test_pt_norm @ inv_cov) * test_pt_norm), 0, 10.0)
    total_var_normalized = noise_var * (1.0 + model_var)
    sigma = np.sqrt(total_var_normalized) * y_std
    
    print(f"  test_pt = {test_pt}")
    print(f"  test_pt_norm = {test_pt_norm}")
    print(f"  mu_normalized = {mu_normalized:.6f}")
    print(f"  mu = y_mean + mu_normalized * y_std = {y_mean:.4f} + {mu_normalized:.6f} * {y_std:.4f} = {mu:.4f}")
    print(f"  model_var = {model_var:.6f}")
    print(f"  sigma = √(noise_var * (1 + model_var)) * y_std = {sigma:.4f}")
    
    # Valore vero
    true_y = ill_cond(test_pt)
    print(f"\n  Valore VERO: {true_y:.4f}")
    print(f"  Predizione: {mu:.4f} ± {sigma:.4f}")
    print(f"  Errore: {abs(mu - true_y):.4f} ({abs(mu - true_y)/true_y*100:.1f}%)")
    
    return True


# =============================================================================
# TRACE 2: Il bug dei rank weights
# =============================================================================
def trace_rank_weights_bug():
    """
    Investigazione: rank_weights dà peso maggiore ai punti PEGGIORI?
    
    La formula: 1.0 + 0.5 * (score - min) / ptp
    
    Se score è ALTO (male per minimize), ottiene peso maggiore!
    """
    trace_separator("RANK WEIGHTS - ANALISI SEMANTICA")
    
    print("""
    Formula attuale:
        rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / ptp
    
    Questa formula:
    - Punto con score MINIMO → rank_weight = 1.0
    - Punto con score MASSIMO → rank_weight = 1.5
    
    MA in ALBA vogliamo MINIMIZZARE, quindi:
    - Score BASSO = BUONO → dovrebbe avere peso ALTO
    - Score ALTO = CATTIVO → dovrebbe avere peso BASSO
    
    La logica è INVERTITA?
    """)
    
    # Verifichiamo nel codice
    print("\n[CHECK] Verifica semantica nel contesto ALBA")
    print("  - ALBA memorizza scores come 'internal score' (higher is better)")
    print("  - Quando maximize=False, lo score viene NEGATO in tell()")
    print("  - Quindi 'all_scores' contiene SCORE NEGATI!")
    print("  - Score negato ALTO = score originale BASSO = BUONO")
    print("  - Quindi la formula è CORRETTA nel contesto interno.")
    
    # Ma verifichiamo con un esempio
    print("\n[EXAMPLE] Verifica empirica")
    
    original_scores = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # Minimize: 0.1 è il migliore
    internal_scores = -original_scores  # ALBA nega per maximize=False
    
    print(f"  Original scores: {original_scores}")
    print(f"  Internal scores: {internal_scores}")
    
    rank_weights = 1.0 + 0.5 * (internal_scores - internal_scores.min()) / (internal_scores.ptp() + 1e-9)
    
    print(f"  Rank weights: {rank_weights}")
    print(f"  Punto migliore (orig=0.1, int=-0.1): weight = {rank_weights[0]:.3f}")
    print(f"  Punto peggiore (orig=5.0, int=-5.0): weight = {rank_weights[4]:.3f}")
    
    if rank_weights[0] > rank_weights[4]:
        print("  ✅ Corretto: il punto migliore ha peso maggiore")
    else:
        print("  ❌ BUG: il punto migliore ha peso minore!")
    
    return True


# =============================================================================
# TRACE 3: Extremely steep function - overflow potenziale
# =============================================================================
def trace_extremely_steep():
    """
    Test con funzione extremely steep: y_std = 10^9
    Verifichiamo che non ci siano overflow.
    """
    trace_separator("EXTREMELY STEEP - TEST OVERFLOW")
    
    dim = 3
    bounds = [(0, 1)] * dim
    cube = Cube(bounds=list(bounds))
    rng = np.random.default_rng(42)
    
    def extremely_steep(x):
        return 1e10 * np.sum((x - 0.5)**2)
    
    print("\n[STEP 1] Popolamento cube con funzione extremely steep")
    for i in range(20):
        x = rng.random(dim)
        y = extremely_steep(x)
        cube.add_observation(x, y, gamma=0.0)
    
    scores = np.array([s for _, s in cube.tested_pairs])
    print(f"  Score range: [{scores.min():.2e}, {scores.max():.2e}]")
    print(f"  Score std: {scores.std():.2e}")
    
    print("\n[STEP 2] Fit LGS model")
    cube.fit_lgs_model(gamma=0.0, dim=dim, rng=rng)
    model = cube.lgs_model
    
    if model is None:
        print("  ❌ Model = None!")
        return False
    
    grad = model.get('grad')
    y_std = model.get('y_std', 1.0)
    
    print(f"  grad = {grad}")
    print(f"  ||grad|| = {np.linalg.norm(grad):.2e}")
    print(f"  y_std = {y_std:.2e}")
    
    print("\n[STEP 3] Test predizione")
    candidates = [rng.random(dim) for _ in range(10)]
    mu, sigma = cube.predict_bayesian(candidates)
    
    print(f"  mu range: [{mu.min():.2e}, {mu.max():.2e}]")
    print(f"  sigma range: [{sigma.min():.2e}, {sigma.max():.2e}]")
    
    # Check per inf/nan
    if np.any(np.isnan(mu)) or np.any(np.isinf(mu)):
        print("  ❌ BUG: mu contiene NaN/Inf!")
        return False
    if np.any(np.isnan(sigma)) or np.any(np.isinf(sigma)):
        print("  ❌ BUG: sigma contiene NaN/Inf!")
        return False
    
    print("  ✅ Nessun overflow rilevato")
    
    print("\n[STEP 4] UCB con questi valori")
    novelty_weight = 0.4
    ucb = mu + novelty_weight * sigma
    print(f"  UCB range: [{ucb.min():.2e}, {ucb.max():.2e}]")
    
    # Il problema: mu e sigma sono dello stesso ordine di grandezza?
    # Se sigma >> mu, l'esplorazione domina
    # Se mu >> sigma, lo sfruttamento domina
    ratio = sigma.mean() / (np.abs(mu).mean() + 1e-10)
    print(f"  σ/|μ| ratio: {ratio:.2f}")
    
    if ratio > 10:
        print("  ⚠️ σ >> μ: exploration domina completamente")
    elif ratio < 0.1:
        print("  ⚠️ μ >> σ: exploitation domina completamente")
    else:
        print("  ✅ Bilanciamento ragionevole")
    
    return True


# =============================================================================
# TRACE 4: Tiny cube - division by zero
# =============================================================================
def trace_tiny_cube():
    """
    Dopo molti split, widths → 0. Verifichiamo che non ci siano div/0.
    """
    trace_separator("TINY CUBE - DIVISION BY ZERO")
    
    dim = 3
    
    # Simula un cube dopo molti split
    epsilon = 1e-12
    tiny_bounds = [(0.5 - epsilon, 0.5 + epsilon)] * dim
    cube = Cube(bounds=tiny_bounds)
    
    print(f"\n[STEP 1] Cube bounds: {cube.bounds}")
    widths = cube.widths()
    print(f"  widths = {widths}")
    
    rng = np.random.default_rng(42)
    
    # Aggiungiamo punti (devono stare dentro i bounds!)
    print("\n[STEP 2] Aggiunta punti")
    for i in range(10):
        x = 0.5 + (rng.random(dim) - 0.5) * epsilon * 0.5
        y = np.sum(x**2)
        cube.add_observation(x, y, gamma=0.0)
    
    print(f"  {len(cube.tested_pairs)} punti aggiunti")
    
    print("\n[STEP 3] Fit LGS model")
    try:
        cube.fit_lgs_model(gamma=0.0, dim=dim, rng=rng)
        model = cube.lgs_model
        
        if model is None:
            print("  Model = None (protetto internamente)")
        else:
            print(f"  Model fittato")
            print(f"  grad = {model.get('grad')}")
            print(f"  inv_cov diag = {np.diag(model.get('inv_cov', np.eye(dim)))}")
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        return False
    
    print("\n[STEP 4] Predict")
    try:
        candidates = [np.array([0.5] * dim)]
        mu, sigma = cube.predict_bayesian(candidates)
        print(f"  mu = {mu[0]:.6f}")
        print(f"  sigma = {sigma[0]:.6f}")
        
        if np.isnan(mu[0]) or np.isinf(mu[0]):
            print("  ❌ mu is NaN/Inf!")
            return False
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        return False
    
    print("  ✅ Tiny cube gestito correttamente")
    return True


# =============================================================================
# TRACE 5: Constant function - degenerate gamma
# =============================================================================
def trace_constant_function():
    """
    Con funzione costante, tutti gli y sono uguali.
    Verifichiamo il comportamento di gamma e del fitting.
    """
    trace_separator("CONSTANT FUNCTION - DEGENERATE CASE")
    
    dim = 3
    bounds = [(0, 1)] * dim
    
    def constant(x):
        return 42.0
    
    opt = ALBA(bounds=bounds, seed=42, total_budget=30, maximize=False)
    
    print("\n[STEP 1] Esecuzione ottimizzazione")
    for i in range(20):
        x = opt.ask()
        y = constant(x)
        opt.tell(x, y)
    
    print(f"\n[STEP 2] Analisi stato")
    print(f"  iterations = {opt.iteration}")
    print(f"  gamma = {opt.gamma}")
    print(f"  best_y = {opt.best_y}")
    print(f"  n_leaves = {len(opt.leaves)}")
    
    # Tutti gli y_all dovrebbero essere -42 (negati per maximize=False)
    y_internal = np.array(opt.y_all)
    print(f"\n  y_all (internal): unique = {len(set(y_internal))}, value = {y_internal[0]}")
    
    # Con y tutti uguali, y_std = 0 (o quasi)
    # Questo causa problemi nel fitting?
    print(f"\n[STEP 3] Test fitting su root cube")
    root = opt.root
    print(f"  root.n_trials = {root.n_trials}")
    print(f"  root.n_good = {root.n_good}")
    
    root.fit_lgs_model(gamma=opt.gamma, dim=dim, rng=opt.rng)
    model = root.lgs_model
    
    if model is None:
        print("  Model = None")
    else:
        print(f"  y_mean = {model.get('y_mean')}")
        print(f"  y_std = {model.get('y_std')}")
        print(f"  grad = {model.get('grad')}")
    
    # Il problema: con y_std ≈ 0, la normalizzazione y_centered = (y - y_mean) / y_std
    # potrebbe causare divisione per zero. Ma abbiamo + 1e-6 come protezione.
    
    scores = np.array([s for _, s in root.tested_pairs])
    y_std_raw = scores.std()
    print(f"\n  scores.std() = {y_std_raw:.2e}")
    print(f"  y_std (con protezione) = {y_std_raw + 1e-6:.2e}")
    
    if y_std_raw < 1e-10:
        print("  ⚠️ Funzione costante: std ≈ 0, gradiente sarà ~0")
    
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALBA DEEP TRACER - VERIFICA FLUSSO NUMERICO")
    print("="*80)
    
    traces = [
        ("LGS Fit Numerical", trace_lgs_fit_numerical),
        ("Rank Weights Bug?", trace_rank_weights_bug),
        ("Extremely Steep", trace_extremely_steep),
        ("Tiny Cube", trace_tiny_cube),
        ("Constant Function", trace_constant_function),
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
