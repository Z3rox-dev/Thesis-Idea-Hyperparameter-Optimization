#!/usr/bin/env python3
"""
XGBoost TabularBenchmark con logging dettagliato di HPO Debug (CurvNet/QuadHPO).
Logga:
- PCA: condition number, eigenvalues, reconstruction error
- Surrogate: R², coefficienti, predictions
- EI: candidati, valori, decisioni di sampling
"""
import sys
sys.path.insert(0, '/mnt/workspace/HPOBench')

import numpy as np
import warnings
warnings.filterwarnings('ignore')
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

import ConfigSpace as CS
from hpobench.benchmarks.ml import TabularBenchmark
from datetime import datetime
import math

# Log file
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"/mnt/workspace/thesis/tests/xgb_curvnet_debug_{TIMESTAMP}.log"

# Global log handle
log_handle = None

def log(msg: str):
    """Log to both console and file."""
    global log_handle
    print(msg)
    if log_handle:
        log_handle.write(msg + "\n")
        log_handle.flush()

# Contatori per statistiche aggregate
stats = {
    'pca_calls': 0,
    'pca_ok': 0,
    'pca_fail': 0,
    'pca_cond_numbers': [],
    'pca_recon_errors': [],
    'surrogate_fits': 0,
    'surrogate_good': 0,
    'surrogate_bad': 0,
    'surrogate_r2': [],
    'surrogate_sigma2': [],
    'ei_evals': 0,
    'ei_accepted': 0,
    'ei_rejected': 0,
    'ei_values': [],
    'sampling_modes': {},
}

def curvnet_debug_logger(msg: str):
    """Logger callback per QuadHPO - analizza e logga messaggi dettagliati."""
    log(f"  [CURVNET] {msg}")
    
    # Parse messages per statistiche
    if "PCA Instabile" in msg or "PCA Reconstruction" in msg:
        # Già loggato direttamente, ma potremmo contare
        pass
    if "surrogato ha predetto NaN" in msg or "Predizioni surrogato fuori scala" in msg:
        pass

# Open log file
log_handle = open(LOG_FILE, 'w')

# Config
MODEL = 'xgb'
TASK_ID = 31
BUDGET = 150
SEEDS = [42, 123, 456]

log("="*80)
log(f"XGBoost TabularBenchmark Debug Log - CurvNet/QuadHPO")
log(f"Model: {MODEL} | Task ID: {TASK_ID} | Budget: {BUDGET}")
log(f"Seeds: {SEEDS}")
log(f"Log file: {LOG_FILE}")
log("="*80)

# Setup benchmark
bench = TabularBenchmark(model=MODEL, task_id=TASK_ID)
cs = bench.get_configuration_space()
hps = cs.get_hyperparameters()
max_fid = bench.get_max_fidelity()

log(f"\nHyperparameters ({len(hps)}):")
for hp in hps:
    log(f"  - {hp.name}")

# Build bounds e types
bounds = []
types = []
for hp in hps:
    if hasattr(hp, 'sequence') and hp.sequence:
        seq = list(hp.sequence)
        bounds.append((0.0, float(len(seq) - 1)))
        types.append('index')
    elif isinstance(hp, CS.UniformFloatHyperparameter):
        bounds.append((float(hp.lower), float(hp.upper)))
        types.append('float')
    elif isinstance(hp, CS.UniformIntegerHyperparameter):
        bounds.append((float(hp.lower), float(hp.upper)))
        types.append('int')
    else:
        raise ValueError(f'Unsupported HP: {hp}')

dim = len(bounds)
log(f"\nDimension: {dim}")
log(f"Bounds: {bounds}")
log(f"Types: {types}")

def xnorm_to_config(x_norm):
    values = {}
    for val, hp, (lo, hi), t in zip(x_norm, hps, bounds, types):
        if t == 'index':
            seq = list(hp.sequence)
            idx = int(np.clip(np.floor(val * len(seq)), 0, len(seq) - 1))
            values[hp.name] = seq[idx]
        else:
            v = lo + float(val) * (hi - lo)
            if t == 'int':
                v = int(round(v))
                v = max(int(hp.lower), min(int(hp.upper), int(v)))
            values[hp.name] = v
    return CS.Configuration(cs, values=values)

# Import optimizer
from hpo_debug import QuadHPO

# Funzione per ispezionare i cubi durante l'ottimizzazione
def inspect_cube_state(hpo, cube, trial_id):
    """Ispeziona stato del cubo: PCA, surrogate, EI candidates."""
    
    lines = []
    lines.append(f"\n  === CUBE INSPECTION @ Trial {trial_id} ===")
    lines.append(f"  Cube ID: {id(cube) % 10000} | Depth: {cube.depth} | n_trials: {cube.n_trials}")
    
    # PCA State
    d = len(cube.bounds)
    R, mu, eigvals, pca_ok = cube._principal_axes()
    cond_number = eigvals.max() / (eigvals.min() + 1e-12) if eigvals.min() > 1e-12 else float('inf')
    
    lines.append(f"  [PCA] OK: {pca_ok} | Cond: {cond_number:.2e}")
    lines.append(f"       Eigvals: {eigvals}")
    
    stats['pca_calls'] += 1
    stats['pca_cond_numbers'].append(cond_number)
    if pca_ok:
        stats['pca_ok'] += 1
    else:
        stats['pca_fail'] += 1
    
    # Reconstruction error
    pairs = getattr(cube, "_tested_pairs", [])
    if len(pairs) >= 3:
        pts = np.array([p for (p, s) in pairs], dtype=float)
        k_pca = min(2, d)
        U_k = R[:, :k_pca]
        Z = (pts - mu) @ U_k
        pts_recon = Z @ U_k.T + mu
        recon_errors = np.linalg.norm(pts - pts_recon, axis=1)
        mean_recon = float(np.mean(recon_errors))
        max_recon = float(np.max(recon_errors))
        
        pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
        diag_len = float(np.linalg.norm(pts_range)) + 1e-12
        rel_error = mean_recon / diag_len
        
        lines.append(f"       Recon Error: mean={mean_recon:.4f} max={max_recon:.4f} rel={rel_error:.2%}")
        stats['pca_recon_errors'].append(rel_error)
    
    # Surrogate State
    s2d = cube.surrogate_2d
    if s2d is not None:
        r2 = s2d.get('r2', 0.0)
        sigma2 = s2d.get('sigma2', 1.0)
        n_surr = s2d.get('n', 0)
        stype = s2d.get('type', 'unknown')
        pca_ok_surr = s2d.get('pca_ok', False)
        w = s2d.get('w', [])
        
        lines.append(f"  [SURROGATE] Type: {stype} | n: {n_surr}")
        lines.append(f"       R²: {r2:.4f} | σ²: {sigma2:.4f} | PCA OK: {pca_ok_surr}")
        lines.append(f"       Coeffs (w): {w[:5]}..." if len(w) > 5 else f"       Coeffs (w): {w}")
        
        stats['surrogate_fits'] += 1
        stats['surrogate_r2'].append(r2)
        stats['surrogate_sigma2'].append(sigma2)
        if r2 >= 0.5:
            stats['surrogate_good'] += 1
        else:
            stats['surrogate_bad'] += 1
            
        # Test prediction at cube center
        center_prime = np.zeros(d)
        y_pred, std_pred = cube.predict_surrogate(center_prime)
        lines.append(f"       Pred@center: μ={y_pred:.4f} σ={std_pred:.4f}")
        
        # Check for anomalies
        if not np.isfinite(y_pred) or not np.isfinite(std_pred):
            lines.append(f"       !!! ANOMALY: NaN/Inf in predictions !!!")
        if abs(y_pred) > 100:
            lines.append(f"       !!! WARNING: Large prediction value !!!")
            
    else:
        lines.append(f"  [SURROGATE] Not fitted yet")
    
    # UCB Value
    beta = getattr(hpo.config, 'ucb_beta', 1.0)
    lambda_geo = 1.0 + 0.1 * dim
    ucb_val = cube.ucb(beta=beta, lambda_geo=lambda_geo)
    lines.append(f"  [UCB] Value: {ucb_val:.4f} | Beta: {beta:.3f} | Lambda: {lambda_geo:.3f}")
    
    # Log all
    for line in lines:
        log(line)

# Track all scores per seed
results = []

for seed in SEEDS:
    log(f"\n{'#'*80}")
    log(f"# SEED: {seed}")
    log(f"{'#'*80}")
    
    # Reset RNG
    np.random.seed(seed)
    
    # Reset stats per seed
    for k in stats:
        if isinstance(stats[k], list):
            stats[k] = []
        elif isinstance(stats[k], dict):
            stats[k] = {}
        elif isinstance(stats[k], int):
            stats[k] = 0
    
    # Objective counter
    eval_count = [0]
    all_scores = []
    
    def objective(x_norm):
        eval_count[0] += 1
        cfg = xnorm_to_config(x_norm)
        res = bench.objective_function(cfg, fidelity=max_fid, metric='acc')
        loss = float(res['function_value'])
        all_scores.append(loss)
        
        # Log ogni valutazione con dettagli
        if eval_count[0] <= 20 or eval_count[0] % 25 == 0:
            x_str = " ".join([f"{xi:.4f}" for xi in x_norm])
            log(f"    [EVAL {eval_count[0]:3d}] x=[{x_str}] loss={loss:.6f}")
        
        return loss
    
    # Create optimizer
    hpo = QuadHPO(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        rng_seed=seed,
        debug_log=True,
    )
    # Set budget for adaptive config
    hpo.budget = BUDGET
    hpo.config.adapt(dim, BUDGET)
    
    # Attach debug logger to all cubes
    def attach_logger_to_cubes(hpo):
        for cube in hpo.leaf_cubes:
            cube._debug_logger = curvnet_debug_logger
        if hasattr(hpo, 'root') and hpo.root is not None:
            hpo.root._debug_logger = curvnet_debug_logger
    
    log(f"\n--- Running optimization (budget={BUDGET}) ---")
    
    # Run optimization manually to inject inspections
    for trial in range(BUDGET):
        # Select cube
        cube = hpo.select_cube()
        
        # Attach logger
        attach_logger_to_cubes(hpo)
        
        # Inspect at key points: first 15, then every 25, plus last 5
        do_inspect = (trial < 15) or (trial % 25 == 0) or (trial >= BUDGET - 5)
        
        if do_inspect:
            inspect_cube_state(hpo, cube, trial)
        
        # Run trial (this samples and evaluates)
        hpo.run_trial(cube, lambda x: objective(x))
        
        # Log sampling mode if tracked
        # (QuadHPO doesn't track this externally, but we can infer from logs)
    
    best_x = hpo.best_x_candidate
    best_loss = -hpo.best_score_global if not hpo.maximize else hpo.best_score_global
    
    log(f"\n--- Optimization completed ---")
    log(f"Best loss: {best_loss:.6f}")
    log(f"Best x: {best_x}")
    log(f"Evaluations: {eval_count[0]}")
    log(f"Splits: {hpo.splits_count}")
    log(f"Leaves: {len(hpo.leaf_cubes)}")
    
    # Analisi scores
    all_scores_arr = np.array(all_scores)
    log(f"\n--- Score Analysis ---")
    log(f"Min: {all_scores_arr.min():.6f}")
    log(f"Max: {all_scores_arr.max():.6f}")
    log(f"Mean: {all_scores_arr.mean():.6f}")
    log(f"Std: {all_scores_arr.std():.6f}")
    log(f"Median: {np.median(all_scores_arr):.6f}")
    
    # Check for anomalies
    anomalies = []
    for i, s in enumerate(all_scores):
        if math.isnan(s) or math.isinf(s):
            anomalies.append((i, s, "NaN/Inf"))
        elif s < 0:
            anomalies.append((i, s, "Negative"))
        elif s > 1:
            anomalies.append((i, s, "Above 1"))
    
    if anomalies:
        log(f"\n!!! ANOMALIES DETECTED ({len(anomalies)}) !!!")
        for idx, val, reason in anomalies[:10]:
            log(f"  Eval {idx}: {val} ({reason})")
    else:
        log(f"\nNo anomalies detected in scores.")
    
    # PCA Stats
    log(f"\n--- PCA Stats ---")
    log(f"Calls: {stats['pca_calls']} | OK: {stats['pca_ok']} | Fail: {stats['pca_fail']}")
    if stats['pca_cond_numbers']:
        conds = np.array(stats['pca_cond_numbers'])
        conds_finite = conds[np.isfinite(conds)]
        if len(conds_finite) > 0:
            log(f"Cond Numbers: min={conds_finite.min():.2e} max={conds_finite.max():.2e} mean={conds_finite.mean():.2e}")
    if stats['pca_recon_errors']:
        recons = np.array(stats['pca_recon_errors'])
        log(f"Recon Errors: min={recons.min():.2%} max={recons.max():.2%} mean={recons.mean():.2%}")
    
    # Surrogate Stats
    log(f"\n--- Surrogate Stats ---")
    log(f"Fits: {stats['surrogate_fits']} | Good (R²≥0.5): {stats['surrogate_good']} | Bad: {stats['surrogate_bad']}")
    if stats['surrogate_r2']:
        r2s = np.array(stats['surrogate_r2'])
        log(f"R²: min={r2s.min():.4f} max={r2s.max():.4f} mean={r2s.mean():.4f}")
    if stats['surrogate_sigma2']:
        sig2s = np.array(stats['surrogate_sigma2'])
        log(f"σ²: min={sig2s.min():.4f} max={sig2s.max():.4f} mean={sig2s.mean():.4f}")
    
    # Convergence check
    log(f"\n--- Convergence Check ---")
    log(f"First 10 scores: {all_scores[:10]}")
    log(f"Last 10 scores: {all_scores[-10:]}")
    
    # Best at different points
    running_best = []
    best_so_far = float('inf')
    for s in all_scores:
        if s < best_so_far:
            best_so_far = s
        running_best.append(best_so_far)
    
    log(f"Running best @ 25%: {running_best[int(BUDGET*0.25)]:.6f}")
    log(f"Running best @ 50%: {running_best[int(BUDGET*0.50)]:.6f}")
    log(f"Running best @ 75%: {running_best[int(BUDGET*0.75)]:.6f}")
    log(f"Running best @ 100%: {running_best[-1]:.6f}")
    
    results.append({
        'seed': seed,
        'best_loss': best_loss,
        'all_scores': all_scores,
        'running_best': running_best
    })

# Final summary
log(f"\n{'='*80}")
log(f"FINAL SUMMARY")
log(f"{'='*80}")

for r in results:
    log(f"Seed {r['seed']}: best_loss = {r['best_loss']:.6f}")

mean_loss = np.mean([r['best_loss'] for r in results])
std_loss = np.std([r['best_loss'] for r in results])
log(f"\nMean best loss: {mean_loss:.6f} ± {std_loss:.6f}")

log(f"\nLog saved to: {LOG_FILE}")
log_handle.close()

print(f"\n\nDone! Log saved to: {LOG_FILE}")
