#!/usr/bin/env python3
"""
Diagnostic: analisi approfondita LGS su Rosenbrock
Confronto Grid vs Experimental per capire perché Exp performa meglio.

Ipotesi da verificare:
1. Noise_var e sigma_scale saturano allo stesso modo in entrambi?
2. Differenze nel sampling dei candidati (n_candidates, strategie)
3. Differenze nella selezione finale (argmax vs softmax)
4. Differenze nel numero di candidati valutati per iterazione
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/thesis')

# Patch per tracciare diagnostica
DIAG_GRID = {"noise_vars": [], "sigma_scales": [], "n_candidates_per_iter": [], "selection_method": "argmax"}
DIAG_EXP = {"noise_vars": [], "sigma_scales": [], "n_candidates_per_iter": [], "selection_method": "softmax"}

# --- Rosenbrock ---
def rosenbrock(x):
    x_scaled = (np.array(x) - 0.5) * 4.096
    total = 0
    for i in range(len(x_scaled) - 1):
        total += 100 * (x_scaled[i+1] - x_scaled[i]**2)**2 + (1 - x_scaled[i])**2
    return total

# --- Import e patch Grid ---
# Patch lgs PRIMA di importare optimizer che importa cube che importa lgs
from alba_framework_grid import lgs as grid_lgs

original_grid_fit = grid_lgs.fit_lgs_model

def patched_grid_fit(cube, gamma, dim, rng=None):
    result = original_grid_fit(cube, gamma, dim, rng)
    if result is not None:
        DIAG_GRID["noise_vars"].append(result.get("noise_var", None))
        DIAG_GRID["sigma_scales"].append(result.get("sigma_scale", None))
    return result

grid_lgs.fit_lgs_model = patched_grid_fit

# Ora patch anche nel modulo cube che ha già importato
from alba_framework_grid import cube as grid_cube
grid_cube._fit_lgs_model = patched_grid_fit

# Ora importiamo ALBA
from alba_framework_grid.optimizer import ALBA as AlbaGrid

# --- Import Experimental ---
from ALBA_V1_experimental import ALBA as AlbaExp, Cube as ExpCube

# Patch Experimental Cube.fit_lgs_model
original_exp_fit = ExpCube.fit_lgs_model

def patched_exp_fit(self, gamma, dim, rng=None):
    original_exp_fit(self, gamma, dim, rng)
    if self.lgs_model is not None:
        DIAG_EXP["noise_vars"].append(self.lgs_model.get("noise_var", None))
        # Exp non ha sigma_scale, registra None
        DIAG_EXP["sigma_scales"].append(None)

ExpCube.fit_lgs_model = patched_exp_fit


def run_with_diagnostics(cls, func, dim, budget, seed, diag_dict, name):
    """Run optimizer and collect diagnostics."""
    opt = cls(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        total_budget=budget,
    )
    
    best_val = float('inf')
    history = []
    
    for i in range(budget):
        config = opt.ask()
        if isinstance(config, dict):
            vals = list(config.values())
        else:
            vals = config
        
        score = func(vals)
        opt.tell(config, score)
        best_val = min(best_val, score)
        history.append(best_val)
    
    return best_val, history


def analyze_diag(diag, name):
    """Analyze diagnostic data."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICA: {name}")
    print(f"{'='*60}")
    
    noise_vars = [v for v in diag["noise_vars"] if v is not None]
    sigma_scales = [v for v in diag["sigma_scales"] if v is not None]
    
    if noise_vars:
        nv = np.array(noise_vars)
        print(f"\nnoise_var statistics:")
        print(f"  min: {nv.min():.4f}, max: {nv.max():.4f}, mean: {nv.mean():.4f}")
        print(f"  % at clip max (10.0): {100 * np.mean(nv >= 9.99):.1f}%")
        print(f"  % at clip min (1e-4): {100 * np.mean(nv <= 1e-3):.1f}%")
        
        # Histogram-like distribution
        bins = [0, 0.1, 1.0, 5.0, 9.0, 10.0]
        for i in range(len(bins)-1):
            pct = 100 * np.mean((nv >= bins[i]) & (nv < bins[i+1]))
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {pct:.1f}%")
    else:
        print("  Nessun dato noise_var")
    
    if sigma_scales:
        ss = np.array(sigma_scales)
        print(f"\nsigma_scale statistics:")
        print(f"  min: {ss.min():.4f}, max: {ss.max():.4f}, mean: {ss.mean():.4f}")
        print(f"  % at clip max (5.0): {100 * np.mean(ss >= 4.99):.1f}%")
        print(f"  % at clip min (0.2): {100 * np.mean(ss <= 0.21):.1f}%")
    else:
        print("\n  sigma_scale: non presente (Experimental non lo usa)")
    
    print(f"\nSelezione candidati: {diag['selection_method']}")


def compare_candidate_generation():
    """Analizza differenze nella generazione candidati."""
    print("\n" + "="*70)
    print("ANALISI ARCHITETTURALE: Differenze chiave Grid vs Experimental")
    print("="*70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                        GRID VERSION                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ Sampling:                                                             ║
║   - sample_candidates() → griglia quasi-uniforme su tutto lo spazio  ║
║   - grid_batch_size default: 64 candidati per batch                  ║
║   - grid_batches default: 4 → fino a 256 candidati totali!           ║
║                                                                       ║
║ Selezione:                                                            ║
║   - ARGMAX puro: idx = np.argmax(score)                              ║
║   - score = mu + beta*sigma - penalty                                 ║
║   - Con LGS rumoroso → sceglie spesso outlier "falsi buoni"          ║
║                                                                       ║
║ sigma_scale:                                                          ║
║   - EMA calibration che può saturare a 5.0                           ║
║   - Amplifica l'errore UCB quando LGS è mal calibrato                ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                     EXPERIMENTAL VERSION                              ║
╠══════════════════════════════════════════════════════════════════════╣
║ Sampling:                                                             ║
║   - _generate_candidates() con 4 strategie miste:                    ║
║     * 25%: perturbazione top-k points                                ║
║     * 15%: gradient direction                                         ║
║     * 15%: gaussian attorno al centro                                 ║
║     * 45%: uniform random nel cubo                                    ║
║   - n_candidates default: 25 (molto meno di Grid!)                   ║
║                                                                       ║
║ Selezione:                                                            ║
║   - SOFTMAX probabilistico:                                           ║
║       score_z = (score - mean) / std                                  ║
║       probs = softmax(score_z * temperature)                          ║
║       idx = rng.choice(p=probs)                                       ║
║   - NON sceglie sempre il massimo → evita outlier                    ║
║                                                                       ║
║ sigma_scale:                                                          ║
║   - NON presente! Usa direttamente noise_var                         ║
║   - Meno amplificazione dell'errore                                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def main():
    print("="*70)
    print("DIAGNOSTICA LGS: Grid vs Experimental su Rosenbrock")
    print("="*70)
    
    BUDGET = 250
    SEED = 0
    
    for dim in [8, 15]:
        print(f"\n{'#'*70}")
        print(f"# ROSENBROCK {dim}D")
        print(f"{'#'*70}")
        
        # Reset diagnostics
        DIAG_GRID["noise_vars"] = []
        DIAG_GRID["sigma_scales"] = []
        DIAG_EXP["noise_vars"] = []
        DIAG_EXP["sigma_scales"] = []
        
        # Run Grid
        print(f"\nRunning Grid on Rosenbrock {dim}D...")
        best_grid, hist_grid = run_with_diagnostics(
            AlbaGrid, rosenbrock, dim, BUDGET, SEED, DIAG_GRID, "Grid"
        )
        print(f"  Best: {best_grid:.4f}")
        
        # Run Exp
        print(f"\nRunning Experimental on Rosenbrock {dim}D...")
        best_exp, hist_exp = run_with_diagnostics(
            AlbaExp, rosenbrock, dim, BUDGET, SEED, DIAG_EXP, "Experimental"
        )
        print(f"  Best: {best_exp:.4f}")
        
        # Analyze
        analyze_diag(DIAG_GRID, f"Grid {dim}D")
        analyze_diag(DIAG_EXP, f"Experimental {dim}D")
        
        # Compare convergence
        print(f"\nConvergenza:")
        checkpoints = [50, 100, 150, 200, 250]
        for cp in checkpoints:
            if cp <= len(hist_grid):
                print(f"  iter {cp}: Grid={hist_grid[cp-1]:.2f}, Exp={hist_exp[cp-1]:.2f}, "
                      f"diff={hist_grid[cp-1] - hist_exp[cp-1]:+.2f}")
    
    compare_candidate_generation()
    
    print("\n" + "="*70)
    print("CONCLUSIONI")
    print("="*70)
    print("""
Le differenze chiave che spiegano perché Experimental > Grid su Rosenbrock:

1. NUMERO CANDIDATI:
   - Grid: ~256 candidati (64 * 4 batches)
   - Exp: 25 candidati
   → Con più candidati, argmax ha più possibilità di scegliere outlier

2. SELEZIONE:
   - Grid: ARGMAX deterministico
   - Exp: SOFTMAX probabilistico (temperature=3.0)
   → Softmax non sceglie sempre il "falso migliore" del modello rumoroso

3. SIGMA_SCALE:
   - Grid: EMA calibration che può saturare a 5.0 (amplifica errore UCB)
   - Exp: Non usa sigma_scale, sigma direttamente da noise_var
   → Meno amplificazione dell'incertezza mal calibrata

4. STRATEGIE DI SAMPLING:
   - Grid: Griglia quasi-uniforme (molto dipendente dal modello per ranking)
   - Exp: Mix di strategie (top-k perturbation, gradient, uniform)
   → Exp esplora più localmente attorno ai buoni punti trovati

5. FOCUS LOCALE:
   - Grid: Campiona su tutta la griglia del cubo
   - Exp: 40% dei candidati vicino ai top-k o lungo il gradiente
   → Su Rosenbrock (valle curva) essere vicini ai buoni aiuta molto
""")


if __name__ == "__main__":
    main()
