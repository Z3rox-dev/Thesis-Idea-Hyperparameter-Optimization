#!/usr/bin/env python3
"""
===============================================================================
REPORT COMPLETO: Analisi LGS Grid vs Experimental su Rosenbrock
===============================================================================

Questo script analizza in profondità PERCHÉ Experimental performa meglio di
Grid su Rosenbrock, nonostante entrambi usino LGS.

RISULTATI DIAGNOSTICI CHIAVE:
=============================

1. SATURAZIONE LGS - CONFERMATA IN ENTRAMBI
   -----------------------------------------
   Grid 8D:
     - noise_var: 100% al clip max (10.0)
     - sigma_scale: 100% al clip max (5.0)
   
   Exp 8D:
     - noise_var: 99.5% al clip max (10.0)
     - sigma_scale: NON PRESENTE
   
   → Entrambi hanno LGS che "modella male" Rosenbrock (funzione non-lineare)
   → Ma Grid amplifica l'errore con sigma_scale!

2. IMPATTO DI SIGMA_SCALE
   -----------------------
   Grid usa: total_sigma = base_sigma * sigma_scale
   
   Con sigma_scale = 5.0 (saturato):
   → UCB = mu + beta * 5.0 * base_sigma
   → L'incertezza è GONFIATA di 5x!
   → argmax seleziona punti con alta incertezza (spesso outlier)
   
   Experimental NON ha sigma_scale:
   → UCB = mu + beta * base_sigma
   → Incertezza non amplificata
   → Meno sensibile a outlier

3. SELEZIONE: ARGMAX vs SOFTMAX
   -----------------------------
   Grid:   idx = np.argmax(score)  [DETERMINISTICO]
   Exp:    idx = rng.choice(p=softmax(score * temp))  [STOCASTICO]
   
   Con LGS mal calibrato:
   - argmax → sceglie sempre l'outlier peggiore
   - softmax → ha probabilità di NON scegliere l'outlier
   
   Questo spiega perché Exp è più robusto a errori del modello.

4. NUMERO DI CANDIDATI
   --------------------
   Grid:  64 candidati/batch × 4 batches = 256 candidati
   Exp:   25 candidati totali
   
   Con più candidati:
   - Più chance di trovare un "falso positivo" con alta incertezza
   - argmax lo seleziona sempre
   
   Con meno candidati:
   - Meno falsi positivi nel pool
   - softmax riduce ulteriormente il rischio

5. STRATEGIE DI CAMPIONAMENTO
   ---------------------------
   Grid: Griglia quasi-uniforme su tutto il cubo
         → Molto dipendente dal ranking del modello
   
   Exp:  Mix di strategie:
         - 25%: perturbazione top-k (exploitation locale)
         - 15%: gradient direction (sfrutta info LGS)
         - 15%: gaussian centro (esplorazione locale)
         - 45%: uniform (esplorazione)
         
   → Exp dedica ~40% all'exploitation locale vicino ai buoni punti
   → Su Rosenbrock (valle stretta e curva) questo aiuta molto

CONCLUSIONE
===========
Experimental > Grid su Rosenbrock per combinazione di:

1. NIENTE sigma_scale → incertezza non amplificata
2. Softmax selection → robusto a outlier del modello
3. Meno candidati → meno falsi positivi
4. Exploitation locale → 40% sampling vicino ai top-k

Il problema di Grid NON è LGS in sé, ma come lo USA:
- sigma_scale amplifica errori
- argmax su tanti candidati seleziona sempre gli errori

POSSIBILI FIX PER GRID
======================
1. Ridurre sigma_scale max da 5.0 a 1.5
2. Softmax selection invece di argmax
3. Ridurre grid_batch_size o grid_batches
4. Aggiungere sampling top-k local come Exp
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/thesis')

# Storage per diagnostica
DIAG_GRID = {"noise_vars": [], "sigma_scales": [], "mu_vals": [], "sigma_vals": []}
DIAG_EXP = {"noise_vars": [], "mu_vals": [], "sigma_vals": []}

# --- Rosenbrock ---
def rosenbrock(x):
    x_scaled = (np.array(x) - 0.5) * 4.096
    total = 0
    for i in range(len(x_scaled) - 1):
        total += 100 * (x_scaled[i+1] - x_scaled[i]**2)**2 + (1 - x_scaled[i])**2
    return total

# --- Patch Grid LGS ---
from alba_framework_grid import lgs as grid_lgs

original_grid_fit = grid_lgs.fit_lgs_model
def patched_grid_fit(cube, gamma, dim, rng=None):
    result = original_grid_fit(cube, gamma, dim, rng)
    if result is not None:
        DIAG_GRID["noise_vars"].append(result.get("noise_var", None))
        DIAG_GRID["sigma_scales"].append(result.get("sigma_scale", None))
    return result
grid_lgs.fit_lgs_model = patched_grid_fit

from alba_framework_grid import cube as grid_cube
grid_cube._fit_lgs_model = patched_grid_fit

from alba_framework_grid.optimizer import ALBA as AlbaGrid

# --- Patch Experimental ---
from ALBA_V1_experimental import ALBA as AlbaExp, Cube as ExpCube

original_exp_fit = ExpCube.fit_lgs_model
def patched_exp_fit(self, gamma, dim, rng=None):
    original_exp_fit(self, gamma, dim, rng)
    if self.lgs_model is not None:
        DIAG_EXP["noise_vars"].append(self.lgs_model.get("noise_var", None))
ExpCube.fit_lgs_model = patched_exp_fit


def run_comparison(dim, budget=250, n_seeds=5):
    """Run comparison on multiple seeds."""
    results = {"grid": [], "exp": []}
    
    for seed in range(n_seeds):
        # Grid
        opt = AlbaGrid(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed, total_budget=budget)
        for _ in range(budget):
            x = opt.ask()
            y = rosenbrock(x if not isinstance(x, dict) else list(x.values()))
            opt.tell(x, y)
        results["grid"].append(opt.best_y if hasattr(opt, 'best_y') else -opt.best_y_internal)
        
        # Exp
        opt = AlbaExp(bounds=[(0.0, 1.0)] * dim, maximize=False, seed=seed, total_budget=budget)
        for _ in range(budget):
            x = opt.ask()
            y = rosenbrock(x if not isinstance(x, dict) else list(x.values()))
            opt.tell(x, y)
        results["exp"].append(-opt.best_y_internal)
    
    return results


def print_summary():
    print("="*70)
    print("ANALISI LGS GRID vs EXPERIMENTAL - ROSENBROCK")
    print("="*70)
    
    print("\n" + "-"*70)
    print("1. CONFRONTO PERFORMANCE (5 seeds, budget=250)")
    print("-"*70)
    
    for dim in [8, 15]:
        # Reset diagnostics
        DIAG_GRID["noise_vars"] = []
        DIAG_GRID["sigma_scales"] = []
        DIAG_EXP["noise_vars"] = []
        
        results = run_comparison(dim, budget=250, n_seeds=5)
        
        print(f"\nRosenbrock {dim}D:")
        print(f"  Grid:  mean={np.mean(results['grid']):.2f}, std={np.std(results['grid']):.2f}")
        print(f"  Exp:   mean={np.mean(results['exp']):.2f}, std={np.std(results['exp']):.2f}")
        
        # Diagnostics
        grid_nv = [v for v in DIAG_GRID["noise_vars"] if v is not None]
        grid_ss = [v for v in DIAG_GRID["sigma_scales"] if v is not None]
        exp_nv = [v for v in DIAG_EXP["noise_vars"] if v is not None]
        
        print(f"\n  LGS Diagnostics {dim}D:")
        if grid_nv:
            print(f"    Grid noise_var:  min={min(grid_nv):.2f}, max={max(grid_nv):.2f}, "
                  f"at_clip_max={100*sum(v>=9.99 for v in grid_nv)/len(grid_nv):.0f}%")
        if grid_ss:
            print(f"    Grid sigma_scale: min={min(grid_ss):.2f}, max={max(grid_ss):.2f}, "
                  f"at_clip_max={100*sum(v>=4.99 for v in grid_ss)/len(grid_ss):.0f}%")
        if exp_nv:
            print(f"    Exp  noise_var:  min={min(exp_nv):.2f}, max={max(exp_nv):.2f}, "
                  f"at_clip_max={100*sum(v>=9.99 for v in exp_nv)/len(exp_nv):.0f}%")
    
    print("\n" + "-"*70)
    print("2. DIFFERENZE ARCHITETTURALI")
    print("-"*70)
    
    print("""
    ┌─────────────────────┬───────────────────────┬───────────────────────┐
    │      Feature        │        GRID           │     EXPERIMENTAL      │
    ├─────────────────────┼───────────────────────┼───────────────────────┤
    │ Candidati/iter      │  ~256 (64×4 batches)  │  25                   │
    │ Selezione           │  argmax (determin.)   │  softmax (stochastic) │
    │ sigma_scale         │  EMA, clip [0.2, 5.0] │  NON PRESENTE         │
    │ Sampling strategy   │  griglia uniforme     │  mix (40% top-k local)│
    │ noise_var clip      │  [1e-4, 10.0]         │  [1e-4, 10.0]         │
    └─────────────────────┴───────────────────────┴───────────────────────┘
    """)
    
    print("-"*70)
    print("3. PERCHÉ EXPERIMENTAL VINCE SU ROSENBROCK")
    print("-"*70)
    print("""
    A) sigma_scale AMPLIFICA L'ERRORE IN GRID:
       - LGS non riesce a modellare la valle curva di Rosenbrock
       - residui² alti → sigma_scale satura a 5.0
       - UCB = mu + beta × 5.0 × base_sigma
       - Incertezza gonfiata 5× → argmax sceglie outlier
    
    B) SOFTMAX SELECTION È ROBUSTO AGLI OUTLIER:
       - Anche se LGS sbaglia, softmax non sceglie sempre il max
       - Temperature=3.0 in Exp → distribuzione non troppo peaked
       - Ha probabilità di evitare i "falsi buoni"
    
    C) MENO CANDIDATI = MENO FALSI POSITIVI:
       - Grid valuta 256 candidati → più chance di outlier
       - Exp valuta 25 candidati → pool più piccolo, meno rischi
       - argmax(256) molto più vulnerabile di softmax(25)
    
    D) EXPLOITATION LOCALE AIUTA SU ROSENBROCK:
       - Rosenbrock ha valle stretta e curva
       - Exp: 40% candidati vicino ai top-k (perturbation + gradient)
       - Questo "segue la valle" meglio della griglia uniforme
    """)
    
    print("-"*70)
    print("4. POSSIBILI FIX PER GRID")
    print("-"*70)
    print("""
    1. RIDURRE sigma_scale_max: da 5.0 → 1.5 o 2.0
       → Limita amplificazione errore UCB
    
    2. SOFTMAX SELECTION: invece di argmax puro
       → probs = softmax(score * temperature)
       → idx = rng.choice(p=probs)
    
    3. RIDURRE CANDIDATI: grid_batch_size=32, grid_batches=2
       → ~64 candidati invece di 256
    
    4. AGGIUNGERE TOP-K LOCAL: 20-30% candidati vicino ai migliori
       → Come fa Experimental
    
    5. SURROGATO ALTERNATIVO: knn_lgs invece di lgs puro
       → Codex ha mostrato che knn_lgs su Rosenbrock 8D → 44.89 (meglio di tutto)
    """)


if __name__ == "__main__":
    print_summary()
