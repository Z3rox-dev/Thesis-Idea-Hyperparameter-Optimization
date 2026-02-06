#!/usr/bin/env python3
"""
Test dell'intuizione V2: PF aiuta su funzioni con RUMORE (RF-style)?

La RF di ParamNet non produce solo gradini, ma anche:
1. Rumore nella stima (varianza tra alberi)
2. Superfici localmente "piatte" dove il gradiente svanisce

Testiamo: funzione smooth vs funzione con rumore additivo.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# FUNZIONI TEST
# ============================================================================

def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)

def rosenbrock(x: np.ndarray) -> float:
    total = 0.0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total

def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def levy(x: np.ndarray) -> float:
    """Levy function - smooth with global min at (1,1,...,1)."""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return term1 + term2 + term3


class NoisyFunction:
    """
    Wrapper che aggiunge rumore gaussiano correlato (simula RF).
    Il rumore è correlato spazialmente: punti vicini hanno rumore simile.
    """
    def __init__(self, func: Callable, noise_std: float = 0.1, 
                 correlation_scale: float = 0.5, seed: int = 42):
        self.func = func
        self.noise_std = noise_std
        self.correlation_scale = correlation_scale
        self.rng = np.random.RandomState(seed)
        self.cache = {}  # Cache per consistenza
        self.name = f"{func.__name__}_noisy"
    
    def __call__(self, x: np.ndarray) -> float:
        y_clean = self.func(x)
        
        # Rumore correlato: hash della posizione quantizzata
        x_quantized = tuple(np.round(x / self.correlation_scale).astype(int))
        
        if x_quantized not in self.cache:
            self.cache[x_quantized] = self.rng.randn()
        
        noise = self.cache[x_quantized] * self.noise_std * (1 + abs(y_clean) * 0.01)
        
        return y_clean + noise


class StepFunction:
    """
    Wrapper che crea superfici a gradini (plateau locali).
    Simula l'effetto della media di alberi in una RF.
    """
    def __init__(self, func: Callable, step_size: float = 0.3, seed: int = 42):
        self.func = func
        self.step_size = step_size
        self.rng = np.random.RandomState(seed)
        self.name = f"{func.__name__}_steps"
    
    def __call__(self, x: np.ndarray) -> float:
        # Quantizza lo spazio input in celle
        x_cell = np.floor(x / self.step_size)
        
        # Calcola valore medio nella cella (simula RF averaging)
        cell_center = (x_cell + 0.5) * self.step_size
        y_center = self.func(cell_center)
        
        # Aggiungi piccolo rumore per celle diverse
        cell_hash = hash(tuple(x_cell.astype(int))) % 10000
        self.rng.seed(cell_hash)
        noise = self.rng.randn() * 0.05 * (1 + abs(y_center) * 0.01)
        
        return y_center + noise


# ============================================================================
# BENCHMARK
# ============================================================================

def run_alba(
    func: Callable,
    dim: int,
    n_trials: int,
    seed: int,
    use_potential_field: bool,
) -> float:
    """Run ALBA, return best value found."""
    
    bounds = [(-5.0, 5.0)] * dim
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        maximize=False,
        total_budget=n_trials,
        use_potential_field=use_potential_field,
        use_coherence_gating=True,
    )
    
    best_y = np.inf
    for _ in range(n_trials):
        x = opt.ask()
        if isinstance(x, dict):
            x_arr = np.array(list(x.values()))
        else:
            x_arr = np.array(x)
        y = func(x_arr)
        opt.tell(x, y)
        if y < best_y:
            best_y = y
    
    return best_y


def compare_variants(
    base_func: Callable,
    dim: int,
    budget: int,
    n_seeds: int,
    noise_std: float = 0.2,
    step_size: float = 0.3,
) -> dict:
    """Compare smooth, noisy, and step versions."""
    
    print(f"\n{'='*70}")
    print(f"  {base_func.__name__.upper()} (dim={dim})")
    print(f"{'='*70}")
    
    variants = {
        'smooth': base_func,
        'noisy': NoisyFunction(base_func, noise_std=noise_std, seed=12345),
        'steps': StepFunction(base_func, step_size=step_size, seed=12345),
    }
    
    results = {}
    
    for var_name, func in variants.items():
        print(f"\n  {var_name.upper()} version:")
        
        cov_vals = []
        pf_vals = []
        wins_cov = 0
        wins_pf = 0
        
        for seed in range(n_seeds):
            # Importante: seed diversi per COV e PF per avere percorsi diversi
            val_cov = run_alba(func, dim, budget, 1000 + seed, use_potential_field=False)
            val_pf = run_alba(func, dim, budget, 2000 + seed, use_potential_field=True)
            
            cov_vals.append(val_cov)
            pf_vals.append(val_pf)
            
            if val_cov < val_pf:
                wins_cov += 1
                w = "COV"
            elif val_pf < val_cov:
                wins_pf += 1
                w = "PF"
            else:
                w = "TIE"
            
            print(f"    Seed {seed}: COV={val_cov:.4f} PF={val_pf:.4f} → {w}")
        
        mean_cov = np.mean(cov_vals)
        mean_pf = np.mean(pf_vals)
        delta_pct = (mean_cov - mean_pf) / abs(mean_cov) * 100 if mean_cov != 0 else 0
        
        results[var_name] = {
            'mean_cov': mean_cov,
            'mean_pf': mean_pf,
            'wins_cov': wins_cov,
            'wins_pf': wins_pf,
            'delta_pct': delta_pct,
        }
        
        print(f"    → COV: {mean_cov:.4f}, PF: {mean_pf:.4f}, Wins: COV={wins_cov}, PF={wins_pf}")
    
    # Calcola boost PF su noisy/steps vs smooth
    pf_adv_smooth = results['smooth']['wins_pf'] - results['smooth']['wins_cov']
    pf_adv_noisy = results['noisy']['wins_pf'] - results['noisy']['wins_cov']
    pf_adv_steps = results['steps']['wins_pf'] - results['steps']['wins_cov']
    
    results['pf_boost_noisy'] = pf_adv_noisy - pf_adv_smooth
    results['pf_boost_steps'] = pf_adv_steps - pf_adv_smooth
    
    print(f"\n  PF boost su NOISY vs SMOOTH: {results['pf_boost_noisy']:+d}")
    print(f"  PF boost su STEPS vs SMOOTH: {results['pf_boost_steps']:+d}")
    
    return results


def main():
    print("=" * 75)
    print("  TEST V2: Potential Field su funzioni NOISY e STEPS (RF-style)")
    print("  Ipotesi: PF stabilizza su superfici rumorose/a gradini")
    print("=" * 75)
    
    DIM = 8
    BUDGET = 200
    N_SEEDS = 15
    NOISE_STD = 0.3
    STEP_SIZE = 0.4
    
    test_funcs = [sphere, rosenbrock, rastrigin, levy]
    
    all_results = {}
    
    for func in test_funcs:
        results = compare_variants(
            base_func=func,
            dim=DIM,
            budget=BUDGET,
            n_seeds=N_SEEDS,
            noise_std=NOISE_STD,
            step_size=STEP_SIZE,
        )
        all_results[func.__name__] = results
    
    # ========================================================================
    # FINAL ANALYSIS
    # ========================================================================
    print("\n" + "=" * 75)
    print("  ANALISI FINALE")
    print("=" * 75)
    
    print(f"\n{'Function':<12} {'SMOOTH':<18} {'NOISY':<18} {'STEPS':<18} {'Boost N':<8} {'Boost S':<8}")
    print("-" * 85)
    
    total_boost_noisy = 0
    total_boost_steps = 0
    
    for func_name, r in all_results.items():
        s = r['smooth']
        n = r['noisy']
        t = r['steps']
        
        bn = r['pf_boost_noisy']
        bs = r['pf_boost_steps']
        
        sn = "✅" if bn > 0 else ("❌" if bn < 0 else "➖")
        ss = "✅" if bs > 0 else ("❌" if bs < 0 else "➖")
        
        print(f"{func_name:<12} COV:{s['wins_cov']:>2} PF:{s['wins_pf']:>2}     "
              f"COV:{n['wins_cov']:>2} PF:{n['wins_pf']:>2}     "
              f"COV:{t['wins_cov']:>2} PF:{t['wins_pf']:>2}     "
              f"{bn:>+3} {sn}   {bs:>+3} {ss}")
        
        total_boost_noisy += bn
        total_boost_steps += bs
    
    print("-" * 85)
    print(f"{'TOTALE':<12} {'':<18} {'':<18} {'':<18} {total_boost_noisy:>+3}      {total_boost_steps:>+3}")
    
    print("\n" + "=" * 75)
    
    # Verdetto
    if total_boost_noisy > 0 or total_boost_steps > 0:
        if total_boost_noisy > total_boost_steps:
            print(f"🔥 IPOTESI PARZIALMENTE CONFERMATA:")
            print(f"   PF aiuta di più su funzioni NOISY (+{total_boost_noisy} boost)")
        else:
            print(f"🔥 IPOTESI PARZIALMENTE CONFERMATA:")
            print(f"   PF aiuta di più su funzioni a GRADINI (+{total_boost_steps} boost)")
    elif total_boost_noisy < 0 and total_boost_steps < 0:
        print(f"❌ IPOTESI RIFIUTATA:")
        print(f"   PF NON aiuta su funzioni rumorose/a gradini")
        print(f"   (boost noisy: {total_boost_noisy}, boost steps: {total_boost_steps})")
    else:
        print(f"➖ RISULTATI MISTI: nessuna conclusione chiara")
    
    print("=" * 75)
    
    # Extra: confronto con diversi livelli di rumore
    print("\n\n" + "=" * 75)
    print("  EXTRA: Effetto del livello di rumore su Rosenbrock")
    print("=" * 75)
    
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
    
    print(f"\n  {'Noise':<8} {'COV wins':<12} {'PF wins':<12} {'PF rate':<12}")
    print("  " + "-" * 50)
    
    for noise in noise_levels:
        if noise == 0:
            func = rosenbrock
        else:
            func = NoisyFunction(rosenbrock, noise_std=noise, seed=99999)
        
        wins_cov = 0
        wins_pf = 0
        
        for seed in range(N_SEEDS):
            val_cov = run_alba(func, DIM, BUDGET, 1000 + seed, use_potential_field=False)
            val_pf = run_alba(func, DIM, BUDGET, 2000 + seed, use_potential_field=True)
            
            if val_cov < val_pf:
                wins_cov += 1
            elif val_pf < val_cov:
                wins_pf += 1
        
        pf_rate = wins_pf / N_SEEDS * 100
        status = "🔥" if wins_pf > wins_cov else ("❌" if wins_cov > wins_pf else "➖")
        print(f"  {noise:<8.1f} {wins_cov:<12} {wins_pf:<12} {pf_rate:>5.0f}% {status}")
    
    print("\n  Se l'ipotesi è corretta: più rumore → più PF wins")


if __name__ == "__main__":
    main()
