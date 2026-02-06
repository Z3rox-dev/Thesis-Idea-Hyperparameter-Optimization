#!/usr/bin/env python3
"""
Deep Analysis: Perché Cov fallisce ancora su alcuni seed dopo il fix?

Seed problematici identificati:
- Seed 3: Gauss=8.00, Cov=73.72 (9x peggio)

Ipotesi da verificare:
1. Il check del condizionamento non scatta quando dovrebbe
2. best_x è in una zona sbagliata
3. La covarianza ha una forma che "spinge" lontano dall'ottimo
4. Il fallback a Gaussian scatta troppo tardi
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

np.set_printoptions(precision=4, suppress=True)

def rosenbrock(x):
    x = np.array(list(x.values())) if isinstance(x, dict) else np.array(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))


class TracingCovarianceSampler(CovarianceLocalSearchSampler):
    """Wrapper che traccia ogni decisione del sampler."""
    
    def __init__(self):
        super().__init__()
        self.trace = []
        self.call_count = 0
    
    def sample(self, best_x, bounds, global_widths, progress, rng, X_history=None, y_history=None):
        self.call_count += 1
        
        dim = len(bounds) if bounds else 3
        
        # Traccia lo stato PRIMA di chiamare il sampler
        trace_entry = {
            'call': self.call_count,
            'best_x': best_x.copy() if best_x is not None else None,
            'progress': progress,
            'n_history': len(X_history) if X_history else 0,
        }
        
        # Analizza cosa farebbe il sampler
        if X_history is not None and y_history is not None:
            n = len(X_history)
            min_needed = max(self.min_points_fit, dim + 2)
            
            if n >= min_needed:
                k = max(min_needed, int(n * self.top_k_fraction))
                indices = np.argsort(y_history)
                top_indices = indices[-k:][::-1]
                
                top_X = np.array([X_history[i] for i in top_indices])
                top_y = np.array([y_history[i] for i in top_indices])
                
                weights = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                weights = weights / np.sum(weights)
                
                mu_w = np.average(top_X, axis=0, weights=weights)
                
                centered = top_X - mu_w
                C = np.dot((centered.T * weights), centered)
                C += 1e-6 * np.eye(dim)
                
                eigvals = np.linalg.eigvalsh(C)
                condition = eigvals.max() / max(eigvals.min(), 1e-10)
                
                trace_entry['k'] = k
                trace_entry['mu_w'] = mu_w.copy()
                trace_entry['top_y_costs'] = [-y for y in top_y[:5]]
                trace_entry['condition'] = condition
                trace_entry['eigvals'] = eigvals.copy()
                trace_entry['will_use_cov'] = condition <= 1000
                
                # Distanze dall'ottimo [1,1,1]
                opt = np.array([1, 1, 1])
                trace_entry['dist_mu_w_to_opt'] = np.linalg.norm(mu_w - opt)
                trace_entry['dist_best_x_to_opt'] = np.linalg.norm(best_x - opt) if best_x is not None else None
        
        self.trace.append(trace_entry)
        
        # Chiama il sampler originale
        return super().sample(best_x, bounds, global_widths, progress, rng, X_history, y_history)


def deep_analysis_seed(seed, verbose=True):
    """Analisi profonda di un singolo seed."""
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS SEED {seed}")
    print(f"{'='*70}")
    
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    # --- RUN CON TRACING ---
    tracer = TracingCovarianceSampler()
    opt_cov = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=tracer,
        use_drilling=False,
        seed=seed
    )
    
    history = []
    for i in range(budget):
        x = opt_cov.ask()
        y = rosenbrock(x)
        opt_cov.tell(x, y)
        history.append({'iter': i, 'x': np.array(x), 'y': y, 'best_y': opt_cov.best_y})
    
    # --- RUN GAUSSIAN PER CONFRONTO ---
    opt_gauss = ALBA(
        bounds=bounds,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=GaussianLocalSearchSampler(),
        use_drilling=False,
        seed=seed
    )
    
    for _ in range(budget):
        x = opt_gauss.ask()
        y = rosenbrock(x)
        opt_gauss.tell(x, y)
    
    print(f"\nRISULTATI:")
    print(f"  Gaussian: best_y = {opt_gauss.best_y:.4f}, best_x = {opt_gauss.best_x}")
    print(f"  Cov:      best_y = {opt_cov.best_y:.4f}, best_x = {opt_cov.best_x}")
    print(f"  Ottimo:   y = 0, x = [1, 1, 1]")
    
    if opt_cov.best_y > opt_gauss.best_y:
        print(f"\n⚠️ Cov PEGGIO di Gaussian di {opt_cov.best_y - opt_gauss.best_y:.2f}")
    else:
        print(f"\n✓ Cov MEGLIO di Gaussian di {opt_gauss.best_y - opt_cov.best_y:.2f}")
    
    # --- ANALISI TRACE ---
    print(f"\n{'-'*70}")
    print("ANALISI TRACE (chiamate al sampler)")
    print(f"{'-'*70}")
    
    n_cov_calls = sum(1 for t in tracer.trace if t.get('will_use_cov', False))
    n_fallback = sum(1 for t in tracer.trace if t.get('will_use_cov') == False and t.get('k') is not None)
    n_no_history = sum(1 for t in tracer.trace if t.get('k') is None)
    
    print(f"Totale chiamate sampler: {len(tracer.trace)}")
    print(f"  - Usato Cov: {n_cov_calls}")
    print(f"  - Fallback a Gaussian (condition>1000): {n_fallback}")
    print(f"  - No history sufficiente: {n_no_history}")
    
    # Trova quando il condizionamento esplode
    print(f"\n{'-'*70}")
    print("EVOLUZIONE CONDIZIONAMENTO")
    print(f"{'-'*70}")
    
    cond_history = [(t['call'], t.get('condition', 0)) for t in tracer.trace if 'condition' in t]
    
    for call, cond in cond_history[::10]:  # Ogni 10 chiamate
        status = "FALLBACK" if cond > 1000 else "OK"
        print(f"  Call {call:3d}: condition = {cond:10.2f} [{status}]")
    
    # Trova le chiamate problematiche
    print(f"\n{'-'*70}")
    print("CHIAMATE CON ALTO CONDIZIONAMENTO (>500)")
    print(f"{'-'*70}")
    
    high_cond = [t for t in tracer.trace if t.get('condition', 0) > 500]
    for t in high_cond[:5]:
        print(f"  Call {t['call']}: cond={t['condition']:.0f}, will_use_cov={t.get('will_use_cov')}")
        print(f"    best_x = {t.get('best_x')}")
        print(f"    mu_w = {t.get('mu_w')}")
        print(f"    dist_best_x_to_opt = {t.get('dist_best_x_to_opt'):.2f}")
        print(f"    dist_mu_w_to_opt = {t.get('dist_mu_w_to_opt'):.2f}")
    
    # Analizza distanza da ottimo nel tempo
    print(f"\n{'-'*70}")
    print("EVOLUZIONE DISTANZA DA OTTIMO")
    print(f"{'-'*70}")
    
    opt = np.array([1, 1, 1])
    for step in [10, 20, 30, 50, 70, 100]:
        if step <= len(history):
            best_y = min(h['y'] for h in history[:step])
            best_x = history[step-1]['best_y']
            # Trova il best_x effettivo
            best_x_arr = None
            for h in history[:step]:
                if h['y'] == best_y:
                    best_x_arr = h['x']
                    break
            if best_x_arr is not None:
                dist = np.linalg.norm(best_x_arr - opt)
                print(f"  Step {step:3d}: best_y = {best_y:8.2f}, dist_to_opt = {dist:.2f}")
    
    return tracer.trace, opt_cov, opt_gauss


def analyze_failure_pattern():
    """Analizza il pattern di fallimento su più seed."""
    print("\n" + "="*70)
    print("PATTERN DI FALLIMENTO SU SEED MULTIPLI")
    print("="*70)
    
    # Test su seed problematici e non problematici
    seeds_to_test = [0, 1, 2, 3, 7, 17, 19, 25, 27]  # Mix di buoni e cattivi
    
    results = []
    for seed in seeds_to_test:
        bounds = [(-5.0, 10.0)] * 3
        budget = 100
        
        # Cov
        tracer = TracingCovarianceSampler()
        opt_cov = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                       local_search_sampler=tracer, use_drilling=False, seed=seed)
        for _ in range(budget):
            x = opt_cov.ask()
            opt_cov.tell(x, rosenbrock(x))
        
        # Gauss
        opt_gauss = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                         local_search_sampler=GaussianLocalSearchSampler(), use_drilling=False, seed=seed)
        for _ in range(budget):
            x = opt_gauss.ask()
            opt_gauss.tell(x, rosenbrock(x))
        
        # Statistiche trace
        n_cov = sum(1 for t in tracer.trace if t.get('will_use_cov', False))
        n_fallback = sum(1 for t in tracer.trace if t.get('will_use_cov') == False and t.get('k') is not None)
        max_cond = max((t.get('condition', 0) for t in tracer.trace), default=0)
        
        results.append({
            'seed': seed,
            'gauss_y': opt_gauss.best_y,
            'cov_y': opt_cov.best_y,
            'diff': opt_cov.best_y - opt_gauss.best_y,
            'n_cov_calls': n_cov,
            'n_fallback': n_fallback,
            'max_condition': max_cond,
        })
    
    print(f"\n{'Seed':>4} | {'Gauss':>8} | {'Cov':>8} | {'Diff':>8} | {'Cov calls':>9} | {'Fallback':>8} | {'Max Cond':>10}")
    print("-" * 80)
    
    for r in results:
        marker = "⚠️" if r['diff'] > 10 else "✓"
        print(f"{r['seed']:4d} | {r['gauss_y']:8.2f} | {r['cov_y']:8.2f} | {r['diff']:+8.2f} | {r['n_cov_calls']:9d} | {r['n_fallback']:8d} | {r['max_condition']:10.0f} {marker}")
    
    return results


def investigate_early_divergence():
    """Indaga QUANDO Cov inizia a divergere da Gaussian."""
    print("\n" + "="*70)
    print("QUANDO COV INIZIA A DIVERGERE? (seed 3)")
    print("="*70)
    
    seed = 3
    bounds = [(-5.0, 10.0)] * 3
    budget = 100
    
    # Esegui entrambi step-by-step
    opt_cov = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                   local_search_sampler=CovarianceLocalSearchSampler(), use_drilling=False, seed=seed)
    
    opt_gauss = ALBA(bounds=bounds, total_budget=budget, local_search_ratio=0.3,
                     local_search_sampler=GaussianLocalSearchSampler(), use_drilling=False, seed=seed)
    
    print(f"\n{'Iter':>4} | {'Phase':>8} | {'Gauss y':>10} | {'Cov y':>10} | {'Gauss best':>10} | {'Cov best':>10} | {'Note'}")
    print("-" * 100)
    
    for i in range(budget):
        x_cov = opt_cov.ask()
        y_cov = rosenbrock(x_cov)
        opt_cov.tell(x_cov, y_cov)
        
        x_gauss = opt_gauss.ask()
        y_gauss = rosenbrock(x_gauss)
        opt_gauss.tell(x_gauss, y_gauss)
        
        # Determina la fase
        phase = "explore" if i < budget * 0.7 else "local"
        
        # Stampa solo quando divergono significativamente
        if abs(opt_cov.best_y - opt_gauss.best_y) > 5 or i in [10, 20, 30, 50, 70, 99]:
            note = ""
            if opt_cov.best_y > opt_gauss.best_y * 1.5:
                note = "← Cov worse"
            elif opt_gauss.best_y > opt_cov.best_y * 1.5:
                note = "← Gauss worse"
            
            print(f"{i:4d} | {phase:>8} | {y_gauss:10.2f} | {y_cov:10.2f} | {opt_gauss.best_y:10.2f} | {opt_cov.best_y:10.2f} | {note}")
    
    print(f"\nFINALE:")
    print(f"  Gaussian: {opt_gauss.best_y:.4f} at {opt_gauss.best_x}")
    print(f"  Cov:      {opt_cov.best_y:.4f} at {opt_cov.best_x}")


def main():
    # Deep analysis dei seed problematici
    trace, opt_cov, opt_gauss = deep_analysis_seed(3)
    
    # Pattern analysis
    analyze_failure_pattern()
    
    # Quando divergono?
    investigate_early_divergence()


if __name__ == "__main__":
    main()
