#!/usr/bin/env python3
"""
Analisi comparativa ORIGINAL vs FIXED per capire perché seed 10 va male con FIXED
"""

from __future__ import annotations
import numpy as np
import sys
import types
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace/thesis')

from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
from ConfigSpace import Configuration


def _enable_fast_lookup(benchmark):
    """Abilita lookup veloce con cache"""
    import pandas as pd
    
    df = benchmark.table
    if not hasattr(df, "columns") or "result" not in df.columns:
        return
    
    idx_cols = [c for c in df.columns if c != "result"]
    indexed = df.set_index(idx_cols, drop=False)
    try:
        indexed = indexed.sort_index()
    except Exception:
        pass
    
    cache = {}
    max_cache = 200_000
    
    def _search_dataframe_fast(self, row_dict, _df_unused):
        key = tuple(row_dict[c] for c in idx_cols)
        hit = cache.get(key)
        if hit is not None:
            return hit
        
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise AssertionError(f"Multiple matches for {row_dict}")
            row = row.iloc[0]
        res = row["result"]
        
        if len(cache) < max_cache:
            cache[key] = res
        return res
    
    benchmark._search_dataframe = types.MethodType(_search_dataframe_fast, benchmark)


def run_with_detailed_trace(optimizer_class, bounds, seed, budget, objective, name):
    """Esegue con trace dettagliato"""
    
    opt = optimizer_class(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        split_depth_max=8,
    )
    
    trace = []
    best_history = []
    
    for i in range(budget):
        x = opt.ask()
        y = objective(x)
        
        # Capture pre-tell state
        state = {
            'iter': i,
            'x': x.copy(),
            'y': y,
            'best_before': opt.best_y if hasattr(opt, 'best_y') else (opt.inner.best_y if hasattr(opt, 'inner') else None),
        }
        
        # Get gamma and leaf info
        if hasattr(opt, 'inner'):
            inner = opt.inner
        else:
            inner = opt
            
        state['gamma'] = inner.gamma
        state['n_leaves'] = len(inner.leaves)
        state['phase'] = 'exploration' if inner.iteration < inner.exploration_budget else 'local_search'
        
        # Leaf selection info
        if inner.last_cube is not None:
            state['selected_cube_trials'] = inner.last_cube.n_trials
            state['selected_cube_good'] = inner.last_cube.n_good
            state['selected_cube_ratio'] = inner.last_cube.good_ratio()
            state['selected_cube_depth'] = inner.last_cube.depth
        
        opt.tell(x, y) if hasattr(opt, 'tell') else None
        
        if hasattr(opt, 'best_y'):
            best = opt.best_y
        elif hasattr(opt, 'inner'):
            best = opt.inner.best_y
        else:
            best = inner.best_y
            
        state['best_after'] = best
        state['improved'] = state['best_after'] < state['best_before'] if state['best_before'] is not None else False
        
        trace.append(state)
        best_history.append(best)
    
    return best_history[-1], trace


def compare_traces(trace_orig, trace_fixed, name_orig="ORIGINAL", name_fixed="FIXED"):
    """Confronta due trace per trovare divergenze"""
    
    print("\n" + "=" * 80)
    print("TRACE COMPARISON: Dove divergono?")
    print("=" * 80)
    
    # Find first divergence
    for i, (orig, fixed) in enumerate(zip(trace_orig, trace_fixed)):
        # Check if y values are different (same x should give same y)
        if abs(orig['y'] - fixed['y']) > 1e-9:
            print(f"\nDIVERGENZA trovata a iter {i}!")
            print(f"  {name_orig}: x={orig['x'][:3]}..., y={orig['y']:.6f}")
            print(f"  {name_fixed}: x={fixed['x'][:3]}..., y={fixed['y']:.6f}")
            
            # Show context
            if i > 0:
                print(f"\n  Stato a iter {i-1}:")
                print(f"    {name_orig}: best={trace_orig[i-1]['best_after']:.6f}, "
                      f"n_leaves={trace_orig[i-1]['n_leaves']}, gamma={trace_orig[i-1]['gamma']:.4f}")
                print(f"    {name_fixed}: best={trace_fixed[i-1]['best_after']:.6f}, "
                      f"n_leaves={trace_fixed[i-1]['n_leaves']}, gamma={trace_fixed[i-1]['gamma']:.4f}")
            
            # This is where they diverge - likely due to random sampling difference
            break
    
    # Compare improvement patterns
    orig_improvements = [i for i, t in enumerate(trace_orig) if t['improved']]
    fixed_improvements = [i for i, t in enumerate(trace_fixed) if t['improved']]
    
    print(f"\n{name_orig} improvements at iters: {orig_improvements[:20]}...")
    print(f"{name_fixed} improvements at iters: {fixed_improvements[:20]}...")
    
    # Compare exploration vs local search performance
    exp_budget = int(600 * 0.70)  # 420
    
    orig_exp_best = min(t['best_after'] for t in trace_orig[:exp_budget])
    fixed_exp_best = min(t['best_after'] for t in trace_fixed[:exp_budget])
    
    print(f"\nBest at end of EXPLORATION (iter {exp_budget}):")
    print(f"  {name_orig}: {orig_exp_best:.6f}")
    print(f"  {name_fixed}: {fixed_exp_best:.6f}")
    
    # Check global random samples in FIXED
    # The FIXED version has 5% global random - let's see where those hit
    print(f"\nAnalisi campioni globali random (FIXED ha 5% global random):")
    
    # Count how many iterations had very different x values
    # (indicative of global random vs local sampling)
    
    return orig_improvements, fixed_improvements


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--budget', type=int, default=600)
    parser.add_argument('--task_id', type=int, default=31)
    args = parser.parse_args()
    
    print("Loading benchmark...")
    bench = TabularBenchmark(model='nn', task_id=args.task_id)
    _enable_fast_lookup(bench)
    
    cs = bench.get_configuration_space()
    hp_names = list(cs.get_hyperparameter_names())
    
    # Get bounds
    bounds = []
    for hp_name in hp_names:
        hp = cs.get_hyperparameter(hp_name)
        if hasattr(hp, 'lower'):
            bounds.append((float(hp.lower), float(hp.upper)))
        elif hasattr(hp, 'sequence'):
            bounds.append((0.0, float(len(hp.sequence) - 1)))
        else:
            bounds.append((0.0, 1.0))
    
    def objective(x):
        config_dict = {}
        for i, hp_name in enumerate(hp_names):
            hp = cs.get_hyperparameter(hp_name)
            if hasattr(hp, 'sequence'):
                idx = int(round(np.clip(x[i], 0, len(hp.sequence) - 1)))
                config_dict[hp_name] = hp.sequence[idx]
            else:
                config_dict[hp_name] = float(np.clip(x[i], hp.lower, hp.upper))
        config = Configuration(cs, values=config_dict)
        result = bench.objective_function(config)
        return float(result['function_value'])
    
    # Import both versions
    from hpo_v5s_more_novelty_standalone import HPOptimizerV5s
    from debug_v5s_fixed import HPOptimizerV5sFixed
    
    print(f"\nRunning ORIGINAL (seed {args.seed})...")
    best_orig, trace_orig = run_with_detailed_trace(
        HPOptimizerV5s, bounds, args.seed, args.budget, objective, "ORIGINAL"
    )
    print(f"  Final: {best_orig:.6f}")
    
    print(f"\nRunning FIXED (seed {args.seed})...")
    best_fixed, trace_fixed = run_with_detailed_trace(
        HPOptimizerV5sFixed, bounds, args.seed, args.budget, objective, "FIXED"  
    )
    print(f"  Final: {best_fixed:.6f}")
    
    # Compare
    compare_traces(trace_orig, trace_fixed)
    
    # Detailed iteration-by-iteration for first divergence
    print("\n" + "=" * 80)
    print("ANALISI ITERAZIONE PER ITERAZIONE (prime 30 iter)")
    print("=" * 80)
    
    for i in range(min(30, len(trace_orig))):
        o = trace_orig[i]
        f = trace_fixed[i]
        
        same_x = np.allclose(o['x'], f['x'])
        same_y = abs(o['y'] - f['y']) < 1e-9
        
        marker = "" if same_x else " <-- DIVERGE"
        
        print(f"[{i:3d}] ORIG: y={o['y']:.6f}, best={o['best_after']:.6f} | "
              f"FIXED: y={f['y']:.6f}, best={f['best_after']:.6f}{marker}")
        
        if not same_x:
            print(f"       ORIG x: {o['x']}")
            print(f"       FIXED x: {f['x']}")
            break


if __name__ == "__main__":
    main()
