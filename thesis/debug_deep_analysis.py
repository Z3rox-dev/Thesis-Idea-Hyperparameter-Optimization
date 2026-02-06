#!/usr/bin/env python3
"""
Analisi approfondita del comportamento dell'algoritmo HPO_v5s.
- Sampling del landscape della funzione obiettivo
- Tracing dettagliato di ogni componente
- Visualizzazione delle decisioni dell'algoritmo
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


# =============================================================================
# FAST LOOKUP (copiato da benchmark_triple_nn.py)
# =============================================================================
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
    print(f"  Fast lookup enabled: {len(idx_cols)} index columns, cache size {max_cache}")


# =============================================================================
# LANDSCAPE ANALYSIS
# =============================================================================
def analyze_landscape(bench, cs, hp_names, n_samples=1000):
    """Campiona random la funzione per capire il landscape"""
    print("\n" + "=" * 80)
    print("LANDSCAPE ANALYSIS")
    print("=" * 80)
    
    losses = []
    configs = []
    
    print(f"Sampling {n_samples} random configurations...")
    for i in range(n_samples):
        config = cs.sample_configuration()
        result = bench.objective_function(config)
        loss = float(result['function_value'])
        losses.append(loss)
        configs.append(config)
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples} sampled, current min={min(losses):.6f}")
    
    losses = np.array(losses)
    
    print(f"\nLandscape statistics:")
    print(f"  Min:    {losses.min():.6f}")
    print(f"  Max:    {losses.max():.6f}")
    print(f"  Mean:   {losses.mean():.6f}")
    print(f"  Std:    {losses.std():.6f}")
    print(f"  Median: {np.median(losses):.6f}")
    
    # Percentili
    percentiles = [1, 5, 10, 20, 50, 80, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        print(f"    {p}%: {np.percentile(losses, p):.6f}")
    
    # Best configs
    best_idx = np.argsort(losses)[:5]
    print(f"\n  Top 5 configurations:")
    for rank, idx in enumerate(best_idx, 1):
        print(f"    #{rank}: loss={losses[idx]:.6f}")
        for hp_name in hp_names:
            print(f"         {hp_name}={configs[idx][hp_name]}")
    
    return losses, configs


# =============================================================================
# TRACING OPTIMIZER
# =============================================================================
class HPOptimizerV5sTraced:
    """Versione con tracing dettagliato di ogni componente"""
    
    def __init__(self, bounds, maximize=False, seed=42, total_budget=200, 
                 split_depth_max=8, version='original'):
        from hpo_v5s_more_novelty_standalone import HPOptimizerV5s, Cube
        
        self.version = version
        self.inner = HPOptimizerV5s(
            bounds=bounds,
            maximize=maximize,
            seed=seed,
            total_budget=total_budget,
            split_depth_max=split_depth_max,
        )
        
        # Trace storage
        self.trace = []
        self.leaf_selection_trace = []
        self.candidate_trace = []
        
    def ask(self):
        x = self.inner.ask()
        return x
    
    def tell(self, x, y_raw):
        # Pre-tell state
        pre_state = self._capture_state()
        pre_state['x'] = x.copy()
        pre_state['y_raw'] = y_raw
        
        # Capture leaf selection details if in exploration
        if self.inner.iteration < self.inner.exploration_budget and self.inner.last_cube is not None:
            leaf_info = self._capture_leaf_selection()
            pre_state['leaf_selection'] = leaf_info
        
        self.inner.tell(x, y_raw)
        
        # Post-tell state
        post_state = self._capture_state()
        
        # Detect anomalies
        anomalies = self._detect_anomalies(pre_state, post_state)
        pre_state['anomalies'] = anomalies
        
        self.trace.append(pre_state)
        
        return anomalies
    
    def _capture_state(self):
        opt = self.inner
        return {
            'iteration': opt.iteration,
            'phase': 'exploration' if opt.iteration < opt.exploration_budget else 'local_search',
            'best_y': opt.best_y,
            'gamma': opt.gamma,
            'n_leaves': len(opt.leaves),
            'max_depth': max((l.depth for l in opt.leaves), default=0),
            'leaf_trials': [l.n_trials for l in opt.leaves],
            'leaf_good_ratios': [l.good_ratio() for l in opt.leaves],
        }
    
    def _capture_leaf_selection(self):
        opt = self.inner
        if not opt.leaves:
            return None
        
        scores = []
        details = []
        for c in opt.leaves:
            ratio = c.good_ratio()
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= opt.dim + 2:
                    model_bonus = 0.1
            score = ratio + exploration + model_bonus
            scores.append(score)
            details.append({
                'n_trials': c.n_trials,
                'n_good': c.n_good,
                'good_ratio': ratio,
                'exploration_bonus': exploration,
                'model_bonus': model_bonus,
                'total_score': score,
                'depth': c.depth,
                'has_model': c.lgs_model is not None,
            })
        
        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()
        probs = np.exp(scores_arr * 3)
        probs = probs / probs.sum()
        
        # Find selected leaf
        selected_idx = None
        if opt.last_cube is not None:
            for i, leaf in enumerate(opt.leaves):
                if leaf is opt.last_cube:
                    selected_idx = i
                    break
        
        return {
            'n_leaves': len(opt.leaves),
            'scores': scores,
            'probs': probs.tolist(),
            'selected_idx': selected_idx,
            'selected_prob': probs[selected_idx] if selected_idx is not None else None,
            'details': details,
        }
    
    def _detect_anomalies(self, pre, post):
        anomalies = []
        
        # Check for extreme gamma values
        if abs(pre['gamma']) > 10:
            anomalies.append(f"EXTREME_GAMMA: {pre['gamma']:.4f}")
        
        # Check for imbalanced leaf selection
        if 'leaf_selection' in pre and pre['leaf_selection'] is not None:
            ls = pre['leaf_selection']
            max_prob = max(ls['probs'])
            if max_prob > 0.9:
                anomalies.append(f"DOMINANT_LEAF: prob={max_prob:.4f}")
            
            # Check if selected leaf has very few trials
            if ls['selected_idx'] is not None:
                detail = ls['details'][ls['selected_idx']]
                if detail['n_trials'] == 0 and detail['good_ratio'] == 0.5:
                    anomalies.append("SELECTED_EMPTY_LEAF")
                if detail['n_trials'] == 1 and detail['good_ratio'] == 1.0:
                    anomalies.append(f"SELECTED_1GOOD_LEAF")
        
        # Check for stagnation
        # (will be filled by caller)
        
        return anomalies
    
    @property
    def best_y(self):
        return self.inner.best_y
    
    @property
    def best_x(self):
        return self.inner.best_x


# =============================================================================
# DEEP DEBUG RUN
# =============================================================================
def run_deep_debug(bench, cs, hp_names, bounds, seed, budget, verbose=True):
    """Esegue l'ottimizzazione con tracing completo"""
    
    print(f"\n{'='*80}")
    print(f"DEEP DEBUG RUN - Seed {seed}")
    print(f"{'='*80}")
    
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
    
    opt = HPOptimizerV5sTraced(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        split_depth_max=8,
    )
    
    stagnation = 0
    last_improvement = 0
    best_so_far = float('inf')
    
    anomaly_counts = {}
    
    for i in range(budget):
        x = opt.ask()
        y = objective(x)
        anomalies = opt.tell(x, y)
        
        # Track stagnation
        if y < best_so_far:
            best_so_far = y
            stagnation = 0
            last_improvement = i
        else:
            stagnation += 1
        
        # Count anomalies
        for a in anomalies:
            anomaly_counts[a] = anomaly_counts.get(a, 0) + 1
        
        # Periodic verbose output
        if verbose and (i + 1) % 50 == 0:
            phase = 'EXP' if i < opt.inner.exploration_budget else 'LS'
            print(f"[{i+1:3d}] best={opt.best_y:.6f} | stag={stagnation:3d} | "
                  f"leaves={len(opt.inner.leaves):2d} | gamma={opt.inner.gamma:.4f} | {phase}")
            
            # Show anomalies if any
            if anomalies:
                print(f"       ANOMALIES: {anomalies}")
    
    print(f"\nFinal: best={opt.best_y:.6f}, last_improvement={last_improvement}")
    
    # Anomaly summary
    if anomaly_counts:
        print(f"\nAnomaly summary:")
        for a, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            print(f"  {a}: {count} times")
    
    return opt, opt.trace


def analyze_trace(trace, target_loss=0.092):
    """Analizza la trace per capire cosa è andato storto"""
    print(f"\n{'='*80}")
    print("TRACE ANALYSIS")
    print(f"{'='*80}")
    
    # Find when we got stuck
    best_history = [t['best_y'] for t in trace]
    best_final = best_history[-1]
    
    # Find last improvement
    last_improvement = 0
    for i, b in enumerate(best_history):
        if i > 0 and b < best_history[i-1]:
            last_improvement = i
    
    print(f"Final best: {best_final:.6f}")
    print(f"Last improvement at iter: {last_improvement}")
    print(f"Stuck: {'YES' if best_final > target_loss else 'NO'}")
    
    # Analyze leaf selection patterns
    leaf_selections = [t.get('leaf_selection') for t in trace if t.get('leaf_selection')]
    
    if leaf_selections:
        # Check for dominant leaf problem
        dominant_count = 0
        for ls in leaf_selections:
            if max(ls['probs']) > 0.8:
                dominant_count += 1
        
        print(f"\nLeaf selection analysis:")
        print(f"  Times with dominant leaf (prob>0.8): {dominant_count}/{len(leaf_selections)}")
        
        # Analyze selected leaf properties
        selected_trials = [ls['details'][ls['selected_idx']]['n_trials'] 
                         for ls in leaf_selections if ls['selected_idx'] is not None]
        selected_ratios = [ls['details'][ls['selected_idx']]['good_ratio']
                         for ls in leaf_selections if ls['selected_idx'] is not None]
        
        print(f"  Selected leaf avg trials: {np.mean(selected_trials):.1f}")
        print(f"  Selected leaf avg good_ratio: {np.mean(selected_ratios):.3f}")
        
        # Check for 1-trial-1-good problem
        one_good_selections = sum(1 for ls in leaf_selections 
                                  if ls['selected_idx'] is not None and
                                  ls['details'][ls['selected_idx']]['n_trials'] == 1 and
                                  ls['details'][ls['selected_idx']]['good_ratio'] == 1.0)
        print(f"  Times selected 1-trial-1-good leaf: {one_good_selections}")
    
    # Gamma evolution
    gammas = [t['gamma'] for t in trace]
    print(f"\nGamma evolution:")
    print(f"  Initial: {gammas[0]:.4f}")
    print(f"  At iter 100: {gammas[min(99, len(gammas)-1)]:.4f}")
    print(f"  At iter 300: {gammas[min(299, len(gammas)-1)]:.4f}")
    print(f"  Final: {gammas[-1]:.4f}")
    
    return {
        'best_final': best_final,
        'last_improvement': last_improvement,
        'stuck': best_final > target_loss,
    }


# =============================================================================
# COMPARE ORIGINAL VS FIXED AT SPECIFIC ITERATIONS
# =============================================================================
def compare_at_iteration(seed, target_iter, bench, cs, hp_names, bounds, budget):
    """Confronta lo stato dell'algoritmo a una specifica iterazione"""
    
    print(f"\n{'='*80}")
    print(f"COMPARISON AT ITERATION {target_iter} - Seed {seed}")
    print(f"{'='*80}")
    
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
    
    from hpo_v5s_more_novelty_standalone import HPOptimizerV5s
    
    # Run original
    opt_orig = HPOptimizerV5s(bounds=bounds, maximize=False, seed=seed, 
                              total_budget=budget, split_depth_max=8)
    
    for i in range(target_iter):
        x = opt_orig.ask()
        y = objective(x)
        opt_orig.tell(x, y)
    
    print(f"\nORIGINAL at iter {target_iter}:")
    print(f"  best_y: {opt_orig.best_y:.6f}")
    print(f"  gamma: {opt_orig.gamma:.4f}")
    print(f"  n_leaves: {len(opt_orig.leaves)}")
    print(f"  Leaf good_ratios: {[f'{l.good_ratio():.2f}' for l in opt_orig.leaves[:10]]}...")
    
    # Show leaf details
    print(f"\n  Top 5 leaves by good_ratio:")
    sorted_leaves = sorted(opt_orig.leaves, key=lambda l: l.good_ratio(), reverse=True)[:5]
    for j, leaf in enumerate(sorted_leaves):
        print(f"    #{j+1}: ratio={leaf.good_ratio():.3f}, trials={leaf.n_trials}, "
              f"good={leaf.n_good}, depth={leaf.depth}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--budget', type=int, default=600)
    parser.add_argument('--task_id', type=int, default=31)
    parser.add_argument('--target_loss', type=float, default=0.092)
    parser.add_argument('--landscape_samples', type=int, default=500)
    parser.add_argument('--compare_iter', type=int, default=None)
    parser.add_argument('--skip_landscape', action='store_true')
    args = parser.parse_args()
    
    print("Loading benchmark...")
    bench = TabularBenchmark(model='nn', task_id=args.task_id)
    _enable_fast_lookup(bench)
    
    cs = bench.get_configuration_space()
    hp_names = list(cs.get_hyperparameter_names())
    
    print(f"Hyperparameters: {hp_names}")
    
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
    
    # 1. Landscape analysis
    if not args.skip_landscape:
        analyze_landscape(bench, cs, hp_names, n_samples=args.landscape_samples)
    
    # 2. Deep debug run
    opt, trace = run_deep_debug(bench, cs, hp_names, bounds, args.seed, args.budget)
    
    # 3. Trace analysis
    analyze_trace(trace, args.target_loss)
    
    # 4. Compare at specific iteration
    if args.compare_iter:
        compare_at_iteration(args.seed, args.compare_iter, bench, cs, hp_names, bounds, args.budget)


if __name__ == "__main__":
    main()
