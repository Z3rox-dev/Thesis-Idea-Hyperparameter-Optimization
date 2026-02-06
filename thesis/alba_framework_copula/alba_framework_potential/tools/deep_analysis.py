
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from typing import List, Tuple

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.benchmarks import get_function

def deep_correlation_analysis(
    task_name: str = "sphere",
    dim: int = 10,
    budget: int = 1000
):
    print("="*80)
    print(f" ALBA-Potential: Deep Correlation Analysis")
    print(f" Task: {task_name} {dim}D | Budget: {budget}")
    print("="*80)
    
    # 1. Run Optimization
    func, bounds = get_function(task_name, dim)
    # Wrap func to track evaluations for truthful landscape analysis
    
    optimizer = ALBA(
        bounds=bounds, 
        maximize=False, 
        seed=42, 
        total_budget=budget,
        use_potential_field=True
        # Auto-scaling will happen
    )
    
    print("Running optimization...")
    optimizer.optimize(func, budget)
    
    # 2. Extract Data from Leaves
    leaves = optimizer.leaves
    tracker = optimizer._coherence_tracker
    
    if tracker is None:
        print("Error: Coherence tracker not initialized.")
        return

    data_potential = []
    data_true_error = []
    data_density = []
    data_visit_counts = []
    
    global_min = 0.0 # Known for Sphere/Rosen. Adjust if general.
    
    print("\nAnalyzing Leaf topology...")
    
    # For ground truth, we evaluate the function at the CENTER of each leaf
    for leaf in leaves:
        center = leaf.center()
        true_val = func(center)
        true_error = true_val - global_min
        
        # Get Potential
        pot = tracker.get_potential(leaf, leaves)
        
        # Get Density (Points / Volume implies infinite ratio, so just use Count)
        # Or better: Count normalized by volume fraction might be unstable.
        # Let's use simple Visit Count for now.
        count = len(leaf.tested_pairs)
        
        data_potential.append(pot)
        data_true_error.append(true_error)
        data_visit_counts.append(count)
        
    # Convert to numpy
    u = np.array(data_potential)
    e = np.array(data_true_error)
    n = np.array(data_visit_counts)
    
    # 3. Statistical Analysis
    
    # Q1: Does Potential correlate with True Error? 
    # (Do "Blue" zones actually have low error?)
    # We expect POSITIVE correlation (Low Pot = Low Error)
    pearson_r, _ = scipy.stats.pearsonr(u, e)
    spearman_r, _ = scipy.stats.spearmanr(u, e)
    
    print("\n[Metric 1] Fidelity: Does Potential map to True Error?")
    print(f"  > Pearson  Correlation (Linear): {pearson_r:.4f}")
    print(f"  > Spearman Correlation (Rank):   {spearman_r:.4f}")
    print("    (Expected: > 0.5. Higher is better.)")
    
    # Q2: Are points concentrated in Low Potential zones?
    # We break potential into bins [0.0-0.2, 0.2-0.4, ...] and sum samples
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    
    print("\n[Metric 2] Allocation: Where did we spend the budget?")
    print(f"  {'Pot Range':<12} {'Leaves':<8} {'Samples':<8} {'Avg Error':<12}")
    print("  " + "-"*45)
    
    points_in_good = 0
    points_in_bad = 0
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        
        # Boolean mask
        mask = (u >= low) & (u < high)
        n_leaves = np.sum(mask)
        n_samples = np.sum(n[mask])
        avg_err = np.mean(e[mask]) if n_leaves > 0 else 0.0
        
        print(f"  {low:.1f}-{high:.1f}      {n_leaves:<8d} {n_samples:<8d} {avg_err:<12.4f}")
        
        if high <= 0.4: points_in_good += n_samples
        if low >= 0.6: points_in_bad += n_samples

    print("  " + "-"*45)
    
    total_samples = np.sum(n)
    good_ratio = points_in_good / total_samples
    bad_ratio = points_in_bad / total_samples
    
    print(f"\n  > Points in LOW POTENTIAL (Blue/Good):  {points_in_good} ({good_ratio:.1%})")
    print(f"  > Points in HIGH POTENTIAL (Red/Bad):   {points_in_bad} ({bad_ratio:.1%})")
    
    # ... (previous code)
    
    # Q3: Field Consistency Analysis (New)
    # Check if the reconstructed potential 'u' actually respects the local gradients 'g'.
    # We compare (u_j - u_i) vs Predicted Drop (d_lm) for all edges in the graph.
    print("\n[Metric 3] Mathematical Consistency (R²)")
    
    edges = tracker._cache.edge_cache if hasattr(tracker._cache, 'edge_cache') else []
    # If edge cache is missing, we rebuild it
    if not edges:
        from alba_framework_potential.coherence import _build_knn_graph, _compute_predicted_drops
        edges = _build_knn_graph(leaves, k=6)
    
    if edges:
        d_lm_pred, _, valid_edges = _compute_predicted_drops(leaves, edges)
        potentials_dict = tracker._cache.potentials # Get raw dict
        
        u_diff_reconstructed = []
        d_lm_actual = []
        
        for idx, (i, j) in enumerate(valid_edges):
            if i in potentials_dict and j in potentials_dict:
                # Note: Potentials in tracker are normalized. We need raw u for physics check?
                # Actually, normalized u should still correlate with drops if scaling is linear.
                # But let's check correlation, which is scale-invariant.
                
                # Reconstructed diff
                diff = potentials_dict[j] - potentials_dict[i]
                u_diff_reconstructed.append(diff)
                
                # Local prediction (Projected Gradient)
                # d_lm is "how much we expect to DROP". 
                # If d_lm > 0, we go downhill. So u_j should be < u_i.
                # So (u_j - u_i) should be proportional to -d_lm.
                d_lm_actual.append(d_lm_pred[idx])
                
        if u_diff_reconstructed:
            r_consist, _ = scipy.stats.pearsonr(u_diff_reconstructed, d_lm_actual)
            r2_consist = r_consist**2
            print(f"  > Gradient Integration Consistency (R²): {r2_consist:.4f}")
            print(f"    (High R² = Local gradients merge into a valid global map)")
            print(f"    (Low  R² = Gradients are contradictory/noisy)")
    
    # Analysis interpretation
    print("\n" + "="*80)
    print("  RESEARCHER INTERPRETATION")
    print("="*80)
    
    # 1. Correlation Check
    if spearman_r > 0.6:
        print("✅ STRONG REALITY CORRELATION: The potential field accurately reflects the true error.")
    elif spearman_r > 0.3:
        print("⚠️ WEAK REALITY CORRELATION: The potential field captures trend but is noisy.")
    else:
        print("❌ NO REALITY CORRELATION: The potential field is disjoint from reality.")

    # 2. Consistency Check (New)
    if 'r2_consist' in locals():
        if r2_consist > 0.5:
            print("✅ MATHEMATICALLY CONSISTENT: The local gradients form a valid conservative field.")
        else:
            print("⚠️ MATHEMATICALLY INCONSISTENT: Local gradients contradict each other (Escher staircases).")
        
    # 3. Allocation Check
    if points_in_bad > total_samples * 0.3:
        print(f"⚠️ HIGH EXPLORATION/WASTE: {bad_ratio:.1%} of points are in Red Zones.")
    else:
        print("✅ EFFICIENT SAMPLING: Most points are in promising regions.")
        
    # Generate scatter plot
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(u, e, c=n, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(sc, label='Number of Samples (Visit Count)')
    plt.xlabel('Potential Value (u)')
    plt.ylabel('True Error (f(x) - f*)')
    plt.title(f'Analytical Validation: Potential vs Truth ({dim}D)')
    plt.grid(True, alpha=0.3)
    plt.axvline(0.4, color='green', linestyle='--', label='Good/Bad Threshold')
    plt.legend()
    plt.savefig('correlation_analysis.png')
    print("\nScatter plot saved to 'correlation_analysis.png'")

if __name__ == "__main__":
    deep_correlation_analysis("rosenbrock", 10, 1000)
