
import numpy as np
import scipy.sparse.csgraph
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.coherence import _build_knn_graph

def run_forensic_diagnostics():
    print("="*80)
    print(" ALBA Forensics: Extended Mathematical Integrity Check")
    print("="*80)
    
    dim = 10
    budget = 1000
    
    # --- Function Definitions & Gradients ---
    
    # 1. Sphere
    def sphere(x): return np.sum(x**2)
    def sphere_grad(x): return 2*x

    # 2. Ellipsoid (Ill-conditioned)
    # Weights from 1 to 1000
    ellipsoid_weights = np.array([1000**(i/(dim-1)) for i in range(dim)])
    def ellipsoid(x): 
        return np.sum(ellipsoid_weights * x**2)
    def ellipsoid_grad(x): 
        return 2 * ellipsoid_weights * x

    # 3. Rastrigin (Multimodal - High Frequency Noise)
    def rastrigin(x):
        return 10*dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    def rastrigin_grad(x):
        return 2*x + 10 * 2*np.pi * np.sin(2*np.pi*x)

    # 4. Ackley (Flat outer, steep inner)
    def ackley(x):
        a, b, c = 20, 0.2, 2*np.pi
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(c*x))
        n = float(dim)
        return -a * np.exp(-b * np.sqrt(s1/n)) - np.exp(s2/n) + a + np.exp(1)
        
    def ackley_grad(x):
        a, b, c = 20, 0.2, 2*np.pi
        n = float(dim)
        s1 = np.sum(x**2)
        
        # Avoid div by zero at origin
        if s1 < 1e-15: return np.zeros_like(x)
        
        term1 = -a * np.exp(-b * np.sqrt(s1/n)) * (-b / (2*np.sqrt(s1/n))) * (2*x/n)
        
        s2 = np.sum(np.cos(c*x))
        term2 = -np.exp(s2/n) * (1/n) * (-np.sin(c*x)*c)
        
        return term1 + term2

    test_suite = [
        ("Sphere", sphere, sphere_grad),
        ("Ellipsoid", ellipsoid, ellipsoid_grad),
        ("Rastrigin", rastrigin, rastrigin_grad),
        ("Ackley", ackley, ackley_grad)
    ]
    
    print(f"Setup: {dim}D, Budget {budget}, Factor 6.0")
    print(f"{'Function':<12} | {'Leaves':<6} | {'GradSim':<9} | {'MagRatio':<8} | {'Connect':<7} | {'PotAcc':<7} | {'Interpretation'}")
    print("-" * 110)

    for name, func, grad_func in test_suite:
        bounds = [(-5.0, 5.0)] * dim
        
        # Reset seed for fair comparison
        opt = ALBA(
            bounds=bounds, 
            maximize=False, 
            total_budget=budget, 
            use_potential_field=True, 
            split_trials_factor=6.0,
            seed=42
        )
        opt.optimize(func, budget)
        
        leaves = opt.leaves
        n_leaves = len(leaves)
        
        # --- Audit 1: LGS Fidelity ---
        cos_sims = []
        mag_ratios = []
        
        for leaf in leaves:
            if leaf.lgs_model and leaf.lgs_model.get('grad') is not None:
                widths = leaf.widths()
                grad_norm = leaf.lgs_model['grad']
                grad_est_real = grad_norm / (widths + 1e-9)
                
                center = leaf.center()
                grad_true = grad_func(center)
                
                norm_est = np.linalg.norm(grad_est_real)
                norm_true = np.linalg.norm(grad_true)
                
                if norm_est > 1e-9 and norm_true > 1e-9:
                    sim = np.dot(grad_est_real, grad_true) / (norm_est * norm_true)
                    cos_sims.append(sim)
                    mag_ratios.append(norm_est / norm_true)
        
        avg_sim = np.mean(cos_sims) if cos_sims else 0.0
        avg_mag = np.mean(mag_ratios) if mag_ratios else 0.0
        
        # --- Audit 2: Connectivity ---
        edges = _build_knn_graph(leaves, k=6)
        adj = np.zeros((n_leaves, n_leaves))
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        n_components, _ = scipy.sparse.csgraph.connected_components(adj, directed=False)
        
        # --- Audit 3: Potential Accuracy ---
        tracker = opt._coherence_tracker
        potentials_dict = tracker._cache.potentials
        agreements = []
        for i, j in edges:
            if i in potentials_dict and j in potentials_dict:
                u_i, u_j = potentials_dict[i], potentials_dict[j]
                pred_better = u_j < u_i
                f_i, f_j = func(leaves[i].center()), func(leaves[j].center())
                true_better = f_j < f_i
                agreements.append(1 if pred_better == true_better else 0)
        
        acc = np.mean(agreements) if agreements else 0.0
        
        # Interpretation
        interp = "✅ OK"
        if avg_sim > -0.2: interp = "⚠️ Noisy Gradient"
        if acc < 0.6: interp = "❌ Bad Guidance"
        if avg_mag > 10.0: interp = "⚠️ Scale Error"
        
        print(f"{name:<12} | {n_leaves:<6} | {avg_sim:<9.4f} | {avg_mag:<8.2f} | {n_components:<7} | {acc:<7.1%} | {interp}")

if __name__ == "__main__":
    run_forensic_diagnostics()
