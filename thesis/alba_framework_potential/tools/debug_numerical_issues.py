#!/usr/bin/env python3
"""
NUMERICAL DEBUGGING SCRIPT FOR ALBA HYBRID DRILL

Questo script cerca bug numerici in:
1. Matrice di covarianza (singolarità, NaN, Inf, condizionamento)
2. Drilling (CMA step size, evolution paths)
3. Gradienti LGS (esplosione, NaN)
4. Campo potenziale (solver, normalizzazione)
5. Local search covariance (degenerate distributions)

Approccio:
- Instrumenta ogni componente critico
- Dumpa statistiche numeriche
- Segnala anomalie
"""

import sys
sys.path.insert(0, '/mnt/workspace')
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import traceback

# =============================================================================
# NUMERICAL HEALTH TRACKER
# =============================================================================

@dataclass
class NumericalHealthReport:
    """Accumula anomalie numeriche."""
    issues: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, List[float]] = field(default_factory=dict)
    
    def log_issue(self, component: str, issue_type: str, details: Dict):
        self.issues.append({
            "component": component,
            "type": issue_type,
            **details
        })
    
    def log_stat(self, name: str, value: float):
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(value)
    
    def has_nan(self, arr: np.ndarray, component: str, var_name: str) -> bool:
        if arr is None:
            return False
        arr = np.asarray(arr)
        if np.any(np.isnan(arr)):
            self.log_issue(component, "NaN", {"variable": var_name, "count": int(np.sum(np.isnan(arr)))})
            return True
        return False
    
    def has_inf(self, arr: np.ndarray, component: str, var_name: str) -> bool:
        if arr is None:
            return False
        arr = np.asarray(arr)
        if np.any(np.isinf(arr)):
            self.log_issue(component, "Inf", {"variable": var_name, "count": int(np.sum(np.isinf(arr)))})
            return True
        return False
    
    def check_condition_number(self, matrix: np.ndarray, component: str, var_name: str, threshold: float = 1e10) -> bool:
        if matrix is None:
            return False
        try:
            cond = np.linalg.cond(matrix)
            self.log_stat(f"{component}_{var_name}_cond", cond)
            if cond > threshold:
                self.log_issue(component, "IllConditioned", {"variable": var_name, "cond": float(cond)})
                return True
        except:
            self.log_issue(component, "ConditionError", {"variable": var_name})
            return True
        return False
    
    def check_range(self, arr: np.ndarray, component: str, var_name: str, 
                    min_val: float = -1e10, max_val: float = 1e10) -> bool:
        if arr is None:
            return False
        arr = np.asarray(arr)
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        self.log_stat(f"{component}_{var_name}_min", arr_min)
        self.log_stat(f"{component}_{var_name}_max", arr_max)
        if arr_min < min_val or arr_max > max_val:
            self.log_issue(component, "OutOfRange", {"variable": var_name, "min": arr_min, "max": arr_max})
            return True
        return False
    
    def print_summary(self):
        print("\n" + "="*80)
        print("NUMERICAL HEALTH REPORT")
        print("="*80)
        
        if not self.issues:
            print("✓ No numerical issues detected!")
        else:
            print(f"✗ Found {len(self.issues)} issues:\n")
            for issue in self.issues:
                print(f"  [{issue['component']}] {issue['type']}: {issue}")
        
        print("\n" + "-"*80)
        print("STATISTICS SUMMARY:")
        print("-"*80)
        for name, values in sorted(self.stats.items()):
            arr = np.array(values)
            if len(arr) > 0:
                print(f"  {name:<40}: min={np.min(arr):.2e}, max={np.max(arr):.2e}, mean={np.mean(arr):.2e}, std={np.std(arr):.2e}")


# =============================================================================
# INSTRUMENTED ALBA
# =============================================================================

def run_instrumented_alba(func, param_space, budget: int, report: NumericalHealthReport):
    """Run ALBA with full numerical instrumentation."""
    
    from alba_framework_potential.optimizer import ALBA
    from alba_framework_potential.local_search import CovarianceLocalSearchSampler
    
    cov_sampler = CovarianceLocalSearchSampler(
        radius_start=0.15,
        radius_end=0.01,
        top_k_fraction=0.15,
        min_points_fit=10
    )
    
    opt = ALBA(
        param_space=param_space,
        total_budget=budget,
        local_search_ratio=0.3,
        local_search_sampler=cov_sampler,
        use_drilling=True,
        use_potential_field=True,
        seed=42
    )
    
    for iteration in range(budget):
        try:
            config = opt.ask()
            y = func(config)
            opt.tell(config, y)
            
            # =====================================================================
            # INSTRUMENT: Check internal state
            # =====================================================================
            
            # 1. Check driller state
            if opt.driller is not None:
                driller = opt.driller
                report.has_nan(driller.mu, "Driller", "mu")
                report.has_inf(driller.mu, "Driller", "mu")
                report.check_range(driller.mu, "Driller", "mu")
                
                report.log_stat("Driller_sigma", driller.sigma)
                if driller.sigma < 1e-15:
                    report.log_issue("Driller", "SigmaCollapse", {"sigma": driller.sigma, "iter": iteration})
                if driller.sigma > 1e10:
                    report.log_issue("Driller", "SigmaExplosion", {"sigma": driller.sigma, "iter": iteration})
                
                report.has_nan(driller.C, "Driller", "C")
                report.has_inf(driller.C, "Driller", "C")
                report.check_condition_number(driller.C, "Driller", "C")
                
                report.has_nan(driller.pc, "Driller", "pc")
                report.has_nan(driller.ps, "Driller", "ps")
            
            # 2. Check all cubes' LGS models
            leaves = opt.leaves
            for leaf_idx, leaf in enumerate(leaves):
                if leaf.lgs_model is not None:
                    model = leaf.lgs_model
                    
                    # Gradient
                    grad = model.get("grad")
                    if grad is not None:
                        report.has_nan(grad, "LGS", f"grad_leaf{leaf_idx}")
                        report.has_inf(grad, "LGS", f"grad_leaf{leaf_idx}")
                        grad_norm = np.linalg.norm(grad)
                        report.log_stat("LGS_grad_norm", grad_norm)
                        if grad_norm > 1e6:
                            report.log_issue("LGS", "GradientExplosion", {"leaf": leaf_idx, "norm": grad_norm, "iter": iteration})
                    
                    # Inverse covariance
                    inv_cov = model.get("inv_cov")
                    if inv_cov is not None:
                        report.has_nan(inv_cov, "LGS", f"inv_cov_leaf{leaf_idx}")
                        report.has_inf(inv_cov, "LGS", f"inv_cov_leaf{leaf_idx}")
                        report.check_condition_number(inv_cov, "LGS", f"inv_cov_leaf{leaf_idx}")
                    
                    # Noise variance
                    noise_var = model.get("noise_var")
                    if noise_var is not None:
                        report.log_stat("LGS_noise_var", noise_var)
                        if noise_var < 0:
                            report.log_issue("LGS", "NegativeVariance", {"leaf": leaf_idx, "var": noise_var})
            
            # 3. Check coherence tracker
            if opt._coherence_tracker is not None:
                cache = opt._coherence_tracker._cache
                if cache is not None:
                    # Potentials
                    for leaf_id, potential in cache.potentials.items():
                        report.log_stat("Coherence_potential", potential)
                        if np.isnan(potential):
                            report.log_issue("Coherence", "NaN_potential", {"leaf": leaf_id, "iter": iteration})
                        if np.isinf(potential):
                            report.log_issue("Coherence", "Inf_potential", {"leaf": leaf_id, "iter": iteration})
                    
                    # Scores
                    for leaf_id, score in cache.scores.items():
                        report.log_stat("Coherence_score", score)
                        if score < 0 or score > 1:
                            report.log_issue("Coherence", "ScoreOutOfRange", {"leaf": leaf_id, "score": score})
            
            # 4. Check X_all and y_all
            if len(opt.X_all) > 0:
                X_arr = np.array(opt.X_all)
                y_arr = np.array(opt.y_all)
                report.has_nan(X_arr, "History", "X_all")
                report.has_nan(y_arr, "History", "y_all")
                report.has_inf(y_arr, "History", "y_all")
                report.log_stat("y_all_min", float(np.min(y_arr)))
                report.log_stat("y_all_max", float(np.max(y_arr)))
            
            # 5. Check gamma threshold
            report.log_stat("gamma", opt.gamma)
            
        except Exception as e:
            report.log_issue("Execution", "Exception", {"iter": iteration, "error": str(e), "traceback": traceback.format_exc()})
            break
    
    return opt.best_y


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def sphere_mixed(config):
    """Simple sphere with mixed types."""
    total = config['x0']**2 + config['x1']**2 + config['x2']**2
    total += config['i0']**2 + config['i1']**2
    cat_penalty = {'a': 0.0, 'b': 5.0, 'c': 10.0}
    total += cat_penalty.get(config['cat'], 15.0)
    return total

def rosenbrock_mixed(config):
    """Rosenbrock - notorious for gradient issues."""
    x = np.array([config['x0'], config['x1'], config['x2'], config['x3']])
    total = sum(100.0*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))
    return total + config['i0']**2

def ill_conditioned(config):
    """Function designed to cause numerical issues."""
    x = np.array([config['x0'], config['x1'], config['x2'], config['x3'], config['x4']])
    # Highly anisotropic: some dims matter 1e6 more than others
    weights = np.array([1e6, 1e4, 1e2, 1.0, 1e-2])
    return float(np.sum(weights * x**2))

def noisy_multimodal(config):
    """Noisy function with many local minima."""
    x = np.array([config['x0'], config['x1'], config['x2']])
    # Rastrigin-like
    base = 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    noise = np.random.normal(0, 1)
    return float(base + noise)

def near_flat(config):
    """Nearly flat function - tests gradient estimation."""
    x = np.array([config['x0'], config['x1'], config['x2']])
    return float(1e-8 * np.sum(x**2))  # Very small gradients


TESTS = [
    {
        "name": "SphereMixed",
        "func": sphere_mixed,
        "param_space": {
            'x0': (-5.0, 5.0), 'x1': (-5.0, 5.0), 'x2': (-5.0, 5.0),
            'i0': (-10, 10, 'int'), 'i1': (-10, 10, 'int'),
            'cat': ['a', 'b', 'c']
        }
    },
    {
        "name": "RosenbrockMixed",
        "func": rosenbrock_mixed,
        "param_space": {
            'x0': (-5.0, 5.0), 'x1': (-5.0, 5.0), 'x2': (-5.0, 5.0), 'x3': (-5.0, 5.0),
            'i0': (-5, 5, 'int')
        }
    },
    {
        "name": "IllConditioned",
        "func": ill_conditioned,
        "param_space": {
            'x0': (-1.0, 1.0), 'x1': (-1.0, 1.0), 'x2': (-1.0, 1.0),
            'x3': (-1.0, 1.0), 'x4': (-1.0, 1.0)
        }
    },
    {
        "name": "NoisyMultimodal",
        "func": noisy_multimodal,
        "param_space": {
            'x0': (-5.12, 5.12), 'x1': (-5.12, 5.12), 'x2': (-5.12, 5.12)
        }
    },
    {
        "name": "NearFlat",
        "func": near_flat,
        "param_space": {
            'x0': (-10.0, 10.0), 'x1': (-10.0, 10.0), 'x2': (-10.0, 10.0)
        }
    },
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("ALBA HYBRID DRILL - NUMERICAL DEBUGGING")
    print("="*80)
    
    BUDGET = 200
    
    all_reports = {}
    
    for test in TESTS:
        print(f"\n>>> Testing: {test['name']}")
        print("-"*60)
        
        report = NumericalHealthReport()
        
        try:
            best_y = run_instrumented_alba(
                test["func"],
                test["param_space"],
                BUDGET,
                report
            )
            print(f"  Best y: {best_y:.6f}")
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()
        
        all_reports[test["name"]] = report
        
        # Quick summary
        if report.issues:
            print(f"  ⚠ Found {len(report.issues)} issues")
        else:
            print(f"  ✓ No issues")
    
    # Full reports
    print("\n\n" + "="*80)
    print("DETAILED REPORTS")
    print("="*80)
    
    for name, report in all_reports.items():
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"{'='*40}")
        report.print_summary()
    
    # Global analysis
    print("\n\n" + "="*80)
    print("GLOBAL ANALYSIS")
    print("="*80)
    
    total_issues = sum(len(r.issues) for r in all_reports.values())
    print(f"\nTotal issues across all tests: {total_issues}")
    
    # Categorize issues
    issue_types = {}
    for name, report in all_reports.items():
        for issue in report.issues:
            key = (issue["component"], issue["type"])
            if key not in issue_types:
                issue_types[key] = []
            issue_types[key].append((name, issue))
    
    if issue_types:
        print("\nIssue breakdown:")
        for (component, issue_type), occurrences in sorted(issue_types.items()):
            print(f"  {component}/{issue_type}: {len(occurrences)} occurrences")
            for test_name, issue in occurrences[:3]:  # Show first 3
                print(f"    - {test_name}: {issue}")


if __name__ == "__main__":
    main()
