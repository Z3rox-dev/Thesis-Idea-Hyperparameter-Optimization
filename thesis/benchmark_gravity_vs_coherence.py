"""
Benchmark: ALBA Framework (Gravity) vs ALBA Framework (Coherence)

Compares:
- alba_framework_gravity: baseline with gravity/physics extensions
- alba_framework_coherence: new version with geometric coherence gating

Tests on synthetic functions (Sphere, Rosenbrock, Rastrigin, Ackley) across
multiple dimensions and seeds.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Add thesis directory to path
sys.path.insert(0, "/mnt/workspace/thesis")

# Import both frameworks
from alba_framework_gravity import ALBA as ALBA_Gravity
from alba_framework_coherence import ALBA as ALBA_Coherence


# =============================================================================
# Synthetic Test Functions
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function (minimum at origin)."""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function (minimum at [1, 1, ...])."""
    return float(
        sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))
    )


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (highly multimodal)."""
    A = 10
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function (multimodal with global basin)."""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e)


def levy(x: np.ndarray) -> float:
    """Levy function (minimum at [1, 1, ...])."""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)


FUNCTIONS = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-5.0, 5.0)),
    "levy": (levy, (-10.0, 10.0)),
}

KNOWN_OPTIMA = {
    "sphere": 0.0,
    "rosenbrock": 0.0,
    "rastrigin": 0.0,
    "ackley": 0.0,
    "levy": 0.0,
}


# =============================================================================
# Benchmark Utilities
# =============================================================================

def run_single_trial(
    optimizer_class,
    func_name: str,
    dim: int,
    budget: int,
    seed: int,
    **optimizer_kwargs,
) -> Dict[str, Any]:
    """Run a single optimization trial."""
    func, (lb, ub) = FUNCTIONS[func_name]
    bounds = [(lb, ub)] * dim
    
    # Create optimizer
    opt = optimizer_class(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        **optimizer_kwargs,
    )
    
    # Track convergence
    best_so_far = float("inf")
    history = []
    
    start_time = time.time()
    
    for i in range(budget):
        x = opt.ask()
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        y = func(x)
        opt.tell(x, y)
        
        if y < best_so_far:
            best_so_far = y
        history.append(best_so_far)
    
    elapsed = time.time() - start_time
    
    # Get final stats
    stats = opt.get_statistics() if hasattr(opt, "get_statistics") else {}
    
    # Gap to optimum
    optimum = KNOWN_OPTIMA.get(func_name, 0.0)
    gap = best_so_far - optimum
    
    return {
        "best_y": best_so_far,
        "gap": gap,
        "history": history,
        "elapsed": elapsed,
        "n_leaves": stats.get("n_leaves", -1),
        "coherence": stats.get("coherence", None),
    }


def run_benchmark(
    func_names: List[str],
    dims: List[int],
    budget: int,
    seeds: List[int],
    output_path: Path,
):
    """Run full benchmark comparing both frameworks."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "budget": budget,
        "dims": dims,
        "seeds": seeds,
        "functions": func_names,
        "trials": [],
    }
    
    total_trials = len(func_names) * len(dims) * len(seeds) * 2
    trial_idx = 0
    
    for func_name in func_names:
        for dim in dims:
            for seed in seeds:
                # Run Gravity version
                trial_idx += 1
                print(f"[{trial_idx}/{total_trials}] Gravity: {func_name} d={dim} seed={seed}")
                try:
                    gravity_result = run_single_trial(
                        ALBA_Gravity,
                        func_name,
                        dim,
                        budget,
                        seed,
                    )
                    gravity_result["optimizer"] = "gravity"
                    gravity_result["function"] = func_name
                    gravity_result["dim"] = dim
                    gravity_result["seed"] = seed
                    gravity_result["status"] = "ok"
                except Exception as e:
                    gravity_result = {
                        "optimizer": "gravity",
                        "function": func_name,
                        "dim": dim,
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                    }
                    print(f"  ERROR: {e}")
                results["trials"].append(gravity_result)
                
                # Run Coherence version
                trial_idx += 1
                print(f"[{trial_idx}/{total_trials}] Coherence: {func_name} d={dim} seed={seed}")
                try:
                    coherence_result = run_single_trial(
                        ALBA_Coherence,
                        func_name,
                        dim,
                        budget,
                        seed,
                        use_coherence_gating=True,
                    )
                    coherence_result["optimizer"] = "coherence"
                    coherence_result["function"] = func_name
                    coherence_result["dim"] = dim
                    coherence_result["seed"] = seed
                    coherence_result["status"] = "ok"
                except Exception as e:
                    coherence_result = {
                        "optimizer": "coherence",
                        "function": func_name,
                        "dim": dim,
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                    }
                    print(f"  ERROR: {e}")
                results["trials"].append(coherence_result)
                
                # Print comparison
                if gravity_result.get("status") == "ok" and coherence_result.get("status") == "ok":
                    g_gap = gravity_result["gap"]
                    c_gap = coherence_result["gap"]
                    winner = "Coherence" if c_gap < g_gap else "Gravity" if g_gap < c_gap else "Tie"
                    print(f"  Gravity gap: {g_gap:.6f} | Coherence gap: {c_gap:.6f} | Winner: {winner}")
                
                # Save incremental results
                with open(output_path, "w") as f:
                    # Remove history for incremental saves (too large)
                    results_compact = results.copy()
                    for t in results_compact["trials"]:
                        if "history" in t:
                            t["history"] = t["history"][-10:]  # Keep only last 10
                    json.dump(results_compact, f, indent=2)
    
    return results


def summarize_results(results: Dict[str, Any]) -> None:
    """Print summary of benchmark results."""
    trials = [t for t in results["trials"] if t.get("status") == "ok"]
    
    # Group by optimizer
    gravity_trials = [t for t in trials if t["optimizer"] == "gravity"]
    coherence_trials = [t for t in trials if t["optimizer"] == "coherence"]
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Overall stats
    gravity_gaps = [t["gap"] for t in gravity_trials]
    coherence_gaps = [t["gap"] for t in coherence_trials]
    
    print(f"\nGravity:   mean gap = {np.mean(gravity_gaps):.6f}, median = {np.median(gravity_gaps):.6f}")
    print(f"Coherence: mean gap = {np.mean(coherence_gaps):.6f}, median = {np.median(coherence_gaps):.6f}")
    
    # Head-to-head
    wins = {"gravity": 0, "coherence": 0, "tie": 0}
    for g, c in zip(gravity_trials, coherence_trials):
        if abs(g["gap"] - c["gap"]) < 1e-9:
            wins["tie"] += 1
        elif g["gap"] < c["gap"]:
            wins["gravity"] += 1
        else:
            wins["coherence"] += 1
    
    print(f"\nHead-to-head: Gravity {wins['gravity']} | Coherence {wins['coherence']} | Tie {wins['tie']}")
    
    # Per-function breakdown
    print("\nPer-function mean gap:")
    for func_name in results["functions"]:
        g_func = [t["gap"] for t in gravity_trials if t["function"] == func_name]
        c_func = [t["gap"] for t in coherence_trials if t["function"] == func_name]
        if g_func and c_func:
            print(f"  {func_name:12s}: Gravity={np.mean(g_func):.6f}, Coherence={np.mean(c_func):.6f}")
    
    # Per-dimension breakdown
    print("\nPer-dimension mean gap:")
    for dim in results["dims"]:
        g_dim = [t["gap"] for t in gravity_trials if t["dim"] == dim]
        c_dim = [t["gap"] for t in coherence_trials if t["dim"] == dim]
        if g_dim and c_dim:
            print(f"  d={dim:2d}: Gravity={np.mean(g_dim):.6f}, Coherence={np.mean(c_dim):.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Configuration
    FUNC_NAMES = ["sphere", "rosenbrock", "rastrigin", "ackley", "levy"]
    DIMS = [5, 10, 15]
    BUDGET = 200
    SEEDS = [42, 43, 44]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"/mnt/workspace/thesis/benchmark_results/gravity_vs_coherence_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ALBA Framework Benchmark: Gravity vs Coherence")
    print("=" * 70)
    print(f"Functions: {FUNC_NAMES}")
    print(f"Dimensions: {DIMS}")
    print(f"Budget: {BUDGET}")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    results = run_benchmark(
        func_names=FUNC_NAMES,
        dims=DIMS,
        budget=BUDGET,
        seeds=SEEDS,
        output_path=output_path,
    )
    
    summarize_results(results)
    
    print(f"\nResults saved to: {output_path}")
