
import time
import numpy as np
import sys
sys.path.insert(0, "/mnt/workspace")
sys.path.insert(0, "/mnt/workspace/thesis")

from alba_framework_potential.optimizer import ALBA

def sphere(x):
    return np.sum(np.array(x)**2)

def run_test():
    dim = 20
    bounds = [(-5.0, 5.0)] * dim
    budget = 200
    
    print(f"Testing ALBA Variance on Sphere {dim}D (Budget={budget})")
    print("-" * 60)
    print(f"{'Run':<5} {'Seed':<10} {'Best Val':<15} {'Time(s)':<10}")
    print("-" * 60)
    
    vals = []
    
    for i in range(5):
        # Generate a random seed for each run
        seed = np.random.randint(0, 10000)
        
        optimizer = ALBA(
            bounds=bounds,
            maximize=False,
            total_budget=budget,
            use_potential_field=True,
            n_candidates=50, 
            local_search_ratio=0.3,
            seed=seed # EXPLICITLY PASS RANDOM SEED
        )
        
        start_t = time.time()
        optimizer.optimize(sphere, budget)
        end_t = time.time()
        
        val = optimizer.best_y
        vals.append(val)
        
        print(f"{i+1:<5} {seed:<10} {val:<15.4f} {end_t - start_t:<10.2f}")
        
    print("-" * 60)
    mean = np.mean(vals)
    std = np.std(vals)
    print(f"Mean: {mean:.4f}")
    print(f"Std : {std:.4f}")
    
    if std > 0:
        print("\nSUCCESS: Variance detected with variable seeds.")
    else:
        print("\nFAILURE: Variance is still 0.")

if __name__ == "__main__":
    run_test()
