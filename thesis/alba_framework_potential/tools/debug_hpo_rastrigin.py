
import sys
sys.path.insert(0, "/mnt/workspace/thesis")
import numpy as np
from hpo_minimal import HPOptimizer
from nevergrad import functions as ng_funcs

def debug_hpo_logic():
    dim = 20
    budget = 200
    
    # Function
    ng_func = ng_funcs.ArtificialFunction("rastrigin", block_dimension=dim, rotation=False, translation_factor=0.0)
    
    def func_wrapper(x):
        val = ng_func(x)
        if hasattr(val, "item"):
            val = val.item()
        val = float(val)
        if val < 0:
            print(f"[ALERT] Negative val: {val} at {x}")
        return val

    bounds = [(-5.12, 5.12)] * dim
    
    print("Initializing HPOptimizer(maximize=False)")
    opt = HPOptimizer(bounds=bounds, maximize=False, seed=42)
    print(f"opt.sign: {opt.sign}")
    
    # Override log to see internal workings
    def debug_log(msg):
        if "[BEST]" in msg:
            print(msg)
            
    opt._debug_log = debug_log
    
    print("Optimizing...")
    best_x, best_score = opt.optimize(func_wrapper, budget)
    
    print(f"Finished.")
    print(f"Raw best_score from optimize: {best_score}")
    print(f"Calculated result (-best_score): {-best_score}")
    
    # Verify manually
    real_val = func_wrapper(best_x)
    print(f"Re-eval of best_x: {real_val}")

if __name__ == "__main__":
    debug_hpo_logic()
