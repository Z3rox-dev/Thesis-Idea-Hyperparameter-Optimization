
import nevergrad as ng
from nevergrad import functions as ng_funcs
import numpy as np

def check_func(name, dim):
    try:
        # Try ArtificialFunction interface
        if hasattr(ng_funcs, "ArtificialFunction"):
            func = ng_funcs.ArtificialFunction(name, block_dimension=dim, rotation=False, translation_factor=0.0)
        else:
            print("ArtificialFunction class not found directly.")
            return

        print(f"Function: {name}")
        print(f"Dimension: {func.dimension}")
        
        # Check Parametrization
        p = func.parametrization
        print(f"Parametrization: {p}")
        
        # Check Bounds
        if isinstance(p, ng.p.Array):
            bounds = p.bounds
            print(f"Bounds: {bounds}")
            # Verify range
            lower, upper = bounds
            print(f"Lower: {lower[0] if hasattr(lower, '__getitem__') else lower}")
            print(f"Upper: {upper[0] if hasattr(upper, '__getitem__') else upper}")
        
        # Evaluate
        val = func(np.zeros(dim))
        print(f"f(0): {val}")
        
    except Exception as e:
        print(f"Error checking {name}: {e}")

if __name__ == "__main__":
    check_func("sphere", 20)
    check_func("rastrigin", 20)
    check_func("ackley", 20) # ArtificialFunction might not have Ackley by name, might need Multiopt
    check_func("rosenbrock", 20)
