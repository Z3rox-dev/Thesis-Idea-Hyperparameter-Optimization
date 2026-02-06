
import nevergrad as ng
import numpy as np

def run_debug():
    dim = 5
    budget = 200
    
    # Create function exactly as in benchmark
    from nevergrad import functions as ng_funcs
    ng_func = ng_funcs.ArtificialFunction("sphere", block_dimension=dim, rotation=False, translation_factor=0.0)
    
    def func_wrapper(x):
        val = ng_func(x)
        if hasattr(val, "item"):
            return val.item() 
        return float(val)
        
    print(f"Testing NGOpt on Sphere {dim}D")
    
    parametrization = ng.p.Array(shape=(dim,)).set_bounds(lower=-5.0, upper=5.0)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    
    try:
        rec = optimizer.minimize(func_wrapper)
        print("Success:", rec.value)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Failed:", e)

if __name__ == "__main__":
    run_debug()
