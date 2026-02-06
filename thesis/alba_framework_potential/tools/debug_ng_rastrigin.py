
from nevergrad import functions as ng_funcs
import numpy as np

def check_rastrigin():
    dim = 20
    # Creates function same as benchmark
    func = ng_funcs.ArtificialFunction("rastrigin", block_dimension=dim, rotation=False, translation_factor=0.0)
    
    # Check 0 vector
    x0 = np.zeros(dim)
    y0 = func(x0)
    print(f"Rastrigin({dim}D) at x=0: {y0}")
    
    # Check 1 vector
    x1 = np.ones(dim)
    y1 = func(x1)
    print(f"Rastrigin({dim}D) at x=1: {y1}")

    # Check bounds (random)
    bounds = [(-5.12, 5.12)] * dim
    for _ in range(5):
        xr = np.random.uniform(-5.12, 5.12, dim)
        yr = func(xr)
        print(f"Rastrigin({dim}D) at random: {yr}")

if __name__ == "__main__":
    check_rastrigin()
