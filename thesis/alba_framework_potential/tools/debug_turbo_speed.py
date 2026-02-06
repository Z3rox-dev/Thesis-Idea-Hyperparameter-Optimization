
import time
import numpy as np
from turbo import TurboM
import torch

def sphere(x):
    return np.sum(x**2)

class TurboFunc:
    def __init__(self, f):
        self.f = f
        self.n_evals = 0
    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 1:
            self.n_evals += 1
            return self.f(x)
        y = []
        for xi in x:
            y.append(self.f(xi))
            self.n_evals += 1
        return np.array(y).reshape(-1, 1)

def test_turbo(dim=20, budget=50, training_steps=50, dtype="float64"):
    print(f"Testing TuRBO: Dim={dim}, Budget={budget}, Steps={training_steps}, Dtype={dtype}")
    
    f_wrapper = TurboFunc(sphere)
    lb = np.array([-5.0] * dim)
    ub = np.array([5.0] * dim)
    n_init = 2 * dim
    
    start_t = time.time()
    
    turbo_m = TurboM(
        f=f_wrapper,
        lb=lb,
        ub=ub,
        n_init=n_init,
        max_evals=budget,
        n_trust_regions=5,
        batch_size=1,
        verbose=True, # Enable verbose to see progress
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=training_steps,
        min_cuda=1024,
        device="cpu",
        dtype=dtype,
    )
    
    turbo_m.optimize()
    end_t = time.time()
    print(f"Done in {end_t - start_t:.2f}s")

if __name__ == "__main__":
    # Test 1: Default (Slow?)
    # test_turbo(training_steps=50, dtype="float64")
    
    # Limit threads to avoid CPU contention
    torch.set_num_threads(1)
    
    # Test 2: Minimum allowed steps, single trust region to isolate speed
    # n_init=20, budget=30 => 10 iterations
    # We need to modify test_turbo to accept these args or change defaults
    
    f_wrapper = TurboFunc(sphere)
    dim=20
    lb = np.array([-5.0] * dim)
    ub = np.array([5.0] * dim)
    
    start_t = time.time()
    turbo_m = TurboM(
        f=f_wrapper,
        lb=lb,
        ub=ub,
        n_init=20,
        max_evals=110,
        n_trust_regions=5,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=30,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    )
    turbo_m.optimize()
    end_t = time.time()
    print(f"Done in {end_t - start_t:.2f}s")
