
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class DrillingOptimizer:
    """
    A lightweight, stateful optimizer for 'drilling' down into a local minimum.
    Implements a simplified (1+1)-CMA-ES strategy.
    
    Used by ALBA to refine a new best solution iteratively.
    """
    
    dim: int
    mu: np.ndarray             # Current best estimated position
    sigma: float               # Step size
    C: np.ndarray              # Covariance matrix
    pc: np.ndarray             # Evolution path for C
    ps: np.ndarray             # Evolution path for sigma
    
    # Hyperparameters (CMA-ES defaults simplified)
    cc: float = 0.0
    c1: float = 0.0
    c_cov: float = 0.0
    d_sigma: float = 0.0
    target_success: float = 0.2
    
    # State tracking
    best_y: float = float("inf")
    stagnation_counter: int = 0
    max_steps: int = 20 # Initial grant (Adaptive)
    step_cap: int = 200 # Hard limit
    current_step: int = 0
    
    def __init__(self, start_x: np.ndarray, start_y: float, initial_sigma: float = 0.1, bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize the drilling process from a starting point.
        """
        self.dim = len(start_x)
        self.mu = start_x.copy()
        self.best_y = start_y
        self.sigma = max(initial_sigma, 1e-4)
        self.bounds = bounds
        
        # Initialize identity covariance
        self.C = np.eye(self.dim)
        
        # Evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # Set hyperparameters (approximate standard CMA values)
        # N = dim
        N = self.dim
        self.cc = 4.0 / (N + 4.0)
        self.c_cov = 2.0 / (N**2 + 6.0) # Rank-1 update rate
        self.d_sigma = 1.0 + 2.0 * max(0, np.sqrt((N - 1.0) / (N + 1.0)) - 1.0) + self.c_cov
        
        # Damping for CSA
        self.d_sigma = 1.0 + self.d_sigma 
        
        # Expectation of N(0,I) norm (chi_N)
        self.chiN = np.sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N**2))

    def ask(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate the next candidate solution.
        x = mu + sigma * N(0, C)
        """
        self.current_step += 1
        
        # Eigendecomposition if C is not diagonal?
        # For efficiency in high-D (Drilling is usually short-lived),
        # we can do Cholesky locally.
        
        try:
            # Cholesky decomposition L such that L L^T = C
            # Add jitter for stability
            L = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.dim))
            z = rng.normal(0, 1, self.dim)
            y_mutation = np.dot(L, z) 
        except np.linalg.LinAlgError:
            # Fallback to diagonal if C is broken
            y_mutation = rng.normal(0, 1, self.dim)
        
        # Save z for tell() update (CMA needs z or y_mutation)
        self.last_z = z # Not exactly right for CSA, need y_mutation
        self.last_y_mutation = y_mutation
        
        x = self.mu + self.sigma * y_mutation
        
        # Bound handling (clipping)
        if self.bounds:
            x = np.array([np.clip(x[i], self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)])
            
        return x

    def tell(self, x: np.ndarray, y: float) -> bool:
        """
        Update the strategy based on the evaluation result.
        Returns True if drilling should continue, False if we should stop.
        """
        # Determine success
        success = y < self.best_y
        
        if success:
            self.best_y = y
            # Update Evolution Paths?
            # For (1+1)-CMA, updates are usually done only on success.
            # But standard CMA does rank-mu updates.
            # Simplified (1+1)-CMA with 1/5th rule:
            # If success, increase step size. If fail, decrease.
            
            # Simple 1/5th Rule adaptation:
            # self.sigma *= np.exp(1.0/3.0) 
            # self.mu = x
            # self.stagnation_counter = 0
            
            # CSA (Cumulative Step Adapatation) is better but complex for (1+1).
            # Let's stick to a robust 1/5th + Covariance update for (1+1)-CMA.
            
            # Correct (1+1)-CMA-ES update (Hansen et al.)
            # If success:
            alpha_cov = 2.0 / (self.dim**2 + 6) # Learning rate for C
            p_succ = 1.0 # We succeeded
            
            # Update sigma
            # exp(1/d * (p_succ - p_target) / (1 - p_target)) ??
            # Simple rule:
            self.sigma *= 1.2 # Grow step size aggressively on success
            
            # Covariance Update (Rank-1)
            # vector y = (x - mu_old) / sigma
            y_vec = (x - self.mu) / (self.sigma / 1.2) # Undo sigma growth? No, use old sigma
            y_vec = self.last_y_mutation # Reuse stored mutation vector
            
            # C = (1 - c_cov) * C + c_cov * (y * y^T)
            # This aligns C with the successful step
            
            # Efficient rank-1 update without full outer product if dim is large?
            # For 20D, outer product is fine (20x20).
            try:
                self.C = (1 - self.c_cov) * self.C + self.c_cov * np.outer(y_vec, y_vec)
            except Exception:
                pass # Ignore if math blows up
                
            self.mu = x
            self.stagnation_counter = 0
            
            # ADAPTIVE BUDGET: Earn more steps for success!
            if self.max_steps < self.step_cap:
                self.max_steps += 10
            
        else:
            # Failure
            self.sigma *= 0.85 # Shrink step size
            self.stagnation_counter += 1
        
        # Stop conditions
        if self.current_step >= self.max_steps:
            return False
            
        if self.sigma < 1e-7:
            return False
            
        if self.stagnation_counter > 5:
            # Fast fail (5 instead of 20) to save global budget
            return False
            
        return True
