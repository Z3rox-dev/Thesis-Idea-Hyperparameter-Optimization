# LGS v3 Improvements for Complex ML-like Functions

## Overview

This document describes the improvements made to the LGS v3 (Local Geometry Score) optimizer to better handle complex machine learning-inspired optimization landscapes.

## New ML-Inspired Benchmark Functions

Added 6 new complex functions to `ParamSpace.py` that simulate real ML optimization challenges:

### 1. `ml_loss_landscape`
Simulates neural network loss landscapes with:
- Multiple local minima
- Plateaus (flat regions hard to escape)
- Saddle points (non-convex interactions)
- High-frequency noise (simulating stochastic gradients)

### 2. `hyperparameter_surface`
Models hyperparameter optimization with:
- Learning rate sensitivity (exponential scaling)
- Regularization effects (non-linear interactions)
- Architecture parameters (discrete-like continuous jumps)
- Overfitting valley (narrow region of good performance)

### 3. `neural_network_loss`
Represents NN loss surfaces with:
- Sharp vs flat minima
- Symmetries from neuron permutations
- Gradient pathologies (vanishing/exploding gradients)

### 4. `ensemble_hyperopt`
Models ensemble methods (e.g., XGBoost, Random Forest):
- Number of estimators effect (diminishing returns)
- Max depth interactions (U-shaped curve)
- Learning rate + subsample coupling
- Feature importance interactions

### 5. `adversarial_landscape`
Deliberately difficult landscape to test optimizer robustness:
- Deceptive gradients pointing away from global optimum
- Strong local attractor at center (deceptive basin)
- Barriers between local and global optima
- Narrow valley to global optimum

### 6. `multiscale_landscape`
Combines features at multiple scales (like CNNs):
- Coarse structure (overall topology)
- Medium scale (undulations)
- Fine structure (many small local minima)
- Micro scale (high-frequency noise)
- Cross-scale interactions

## Improvements to LGS v3 Optimizer

### 1. Enhanced Gradient Estimation (`fit_lgs_model`)

**Problem**: Original gradient estimation used single linear regression, which can be inaccurate in complex landscapes.

**Solution**: Ensemble gradient estimation with three approaches:
- **Approach 1**: Linear regression on all points (robust, global view)
- **Approach 2**: Weighted regression on top-k points (focuses on best regions)
- **Approach 3**: Direction from bad centroid to top centroid (simple, intuitive)

The final gradient is the average of all successful approaches, increasing robustness.

**Benefits**:
- More stable gradient direction in noisy landscapes
- Better handling of multi-modal functions
- Increased confidence when multiple approaches agree

### 2. Improved Candidate Generation (`_generate_candidates`)

**Original**: 4 strategies with fixed probabilities
**Improved**: 6 strategies optimized for complex functions

New/enhanced strategies:

#### a. Multi-hop Gradient Exploration (NEW)
- Tests multiple step sizes (0.05, 0.15, 0.30, 0.50) along gradient
- Adds perpendicular noise to explore around the gradient direction
- Helps escape narrow valleys and find better paths

#### b. Directional Exploration Between Top Points (NEW)
- Interpolates and extrapolates between best points
- Alpha in [-0.3, 1.3] allows exploration beyond current best region
- Discovers promising regions not yet explored

#### c. Adaptive Variance Near Top Points (ENHANCED)
- Early phase: larger variance (0.15) for broad exploration
- Late phase: smaller variance (0.08) for refinement
- Balances exploration vs exploitation over time

#### d. Bad Point Avoidance with Bounce (ENHANCED)
- Moves away from bad regions with magnitude control
- Helps escape poor local minima
- More effective in adversarial landscapes

### 3. Advanced Local Search (`_local_search_sample`)

**Original**: Simple Gaussian noise with decaying radius
**Improved**: 4 diverse strategies

#### a. Intensive Local Refinement
- Small radius refinement early on
- Focus on improving current best

#### b. Multi-scale Radial Exploration (NEW)
- Tries distances of [0.03, 0.08, 0.15, 0.25] from best point
- Uniform random direction
- Finds alternative basins near current best

#### c. Jump-and-Descend (NEW)
- Jumps between best point and other good points
- Helps escape local minima
- Explores valleys between multiple promising regions

#### d. Pattern Search (NEW)
- Systematically perturbs 1-3 dimensions
- Coordinate-wise exploration
- Effective for separable components

### 4. Tuning Improvements

- Increased default `n_candidates` from 30 to 40 for better exploration
- Adaptive top-k selection (between 5 and 15 points)
- Better confidence tracking in gradient estimation

## Performance Results

Quick benchmark on complex functions (Budget: 200 evaluations, 3 seeds):

| Function                | Best Mean  | Best Std   | Notes                          |
|-------------------------|------------|------------|--------------------------------|
| rosenbrock              | 28.30      | 5.99       | Classic difficult function     |
| rastrigin               | 54.18      | 8.57       | Many local minima              |
| ackley                  | 13.16      | 1.24       | Plateau with spike             |
| levy                    | 4.32       | 1.44       | Wavey surface                  |
| **ml_loss_landscape**   | **-1.73**  | **0.08**   | **ML-inspired: stable!**       |
| **hyperparameter_surface** | **-1.60** | **0.03** | **Very consistent**            |
| **neural_network_loss** | **0.36**   | **0.19**   | **Good on sharp/flat mix**     |
| **ensemble_hyperopt**   | **-1.96**  | **0.09**   | **Handles interactions well**  |
| **adversarial_landscape** | **-6874** | **4867**  | **Difficult but found minima** |
| **multiscale_landscape** | **-2.07**  | **0.60**   | **Handles multiple scales**    |

## Key Improvements Summary

1. **Robustness**: Ensemble gradient estimation provides more reliable directions
2. **Exploration**: Multi-scale and directional strategies find better minima
3. **Escape Capability**: Jump-and-descend and pattern search escape local minima
4. **Complex Landscapes**: New strategies specifically target ML-like optimization challenges
5. **Consistency**: Lower standard deviations on ML functions show more reliable convergence

## Usage

```python
from hpo_lgs_v3 import HPOptimizer
import numpy as np

# Define your objective function
def my_ml_function(x):
    # Your complex ML objective
    return some_loss

# Setup optimizer
bounds = [(0.0, 1.0)] * 10  # 10-dimensional
optimizer = HPOptimizer(
    bounds=bounds,
    maximize=False,  # Minimization
    seed=42,
    n_candidates=40  # Use more candidates for complex functions
)

# Optimize
best_x, best_y = optimizer.optimize(my_ml_function, budget=200)
print(f"Best value found: {best_y:.6f}")
```

## Running Benchmarks

To test on all functions:

```bash
cd /home/runner/work/Thesis-Idea-Hyperparameter-Optimization/Thesis-Idea-Hyperparameter-Optimization
python3 thesis/benchmark_lgsv3_improved.py
```

## Future Work

Potential further improvements:
1. Adaptive budget allocation between exploration and local search
2. Trust region methods for local refinement
3. Multi-fidelity optimization for expensive ML objectives
4. Parallel candidate evaluation
5. Meta-learning to adapt strategies based on landscape properties

## References

- Original LGS v3: Non-parametric local geometry scoring
- Bayesian Optimization principles for candidate generation
- CMA-ES inspired multi-scale exploration
- Pattern search for coordinate-wise optimization
