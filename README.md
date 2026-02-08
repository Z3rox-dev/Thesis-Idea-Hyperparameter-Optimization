# ALBA — Adaptive Local Bayesian Algorithm

**ALBA** is a novel black-box optimization algorithm for **Hyperparameter Optimization (HPO)** in mixed continuous/categorical search spaces.

ALBA avoids building a single expensive global surrogate and instead:
- **Partitions** the search space adaptively into local regions (cubes)
- Fits **local gradient surrogates (LGS)** in each region
- Builds a **potential field** from local gradients for global guidance
- Uses **geometric coherence** to gate exploit/explore decisions
- Handles **categorical + continuous** parameters natively

## Installation

**Requirements:** Python >= 3.9

```bash
# 1. Clone the repository
git clone https://github.com/Z3rox-dev/Thesis-Idea-Hyperparameter-Optimization.git
cd Thesis-Idea-Hyperparameter-Optimization

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install ALBA in editable mode (installs all dependencies automatically)
pip install -U pip
pip install -e thesis/alba_dist

# 4. Verify the installation
python -m alba_framework.examples.quick_demo
```

The demo compares ALBA vs Optuna (TPE), Random Search, and CMA-ES on
10 continuous and 3 mixed continuous+categorical benchmarks.

## Usage

```python
from alba_framework import ALBA

# Define a mixed parameter space
param_space = {
    "learning_rate": (1e-4, 1e-1, "log"),
    "n_layers":      (1, 8, "int"),
    "hidden_size":   (32, 512, "int"),
    "activation":    ["relu", "tanh", "gelu"],
    "dropout":       (0.0, 0.5),
}

opt = ALBA(param_space=param_space, maximize=False, seed=42, total_budget=200)

for i in range(200):
    config = opt.ask()          # → dict
    loss = train_model(**config)
    opt.tell(config, loss)

best_config = opt.decode(opt.best_x)
print(f"Best: {best_config}  loss={opt.best_y:.6f}")
```

Or with simple bounds:

```python
from alba_framework import ALBA

opt = ALBA(bounds=[(-5, 5)] * 10, maximize=False, total_budget=300)
best_x, best_y = opt.optimize(objective, budget=300)
```

## Package Structure

```
alba_framework/
 optimizer.py       # Main ALBA class (ask/tell/optimize)
 cube.py            # Adaptive space partitioning
 lgs.py             # Local Gradient Surrogate models
 param_space.py     # Mixed-type parameter handling
 categorical.py     # Categorical sampling strategies
 gamma.py           # Dynamic threshold scheduler
 leaf_selection.py  # Region selection (Thompson, UCB, Potential)
 candidates.py      # Candidate generation
 acquisition.py     # UCB acquisition function
 splitting.py       # Split policies
 local_search.py    # Local refinement (Gaussian, Covariance)
 coherence.py       # Geometric coherence tracking
```
