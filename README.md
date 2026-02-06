# ALBA — Adaptive Local Bayesian Algorithm

**ALBA** is a novel black-box optimization algorithm for **Hyperparameter Optimization (HPO)** in mixed continuous/categorical search spaces.

ALBA avoids building a single expensive global surrogate and instead:
- **Partitions** the search space adaptively into local regions (cubes)
- Fits **local gradient surrogates (LGS)** in each region
- Builds a **potential field** from local gradients for global guidance
- Uses **geometric coherence** to gate exploit/explore decisions
- Handles **categorical + continuous** parameters natively

## Quick Start

### Install

```bash
pip install .
```

> Only requires **Python ≥ 3.9**. Dependencies (numpy, scipy) are installed automatically.

### Run the demo

```bash
python examples/quick_demo.py
```

### Use in your code

```python
from alba_framework_potential import ALBA

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
from alba_framework_potential import ALBA

opt = ALBA(bounds=[(-5, 5)] * 10, maximize=False, total_budget=300)
best_x, best_y = opt.optimize(objective, budget=300)
```

## Package Structure

```
alba_framework_potential/
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
 drilling.py        # Drilling-based local optimization
```

See [INSTALL.md](alba_framework_potential/INSTALL.md) for detailed installation instructions.
