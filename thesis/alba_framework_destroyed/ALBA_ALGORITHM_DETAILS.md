# ALBA Framework: Technical Reference

This document provides a comprehensive technical analysis of the **ALBA (Adaptive Linear Bayesian Algorithm)** framework, based on the source code implementation in `thesis/alba_framework/`.

## 1. Architectural Overview

The framework implements a modular **Strategy Pattern** architecture. The core logic is decoupled into interchangeable components, allowing for flexible experimentation with different optimization behaviors.

### Core Components
- **`Optimizer` (`optimizer.py`)**: The central orchestrator (Context). It manages the optimization loop, the tree of cubes, and delegates specific decisions to strategy objects.
- **`Cube` (`cube.py`)**: The fundamental data structure representing a hyper-rectangle in the search space. It maintains local statistics, samples, and a local surrogate model.
- **`LGS` (`lgs.py`)**: The **Local Gradient Surrogate**. A Bayesian Linear Regression model fitted on local data to estimate gradients and predict scores.

### Strategy Protocols
The behavior of the optimizer is defined by the following strategies:
1.  **`GammaScheduler`**: Determines the dynamic threshold $\gamma_t$ for classifying "good" points.
2.  **`LeafSelector`**: Decides which leaf node (Cube) to explore next.
3.  **`CandidateGenerator`**: Generates candidate points within a selected Cube.
4.  **`AcquisitionSelector`**: Selects the best candidate from the generated pool.
5.  **`SplitDecider`**: Determines *when* a Cube should be split.
6.  **`SplitPolicy`**: Determines *how* a Cube is split (axis and cut point).
7.  **`LocalSearchSampler`**: Handles the exploitation phase after the exploration budget is exhausted.

---

## 2. The Algorithm Lifecycle

### Phase 1: Initialization
- The search space is initialized as a single root `Cube`.
- **Categorical Handling**: Categorical dimensions are handled separately. The `Optimizer` uses a `CategoricalSampler` that applies a curiosity-driven selection (prioritizing less-visited categories) before delegating the continuous part to the ALBA logic.

### Phase 2: The Optimization Loop (`ask` / `tell`)

1.  **Gamma Update**: The `GammaScheduler` updates the global threshold $\gamma_t$ based on the history of observed scores.
    - *Implementation*: `QuantileAnnealedGammaScheduler` moves the quantile from $q_{start}=0.5$ to $q_{end}=0.95$ over the budget.

2.  **Leaf Selection**: The `LeafSelector` traverses the tree to find a promising leaf `Cube`.
    - *Implementation*: `UCBSoftmaxLeafSelector`.
    - *Formula*: $S_{cube} = \text{ratio} + C_1 \cdot \sqrt{\frac{\ln N}{n}} + C_2 \cdot (1 - \text{RMSE}_{LGS})$
    - It balances exploitation (good point ratio), exploration (visit counts), and model confidence (low RMSE).

3.  **Candidate Generation**: The `CandidateGenerator` produces a pool of potential points inside the selected Cube.
    - *Implementation*: `MixtureCandidateGenerator`.
    - *Mixture Probabilities*:
        - **25% Top-K**: Gaussian perturbation around the best $k$ points in the cube.
        - **15% Gradient**: Steps in the direction of the LGS gradient ($\nabla f \approx \vec{\beta}$).
        - **15% Center**: Gaussian perturbation around the cube center.
        - **45% Uniform**: Pure random sampling for coverage.

4.  **Acquisition**: The `AcquisitionSelector` picks the single best candidate to evaluate.
    - *Implementation*: `UCBSoftmaxSelector`.
    - It uses the LGS model to predict mean $\mu(x)$ and variance $\sigma^2(x)$ for each candidate, calculating a UCB score: $\mu(x) + \kappa \cdot \sigma(x)$.

5.  **Evaluation**: The user evaluates the point and returns the score.

6.  **Update (Tell)**:
    - The point is added to the Cube.
    - **LGS Update**: The Cube's local model is retrained if enough data is available.
    - **Split Check**: The `SplitDecider` checks if the Cube should be partitioned.
        - *Condition*: `n_trials >= 3 * dim + 6`.

### Phase 3: Splitting (`Cube.split`)

When a split is triggered, the `Cube` determines the split axis and cut point:

1.  **Axis Selection**:
    - **Priority 1**: Gradient Direction. If the LGS gradient is strong ($>0.3$), split along the dimension with the largest coefficient.
    - **Priority 2**: Variance of Good Points. Split along the dimension where high-scoring points are most spread out.
    - **Priority 3**: Widest Dimension (Fallback).

2.  **Cut Point**:
    - Uses a **Weighted Median** of the "good" points (points with score $\ge \gamma_t$).
    - Weights are proportional to `score - gamma`. This centers the new sub-regions around the highest-performing areas.

---

## 3. Deep Dive: Local Gradient Surrogate (LGS)

The LGS (`lgs.py`) is the mathematical engine of ALBA. It is not a standard linear regression but a **Weighted Bayesian Linear Regression**.

### Mathematical Formulation
Given a local dataset $X$ (points) and $y$ (scores):

1.  **Weighting**: Each point $x_i$ is assigned a weight $w_i$:
    $$w_i = w_{\text{rank}} \cdot w_{\text{dist}}$$
    - $w_{\text{dist}} = \exp(-\frac{\|x_i - \mu_{cube}\|^2}{2\sigma^2})$ (Gaussian locality)
    - $w_{\text{rank}}$ boosts points with higher fitness ranks.

2.  **Regression**: We solve for coefficients $\beta$ (gradient approximation):
    $$(X^T W X + \lambda I) \beta = X^T W y$$

3.  **Adaptive Regularization**:
    - The regularization parameter $\lambda$ is not fixed.
    - It scales dynamically based on the **condition number** of the matrix $X^T W X$. If the matrix is ill-conditioned (points are collinear or too few), $\lambda$ increases to stabilize the inversion.

4.  **Bayesian Prediction**:
    - **Mean**: $\mu(x) = x^T \beta$
    - **Variance**: $\sigma^2(x) = x^T (X^T W X + \lambda I)^{-1} x + \sigma_{noise}^2$
    - This variance estimate drives the exploration in the Acquisition step.

### Data Backfilling
If a Cube has too few points ($N < 3 \cdot \text{dim}$) to fit a reliable model, it borrows points from its **parent** Cube to stabilize the regression.

---

## 4. Local Search Phase

When the global iteration count exceeds `exploration_budget` (default 80-90% of total), the `Optimizer` switches to **Local Search Mode**.

- **Strategy**: `GaussianLocalSearchSampler`.
- **Behavior**: It ignores the tree structure entirely.
- **Mechanism**: It samples points from a Gaussian distribution centered strictly on the global `best_x` found so far, with a shrinking standard deviation (fine-tuning).

---

## 5. Default Configuration (V1 Compatibility)

The `ALBA` class in `optimizer.py` is a factory that assembles these components with specific hyperparameters to match the original ALBA V1 performance:

| Component | Parameter | Default Value | Note |
| :--- | :--- | :--- | :--- |
| **Splitting** | Threshold | `3 * dim + 6` | Ensures LGS is solvable |
| **LGS** | Basis | Linear | No quadratic features by default |
| **Candidates** | Top-K Ratio | 0.25 | Exploitation |
| **Candidates** | Gradient Ratio | 0.15 | Gradient-based guidance |
| **Leaf Selection** | Temperature | 3.0 | Softmax smoothing |
| **Gamma** | Quantile Start | 0.50 | Median |
| **Gamma** | Quantile End | 0.95 | Top 5% |

---

## 6. The Optimizer Module (`optimizer.py`)

The `optimizer.py` file is the largest component because it serves multiple critical roles beyond just the main loop. It acts as the **Context** in the Strategy Pattern, but also handles **Data Abstraction**, **State Management**, and **Categorical Logic**.

### Key Responsibilities

1.  **Strategy Composition (Factory Role)**:
    - The `__init__` method acts as a factory. If specific strategy objects (like `LeafSelector` or `SplitDecider`) are not provided, it instantiates the default implementations (e.g., `UCBSoftmaxLeafSelector`, `ThresholdSplitDecider`). This ensures the class is ready to use "out of the box" while remaining fully customizable.

2.  **Parameter Space Abstraction**:
    - It integrates with `ParamSpaceHandler` to support complex parameter definitions (log-scale, integers, categorical strings).
    - It handles the **Encoding/Decoding** of these parameters, converting user-friendly dictionaries into the normalized numerical arrays used internally by the Cubes and LGS models.

3.  **Categorical Variable Handling**:
    - **Separate Logic**: Purely categorical dimensions are often ill-suited for geometric partitioning (Cubes) and gradient estimation (LGS).
    - **`CategoricalSampler`**: The optimizer maintains a separate `CategoricalSampler` instance. Before delegating to the ALBA tree, it first samples the categorical part of the configuration using a curiosity-driven bandit approach (prioritizing less-visited categories).
    - **Hybrid Optimization**: It combines two optimization strategies into a single candidate generation process. For each `ask()` call, the categorical dimensions are selected by the bandit-based `CategoricalSampler`, while the continuous dimensions are selected by the ALBA tree/LGS. These are merged into a single configuration for evaluation, ensuring only one objective function call per iteration.

4.  **Global State & History**:
    - It maintains the global history of all evaluations (`X_all`, `y_all`).
    - It tracks the global `best_x` and `best_y`.
    - It manages the **Stagnation Counter** to detect when the search is stuck, potentially triggering fallback behaviors (though currently primarily used for logging).

5.  **Discrete/Ordinal Logic**:
    - For purely discrete spaces (e.g., integer grids), it maintains specific counters (`_discrete_counts`, `_pair_counts`) to track frequency and success rates of specific values, enabling optimizations for small finite spaces.

6.  **Phase Management**:
    - It explicitly manages the transition from **Exploration Phase** (Tree + LGS) to **Local Search Phase** (Gaussian Perturbation) based on the `local_search_ratio`.

This centralization of state and abstraction allows the core strategies (like `LGS` and `Cube`) to remain mathematically pure and stateless, operating only on normalized numerical data.
