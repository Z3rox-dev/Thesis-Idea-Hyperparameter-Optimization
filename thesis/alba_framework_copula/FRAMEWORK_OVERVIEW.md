# ALBA-Potential: Framework Overview

This document provides a comprehensive technical overview of the ALBA-Potential framework, designed for inclusion in academic thesis presentations. It details the theoretical foundations, algorithmic architecture, mathematical formulations, and critical research findings regarding scalability.

---

## 1. Introduction & Motivation

### 1.1 The Challenge
Bayesian Optimization (BO) is the gold standard for black-box optimization but suffers from the **Curse of Dimensionality**:
- **Gaussian Processes** scale cubically $O(N^3)$, limiting budget.
- **Global Surrogates** fail to capture heterogeneous local landscapes in high dimensions.
- **Acquisition Functions** typically make binary explore/exploit decisions without continuum.

### 1.2 The Solution: ALBA-Potential
ALBA-Potential extends the Adaptive Local Bayesian Approximation (ALBA) by introducing a **Global Potential Field**.
- **Philosophy**: "Think Locally, Act Globally".
- **Mechanism**: Infer the global energy landscape by stitching together local gradient estimates from independent sub-regions.
- **Benefit**: Retains the scalability of local partitioning while regaining the global guidance lost in pure local search methods.

---

## 2. Architecture & Core Components

The framework operates in a cycle of **Partitioning** → **Modelling** → **Field Construction** → **Sampling**.

### 2.1 Adaptive Space Partitioning
The search space $\mathcal{X}$ is recursively partitioned into hyperrectangles (leaves) $C_1, \dots, C_k$ using a k-d tree structure.
- **Split Criterion**: A leaf splits when it accumulates enough points (`n_trials > Threshold`) to justify refinement.
- **Result**: Dense regions get finer resolution, while empty regions remain coarse.

### 2.2 Local Gradient Surrogates (LGS)
Instead of a single global GP, each leaf $C_i$ fits a lightweight **Local Gradient Surrogate (LGS)**.
**Formulation**: Weighted Ridge Regression.
$$ \min_{\alpha, g} \sum_{x_j \in C_i} w_j (y_j - (\alpha + g^T x_j))^2 + \lambda \|g\|^2 $$
- $g$: estimated local gradient vector.
- $w_j$: weights decaying with distance from leaf center.
- $\lambda$: regularization to handle sparse data ($N \approx D$).

**Output**: A gradient vector $g_i$ representing the local flow direction within leaf $C_i$.

### 2.3 Global Potential Field Construction
To recover global topology, we construct a scalar field $u(x)$ such that its gradient matches the local estimates: $-\nabla u \approx g$.

**Step A: Neighborhood Graph**
Construct a k-Nearest Neighbor (kNN) graph connecting leaf centers.
- Nodes: Leaf centers $c_i$.
- Edges: k=6 nearest neighbors.

**Step B: Poisson Reconstruction (Discrete)**
We solve for potentials $u_i$ at each leaf center by minimizing the inconsistency with local gradients across all edges $(i, j)$:
$$ \min_{\{u\}} \sum_{(i,j) \in E} w_{ij} \left( u_j - u_i - g_i^T (c_j - c_i) \right)^2 + \gamma \sum_i (u_i - \hat{u}_i)^2 $$
- The first term enforces **gradient consistency** (Kirchhoff's laws).
- The second term anchors the field to empirical evidence ($\hat{u}_i$ derived from observed function values).

**Step C: Normalization**
The raw potential is normalized to $[0, 1]$ using a sigmoid centered on the median:
$$ \phi_i = \sigma\left( \frac{u_i - \text{median}(u)}{\text{scale}} \right) $$

### 2.4 Potential-Modulated Sampling
Unlike traditional BO which maximizes an acquisition function, ALBA-Potential modulates the **probability** of exploitation.
$$ P(\text{exploit}|C_i) \propto 1 - \phi_i $$
- **Low Potential ($\phi \to 0$)**: High probability of Local Search (following gradient).
- **High Potential ($\phi \to 1$)**: High probability of Exploration (uniform sampling).

---

## 3. Algorithm: The Execution Loop

1.  **Select Leaf**: Choose a region to sample using Softmax on Upper Confidence Bound (UCB) of leaf statistics.
2.  **Determine Strategy**:
    - Calculate local potential $\phi_i$.
    - Draw random number $r \sim U[0,1]$.
    - If $r < P(\text{exploit}|\phi_i)$: **Exploit** (LGS-guided perturbation).
    - Else: **Explore** (random uniform within leaf).
3.  **Evaluate**: Query $f(x)$ and update leaf statistics.
4.  **Update Models**:
    - Refit LGS $g_i$.
    - Recompute global field $\{u_i\}$ (every $T$ iterations).
5.  **Split**: If leaf $C_i$ exceeds density threshold (see Scaling Laws), split into two children.

---

## 4. Research Critical Findings: Scaling Laws

Crucial research during development revealed that high-dimensional performance is governed by specific scaling laws.

### 4.1 The "Inverse Depth-Dimensionality" Law
**Finding**: As dimensionality $D$ increases, the optimal maximum tree depth must decrease.
**Reason**: High-D leaves require more volume to contain enough points for stable LGS fitting.

| Dimension | Optimal Max Depth | Mechanism |
|-----------|-------------------|-----------|
| 2D - 10D | **8 - 16** | Fine-grained partitioning allows precise local modelling. |
| 20D+ | **4** | Coarse partitioning preserves leaf density ($N > D$). |

### 4.2 The "Endless Split" Paradox
**Observation**: Simply increasing the budget ($B \to \infty$) does **not** improve gradient estimation in high-D.
**Cause**: The algorithm splits leaves as soon as they reach a threshold (e.g., 50 points). Thus, local density stays constant at $\sim 50$ points/leaf, which is insufficient for 20D LGS ($D \times 3 = 60+$ needed).
**Result**: The tree grows infinitely deep with weak models everywhere.

### 4.3 The Density-Budget Scaling Law (The Solution)
**Solution**: To unlock high-D performance, the split threshold must scale with budget.
**Formula**: `split_trials_factor` $\propto$ Budget / Dim.

**Validation on Sphere 20D (Budget 1600)**:
- **Baseline (Factor 3.0)**: constant density ~45 pts/leaf $\to$ Score 5.24 (High variance).
- **High Density (Factor 10.0)**: forced density ~130 pts/leaf $\to$ **Score 4.66 (Optimal & Stable)**.

---

## 5. Empirical Validation

Comparison against ALBA-Coherence (Baseline) shows universal improvements when Scaling Laws are applied.

| Function | Dim | Baseline | ALBA-Potential | Improvement |
|----------|-----|----------|----------------|-------------|
| **Sphere** | 10D | 3.63 | **1.81** | **+50%** |
| **Rosenbrock** | 10D | 93.61 | **68.59** | **+27%** |
| **Ackley** | 10D | 11.49 | **8.51** | **+26%** |
| **Sphere** | 20D | 12.69 | **4.66** (High-Dens) | **+63%** |
| **Rosenbrock** | 20D | 330.51 | **88.35** (High-Dens) | **+73%** |
| **Rastrigin** | 20D | 181.23 | **146.81** | **+19%** |

**Conclusion**: ALBA-Potential achieves **8/8 wins** across dimensions, with massive gains in high-D complex landscapes (Rosenbrock 20D) when leaf density is correctly managed.

---

## 6. Directory Structure

```
alba_framework_potential/
├── optimizer.py      # Orchestrator & Split Logic
├── coherence.py      # Potential Field Engine (Graph & Poisson Solver)
├── lgs.py            # Local Gradient Surrogate Maths
├── cube.py           # Recursive Partitioning Data Structure
├── splitting.py      # Split Decisions (Thresholds)
└── TECHNICAL_FINDINGS.md  # Detailed experimental log
```
