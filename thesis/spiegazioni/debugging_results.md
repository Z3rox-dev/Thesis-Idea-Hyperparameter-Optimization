# Debugging Results: Surrogate R2 Fix

## Problem
The `QuadHPO_Subspace` algorithm was observed to be "stable but conservative", often stagnating at suboptimal values (e.g., 10.1% error) and failing to exploit the quadratic surrogate model.

## Diagnosis
Using a deep instrumentation script (`tests/debug_quadhpo_subspace.py`), we analyzed the internal state of the algorithm over 300 trials.
- **Observation**: The surrogate model was being rejected 100% of the time.
- **Metric**: The $R^2$ score of the surrogate fits was consistently massive and negative (e.g., $-4 \times 10^6$).
- **Root Cause**: A bug in `fit_surrogate` in `hpo_adaptive_backprop_subspace.py`.
    - The code was calculating predicted values $\hat{y} = \Phi w$ using **scaled** features $\Phi$ but **unscaled** weights $w$.
    - This mismatch caused the residuals to be enormous, leading to negative $R^2$.

## Fix
We corrected the prediction logic in `fit_surrogate` to use the **scaled** weights $w_{scaled}$ when multiplying with the scaled design matrix $\Phi$.

```python
# Before
y_hat = Phi @ w  # Mismatch: Phi is scaled, w is unscaled

# After
y_hat = Phi @ w_scaled # Correct: Both are scaled
```

## Verification
After applying the fix, we ran the debug session again (300 trials):
1.  **Surrogate Utilization**: The surrogate is now used in >80% of sampling steps (previously 0%).
2.  **Surrogate Quality**: $R^2$ scores are now excellent (Mean > 0.8, Max > 0.99).
3.  **Performance**:
    - Seed 456 improved from 10.1% error to **9.96% error**.
    - Seed 707 reached **10.2% error**.

## Conclusion
The algorithm is now correctly fitting and using the quadratic surrogate model to guide the search. The "stability" fixes (R2 gating) are now working as intended—filtering out *bad* models rather than *all* models.
