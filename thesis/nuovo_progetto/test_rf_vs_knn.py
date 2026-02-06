#!/usr/bin/env python3
"""
Quick test: RF vs k-NN prediction on staircase function
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# Staircase function
def staircase(x, n_steps=10):
    x_q = np.floor(x * n_steps) / n_steps
    return np.sum((x_q - 0.5)**2 * (1 + 0.3 * np.arange(len(x))))

# Generate training data
dim = 5
n_train = 30
X_train = np.random.rand(n_train, dim)
y_train = np.array([staircase(x) for x in X_train])

# Generate test data
n_test = 100
X_test = np.random.rand(n_test, dim)
y_test = np.array([staircase(x) for x in X_test])

# RF prediction
rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# k-NN prediction (k=5)
k = 5
y_pred_knn = np.zeros(n_test)
for i, x in enumerate(X_test):
    dists = np.linalg.norm(X_train - x, axis=1)
    knn_idx = np.argsort(dists)[:k]
    weights = 1.0 / (dists[knn_idx] + 0.01)
    weights /= weights.sum()
    y_pred_knn[i] = np.dot(weights, y_train[knn_idx])

# Evaluate
mse_rf = np.mean((y_pred_rf - y_test)**2)
mse_knn = np.mean((y_pred_knn - y_test)**2)

corr_rf = np.corrcoef(y_pred_rf, y_test)[0, 1]
corr_knn = np.corrcoef(y_pred_knn, y_test)[0, 1]

print("="*60)
print("RF vs k-NN Prediction on Staircase Function")
print("="*60)
print(f"\nTraining points: {n_train}")
print(f"Test points: {n_test}")
print(f"Dimensions: {dim}")

print(f"\nRandom Forest:")
print(f"  MSE:         {mse_rf:.4f}")
print(f"  Correlation: {corr_rf:.4f}")

print(f"\nk-NN (k={k}):")
print(f"  MSE:         {mse_knn:.4f}")
print(f"  Correlation: {corr_knn:.4f}")

print(f"\nWinner: {'RF' if mse_rf < mse_knn else 'k-NN'}")

# Test on optimal point
opt_point = np.array([[0.5] * dim])
print(f"\nPrediction at optimal [0.5, 0.5, ...]:")
print(f"  True value: {staircase(opt_point[0]):.4f}")
print(f"  RF pred:    {rf.predict(opt_point)[0]:.4f}")

# kNN for opt point
dists = np.linalg.norm(X_train - opt_point[0], axis=1)
knn_idx = np.argsort(dists)[:k]
weights = 1.0 / (dists[knn_idx] + 0.01)
weights /= weights.sum()
knn_pred = np.dot(weights, y_train[knn_idx])
print(f"  k-NN pred:  {knn_pred:.4f}")
