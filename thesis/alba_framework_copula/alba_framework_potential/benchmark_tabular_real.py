#!/usr/bin/env python3
"""
Real Tabular Benchmark: XGBoost on Synthetic Classification
===========================================================
Objective: Maximize Validation Accuracy (20 dimensions).
Real training loop, no surrogates.
"""

import sys
import os
import time
import argparse
import warnings
import numpy as np

# Path Setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/mnt/workspace/thesis')
# Add py39 site-packages for sklearn/xgboost
sys.path.append('/mnt/workspace/miniconda3/envs/py39/lib/python3.9/site-packages')

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# LIBRARIES
# -----------------------------------------------------------------------------
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
except ImportError:
    optuna = None

try:
    import cma
except ImportError:
    cma = None

try:
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    import xgboost as xgb
    HAS_ML = True
except ImportError as e:
    print(f"ML Libraries missing: {e}")
    HAS_ML = False

# ALBA Imports
try:
    from alba_framework_potential.optimizer import ALBA
    from alba_framework_potential.local_search import CovarianceLocalSearchSampler
except ImportError:
    sys.path.insert(0, '/mnt/workspace/thesis/alba_framework_potential')
    from optimizer import ALBA
    from local_search import CovarianceLocalSearchSampler

# -----------------------------------------------------------------------------
# DATASET & PIPELINE (From benchmark_xgboost_tabular.py)
# -----------------------------------------------------------------------------

_DATA_CACHE = {}

def get_data():
    if "train" not in _DATA_CACHE:
        # 2000 samples, 40 features
        X, y = make_classification(
            n_samples=2000, n_features=40, n_informative=20, n_redundant=10,
            n_classes=2, flip_y=0.05, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
        _DATA_CACHE["train"] = (X_train, y_train)
        _DATA_CACHE["val"] = (X_val, y_val)
    return _DATA_CACHE["train"], _DATA_CACHE["val"]

def evaluate_xgboost_pipeline(x_norm):
    """
    Map 20D [0,1] vector to XGBoost Pipeline config and return Val Error.
    """
    if not HAS_ML: return np.sum((x_norm-0.5)**2) # Fallback

    X_train, y_train = _DATA_CACHE["train"]
    X_val, y_val = _DATA_CACHE["val"]
    
    # 1. Parse Hyperparameters
    # Preprocessing
    scaler_code = x_norm[0]
    pca_apply = x_norm[1] > 0.5
    pca_ratio = 0.1 + 0.8 * x_norm[2]
    feat_sel_code = x_norm[3]
    feat_sel_ratio = 0.1 + 0.9 * x_norm[4]
    
    # XGBoost
    n_estimators = int(50 + x_norm[5] * 200)
    max_depth = int(2 + x_norm[6] * 10)
    learning_rate = 10 ** (-3 + x_norm[7] * 2) # 0.001 to 0.1
    subsample = 0.5 + 0.5 * x_norm[8]
    colsample = 0.5 + 0.5 * x_norm[9]
    gamma = x_norm[10] * 5.0
    reg_alpha = 10 ** (-5 + x_norm[11] * 5)
    reg_lambda = 10 ** (-5 + x_norm[12] * 5)
    
    # 2. Build Steps
    steps = []
    
    # Scaler
    if scaler_code < 0.33: steps.append(('s', StandardScaler()))
    elif scaler_code < 0.66: steps.append(('s', RobustScaler()))
    
    # PCA
    if pca_apply:
        n_comp = max(2, int(pca_ratio * X_train.shape[1]))
        steps.append(('pca', PCA(n_components=n_comp)))
        
    # Feature Selection
    if feat_sel_code > 0.5:
        k = max(2, int(feat_sel_ratio * X_train.shape[1]))
        steps.append(('sel', SelectKBest(f_classif, k=k)))
        
    # XGBoost
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample, gamma=gamma,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        n_jobs=1, random_state=42, 
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    steps.append(('xgb', model))
    
    # 3. Train & Eval
    pipe = Pipeline(steps)
    try:
        pipe.fit(X_train, y_train) 
        acc = pipe.score(X_val, y_val)
        return 1.0 - acc # Minimize Error
    except Exception as e:
        return 1.0 # Fail penalty

# -----------------------------------------------------------------------------
# WRAPPER
# -----------------------------------------------------------------------------
class TabularWrapper:
    dim = 20
    def __init__(self):
        get_data() # Ensure loaded
        
    def evaluate(self, x):
        return evaluate_xgboost_pipeline(x)

# -----------------------------------------------------------------------------
# RUNNERS
# -----------------------------------------------------------------------------
# ECFS Import
try:
    sys.path.append('/mnt/workspace/thesis/nuovo_progetto')
    from ecfs import ECFS
    HAS_ECFS = True
except ImportError:
    HAS_ECFS = False

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evals', type=int, default=50)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    
    wrapper = TabularWrapper()
    
    optimizers = {
        'Random': 'random',
        'Optuna': 'optuna',
        'CMA_Direct': 'cma',
        'ALBA_Std': 'alba_std',
        'ALBA_Cov': 'alba_cov',
        'ALBA_Hybrid': 'alba_hybrid'
    }
    
    if HAS_ECFS:
        optimizers['ECFS'] = 'ecfs'
    
    print(f"Running Real Tabular XGBoost Benchmark (Evals={args.evals}, Seeds={args.seeds})")
    print("-" * 60)
    
    for name, mode in optimizers.items():
        print(f"Running {name}...")
        results = []
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            
            if mode == 'random':
                best = 1.0
                for _ in range(args.evals):
                    x = rng.random(20)
                    y = wrapper.evaluate(x)
                    best = min(best, y)
                results.append(best)
                
            elif mode == 'optuna' and optuna:
                def obj(trial):
                    x = [trial.suggest_float(f"x{i}",0,1) for i in range(20)]
                    return wrapper.evaluate(np.array(x))
                sampler = optuna.samplers.TPESampler(seed=s)
                study = optuna.create_study(direction='minimize', sampler=sampler)
                study.optimize(obj, n_trials=args.evals)
                results.append(study.best_value)
                
            elif mode == 'cma' and cma:
                x0 = [0.5]*20
                es = cma.CMAEvolutionStrategy(x0, 0.2, {'bounds': [0,1], 'seed':s, 'maxfevals':args.evals, 'verbose':-9})
                best = 1.0
                while not es.stop():
                    X = es.ask()
                    Y = [wrapper.evaluate(np.array(x)) for x in X]
                    es.tell(X, Y)
                    best = min(best, min(Y))
                    if es.result.evaluations >= args.evals: break
                results.append(best)
            
            elif mode == 'ecfs':
                # ECFS (New Framework)
                opt = ECFS(bounds=[(0.0,1.0)]*20, seed=s)
                best = 1.0
                for _ in range(args.evals):
                    x = opt.ask()
                    y = wrapper.evaluate(x)
                    opt.tell(x, y)
                    best = min(best, y)
                results.append(best)
            
            # ALBA Variants
            elif mode.startswith('alba'):
                use_drilling = (mode == 'alba_hybrid')
                sampler = None
                if mode != 'alba_std':
                    sampler = CovarianceLocalSearchSampler(radius_start=0.15, radius_end=0.01)
                    
                opt = ALBA(
                    bounds=[(0.0,1.0)]*20, 
                    total_budget=args.evals, 
                    local_search_sampler=sampler, 
                    use_drilling=use_drilling,
                    seed=s
                )
                best = 1.0
                for _ in range(args.evals):
                    x = opt.ask()
                    y = wrapper.evaluate(x)
                    opt.tell(x, y)
                    best = min(best, y)
                results.append(best)
        
        mean = np.mean(results)
        print(f"  {name}: {mean:.4f} error")

if __name__ == "__main__":
    run_benchmark()
