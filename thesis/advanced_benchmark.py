
from __future__ import annotations

import argparse
import time
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import random

# Local import of curvature optimizer
from hpo_curvature import QuadHPO

# Optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ------------------------------- utilities ---------------------------------

def worker_init_fn(worker_id: int):
    """Initialize DataLoader workers with reproducible seeds.
    
    Ensures that each worker has a unique but deterministic seed derived from
    the generator seed, preventing identical augmentation/sampling across workers.
    """
    # Get the base seed from PyTorch's generator
    seed = torch.initial_seed() % (2**32)
    # Offset by worker_id for uniqueness
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def map_to_domain(x_norm: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    x_norm = np.asarray(x_norm, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + x_norm * (hi - lo)


# ------------------------------ ML benchmark -------------------------------

# Global dataset cache to avoid re-downloading
_CIFAR10_CACHE = {'train': None, 'val': None, 'test': None}

# Global validation DataLoader cache (batch_size is fixed at 512, so we can reuse it)
_VAL_LOADER_CACHE = {'cpu': None, 'gpu': None}

def resnet18_cifar10(x_norm: np.ndarray, use_gpu: bool, trial_seed: int = 97) -> Dict[str, float]:
    """ResNet-18 on CIFAR-10. Returns metrics dict with accuracy, precision, recall, f1.
    
    Args:
        x_norm: Normalized hyperparameters in [0,1]^d
        use_gpu: Whether to use GPU
        trial_seed: Random seed for this trial (for reproducibility)
    """
    if not TORCH_AVAILABLE:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # ============== REPRODUCIBILITY: Set all seeds ==============
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    random.seed(trial_seed)  # Seed Python's random module too
    if torch.cuda.is_available():
        torch.cuda.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)
        # Deterministic behavior (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # GPU optimizations: Disable TF32 for bit-exact reproducibility across machines
    # TF32 provides ~20% speedup but introduces minor numerical differences between GPUs
    # Set to True if you prioritize speed over exact numerical reproducibility
    if use_gpu and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False  # Bit-exact reproducibility
        torch.backends.cudnn.allow_tf32 = False

    bounds = [
        (-4, -1),       # log10(learning_rate): 10^-4 to 10^-1
        (0.8, 0.99),    # momentum
        (512, 1024),    # batch_size - Max safe with 64MB shm + 4-6 workers
    ]
    hp_raw = map_to_domain(x_norm, bounds)
    hp = {
        'lr': 10**hp_raw[0],  # Correct log-uniform mapping: 10^[-4, -1] = [1e-4, 1e-1]
        'momentum': hp_raw[1],
        'batch_size': int(max(512, min(1024, round(hp_raw[2] / 32) * 32))),  # GPU-friendly multiples of 32
    }

    # ============== DATASET: Load once and cache ==============
    global _CIFAR10_CACHE
    
    if _CIFAR10_CACHE['train'] is None:
        # CIFAR-10 normalization: Use standard mean/std computed over training set
        # Mean: [0.4914, 0.4822, 0.4465], Std: [0.2470, 0.2435, 0.2616]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Download once
        full_trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        # ============== CORRECT SPLIT: 45k train / 5k validation ==============
        # Original CIFAR-10 train has 50k images
        # We split into 45k for training and 5k for validation (HPO tuning)
        # Test set (10k) is reserved for final evaluation only
        train_size = 45000
        val_size = 5000
        
        # Fixed split (reproducible)
        rng = np.random.default_rng(12345)  # Fixed seed for dataset split
        indices = rng.permutation(len(full_trainset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        _CIFAR10_CACHE['train'] = Subset(full_trainset, train_indices)
        _CIFAR10_CACHE['val'] = Subset(full_trainset, val_indices)
        _CIFAR10_CACHE['test'] = testset  # Reserved for final evaluation
    
    train_subset = _CIFAR10_CACHE['train']
    val_subset = _CIFAR10_CACHE['val']
    
    # ============== DATALOADERS: Cached datasets ==============
    num_workers = 4 if use_gpu else 2
    
    trainloader = DataLoader(
        train_subset, 
        batch_size=hp['batch_size'], 
        shuffle=True,  # Shuffle for proper training
        num_workers=num_workers,
        pin_memory=use_gpu,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,  # Reproducible workers
        generator=torch.Generator().manual_seed(trial_seed)  # Reproducible shuffling
    )

    # Validation loader: Cache globally since batch_size is fixed at 512
    global _VAL_LOADER_CACHE
    cache_key = 'gpu' if use_gpu else 'cpu'
    
    if _VAL_LOADER_CACHE[cache_key] is None:
        _VAL_LOADER_CACHE[cache_key] = DataLoader(
            val_subset, 
            batch_size=512,  # Large batch for inference
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=use_gpu,
            persistent_workers=False,
            worker_init_fn=worker_init_fn  # Reproducible workers
        )
    
    valloader = _VAL_LOADER_CACHE[cache_key]

    # Model, loss, optimizer
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.to(device)
    
    # Convert to channels_last for better performance on GPU
    if use_gpu and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=hp['lr'], 
        momentum=hp['momentum'], 
        foreach=True  # Fused optimizer for speed
    )

    # Mixed Precision Training with AMP
    use_amp = use_gpu and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ============== TRAINING LOOP: 10 epochs on train split ==============
    model.train()
    for epoch in range(10):  # Full 10 epochs as specified
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Convert inputs to channels_last if using GPU
            if use_gpu and torch.cuda.is_available():
                inputs = inputs.to(memory_format=torch.channels_last)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    # ============== VALIDATION: Evaluate on validation split (NOT test) ==============
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.inference_mode():  # Faster than no_grad()
        for images, labels in valloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Convert to channels_last if using GPU
            if use_gpu and torch.cuda.is_available():
                images = images.to(memory_format=torch.channels_last)
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    
    # Calculate additional metrics
    precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


ML_FUNS: Dict[str, Tuple[Callable[[np.ndarray, bool, int], Dict[str, float]], int]] = {
    'resnet18_cifar10': (resnet18_cifar10, 3),
}


def evaluate_on_test_set(x_norm: np.ndarray, use_gpu: bool, seed: int = 99999) -> Dict[str, float]:
    """Final evaluation on the held-out test set with the best hyperparameters.
    
    This function should ONLY be called once at the end with the best HP configuration
    found during the HPO process. It trains a model from scratch with those HPs and
    evaluates on the true test set.
    """
    if not TORCH_AVAILABLE:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Set all seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Bit-exact reproducibility (disable TF32)
    if use_gpu and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    bounds = [(-4, -1), (0.8, 0.99), (512, 1024)]
    hp_raw = map_to_domain(x_norm, bounds)
    hp = {
        'lr': 10**hp_raw[0],
        'momentum': hp_raw[1],
        'batch_size': int(max(512, min(1024, round(hp_raw[2] / 32) * 32))),  # GPU-friendly multiples of 32
    }
    
    # Use FULL train set (45k) + val set (5k) = 50k for final training
    global _CIFAR10_CACHE
    if _CIFAR10_CACHE['train'] is None:
        # Initialize cache if not done
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        full_trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        train_size = 45000
        val_size = 5000
        rng = np.random.default_rng(12345)
        indices = rng.permutation(len(full_trainset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        _CIFAR10_CACHE['train'] = Subset(full_trainset, train_indices)
        _CIFAR10_CACHE['val'] = Subset(full_trainset, val_indices)
        _CIFAR10_CACHE['test'] = testset
    
    # Combine train + val for final training
    from torch.utils.data import ConcatDataset
    final_trainset = ConcatDataset([_CIFAR10_CACHE['train'], _CIFAR10_CACHE['val']])
    
    trainloader = DataLoader(
        final_trainset,
        batch_size=hp['batch_size'],
        shuffle=True,
        num_workers=4 if use_gpu else 2,
        pin_memory=use_gpu,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    
    testloader = DataLoader(
        _CIFAR10_CACHE['test'],
        batch_size=512,
        shuffle=False,
        num_workers=4 if use_gpu else 2,
        pin_memory=use_gpu,
        worker_init_fn=worker_init_fn
    )
    
    # Train model
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.to(device)
    if use_gpu and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'], foreach=True)
    use_amp = use_gpu and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    model.train()
    for epoch in range(10):
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_gpu and torch.cuda.is_available():
                inputs = inputs.to(memory_format=torch.channels_last)
            
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
    # Evaluate on TRUE test set
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_gpu and torch.cuda.is_available():
                images = images.to(memory_format=torch.channels_last)
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}



# ------------------------------ optimization --------------------------------

def run_optimizer(
    optimizer: str, fun_name: str, seed: int, budget: int, use_gpu: bool, verbose: bool = False
) -> Dict[str, float]:
    func, dim = ML_FUNS[fun_name]
    bounds = [(0.0, 1.0)] * dim
    maximize = True
    
    # Trial counter for reproducible seeds per trial
    trial_counter = {'count': 0}

    def objective_wrapper(x_norm: np.ndarray) -> Dict[str, float]:
        # Generate deterministic seed for this trial
        trial_seed = seed + trial_counter['count'] * 10000
        trial_counter['count'] += 1
        return func(x_norm, use_gpu, trial_seed)

    if optimizer == 'curvature':
        hpo = QuadHPO(bounds=bounds, maximize=maximize, rng_seed=seed)
        
        # Track best metrics
        best_metrics = {'accuracy': -np.inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        def wrapped_objective(x_norm: np.ndarray, epochs: int = 1) -> float:
            """Objective wrapper called by QuadHPO for each trial.

            When `verbose` is True we print a per-trial summary similar to Optuna's trial logs.
            """
            nonlocal best_metrics
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            # Update best
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics.copy()

            # Print per-trial log if requested. QuadHPO increments `trial_id` before calling
            # the objective, so we can read it for a trial index.
            try:
                trial_idx = int(hpo.trial_id)
            except Exception:
                trial_idx = -1

            if verbose:
                # emulate the per-trial line used elsewhere in the script
                print(
                    f"{fun_name:20s} | seed {seed:2d} | curv trial {trial_idx:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s"
                )

            return metrics['accuracy']
        
        hpo.optimize(wrapped_objective, budget=budget)
        return best_metrics

    elif optimizer == 'optuna':
        if not OPTUNA_AVAILABLE:
            return {'accuracy': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')}
        
        best_metrics = {'accuracy': -np.inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        def objective_optuna(trial: optuna.trial.Trial) -> float:
            nonlocal best_metrics
            x_norm = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
            metrics = objective_wrapper(x_norm)
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics.copy()
            return metrics['accuracy']
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize' if maximize else 'minimize', sampler=sampler)
        study.optimize(objective_optuna, n_trials=budget, show_progress_bar=False)
        return best_metrics

    elif optimizer == 'random':
        rng = np.random.default_rng(seed)
        best_metrics = {'accuracy': -np.inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        for _ in range(budget):
            x_norm = rng.random(dim)
            metrics = objective_wrapper(x_norm)
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics.copy()
        
        return best_metrics

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


# ----------------------------------- main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description='Advanced Benchmark: Curvature vs Optuna vs Random')
    parser.add_argument('--budget', type=int, default=20, help='Trials per method per seed')
    parser.add_argument('--seeds', type=str, default='0', help='Comma-separated list of seeds')
    parser.add_argument('--functions', type=str, default='resnet18_cifar10', help='Comma-separated function names')
    parser.add_argument('--methods', type=str, default='curv,optuna,random', help='Which methods to run')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--verbose', action='store_true', help='Print per-seed details')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: tests/benchmark_results_TIMESTAMP.txt)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    names = [n for n in args.functions.split(',') if n in ML_FUNS]
    methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]

    # Setup output file
    if args.output is None:
        os.makedirs('tests', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'tests/benchmark_results_{timestamp}.txt'
    else:
        output_file = args.output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Function to print and log
    def print_and_log(msg: str, file_handle):
        print(msg)
        file_handle.write(msg + '\n')
        file_handle.flush()

    print('=' * 78)
    print('ADVANCED BENCHMARK: Curvature vs Optuna vs Random (Deep Learning)')
    print('=' * 78)
    print(f'Seeds: {seeds}')
    print(f'Budget per method: {args.budget}')
    print(f'GPU requested: {args.gpu}')
    print(f'Functions: {", ".join(names)}')
    print(f'Methods: {", ".join(methods)}')
    print(f'Output file: {output_file}')
    print('=' * 78)

    if args.gpu and not torch.cuda.is_available():
        print('ERROR: GPU requested with --gpu flag, but CUDA is not available. Aborting.')
        return

    if not TORCH_AVAILABLE:
        print('ERROR: PyTorch requested but not available. Please install torch and torchvision.')
        return
    if 'optuna' in methods and not OPTUNA_AVAILABLE:
        print('ERROR: Optuna requested but not available. Install with: pip install optuna')
        return

    # Open output file
    with open(output_file, 'w') as log_file:
        # Write header to file
        print_and_log('=' * 120, log_file)
        print_and_log('ADVANCED BENCHMARK: Curvature vs Optuna vs Random (Deep Learning)', log_file)
        print_and_log('=' * 120, log_file)
        print_and_log(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', log_file)
        print_and_log(f'Seeds: {seeds}', log_file)
        print_and_log(f'Budget per method: {args.budget}', log_file)
        print_and_log(f'GPU requested: {args.gpu}', log_file)
        print_and_log(f'Functions: {", ".join(names)}', log_file)
        print_and_log(f'Methods: {", ".join(methods)}', log_file)
        print_and_log('=' * 120, log_file)
        print_and_log('', log_file)

        results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for name in names:
            results[name] = {}
            for method in methods:
                results[name][method] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'time': []
                }

        for name in names:
            for seed in seeds:
                for method in methods:
                    t0 = time.time()
                    method_map = {'curv': 'curvature', 'opt': 'optuna', 'optuna': 'optuna', 'rand': 'random', 'random': 'random'}
                    if method not in method_map: continue

                    metrics = run_optimizer(method_map[method], name, seed, args.budget, args.gpu, args.verbose)
                    elapsed = time.time() - t0
                    
                    results[name][method]['accuracy'].append(metrics['accuracy'])
                    results[name][method]['precision'].append(metrics['precision'])
                    results[name][method]['recall'].append(metrics['recall'])
                    results[name][method]['f1'].append(metrics['f1'])
                    results[name][method]['time'].append(elapsed)
                    
                    if args.verbose:
                        msg = (f"{name:20s} | seed {seed:2d} | {method:8s} | "
                              f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                              f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s")
                        print_and_log(msg, log_file)

        # --- Enhanced Results Table ---
        print_and_log('\n' + '=' * 120, log_file)
        print_and_log('COMPREHENSIVE BENCHMARK RESULTS - ResNet-18 on CIFAR-10', log_file)
        print_and_log('=' * 120, log_file)
        
        for name in names:
            print_and_log(f'\n{" " * 40}Function: {name}', log_file)
            print_and_log('-' * 120, log_file)
            
            # Header - single row with method names (data will show "mean ± std")
            header_row = f"{'Metric':<15}"
            for method in methods:
                header_row += f" | {method.upper():^35}"
            print_and_log(header_row, log_file)
            print_and_log('-' * 120, log_file)
            
            # Accuracy (with min/max range)
            acc_row = f"{'Accuracy':<15}"
            for method in methods:
                data = np.array(results[name][method]['accuracy'])
                mean_val = np.mean(data) if data.size > 0 else np.nan
                std_val = np.std(data) if data.size > 0 else np.nan
                min_val = np.min(data) if data.size > 0 else np.nan
                max_val = np.max(data) if data.size > 0 else np.nan
                acc_row += f" | {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]"
            print_and_log(acc_row, log_file)
            
            # Precision
            prec_row = f"{'Precision':<15}"
            for method in methods:
                data = np.array(results[name][method]['precision'])
                mean_val = np.mean(data) if data.size > 0 else np.nan
                std_val = np.std(data) if data.size > 0 else np.nan
                prec_row += f" | {mean_val:.4f} ± {std_val:.4f} {' '*17}"
            print_and_log(prec_row, log_file)
            
            # Recall
            rec_row = f"{'Recall':<15}"
            for method in methods:
                data = np.array(results[name][method]['recall'])
                mean_val = np.mean(data) if data.size > 0 else np.nan
                std_val = np.std(data) if data.size > 0 else np.nan
                rec_row += f" | {mean_val:.4f} ± {std_val:.4f} {' '*17}"
            print_and_log(rec_row, log_file)
            
            # F1-Score
            f1_row = f"{'F1-Score':<15}"
            for method in methods:
                data = np.array(results[name][method]['f1'])
                mean_val = np.mean(data) if data.size > 0 else np.nan
                std_val = np.std(data) if data.size > 0 else np.nan
                f1_row += f" | {mean_val:.4f} ± {std_val:.4f} {' '*17}"
            print_and_log(f1_row, log_file)
            
            # Time
            time_row = f"{'Time (s)':<15}"
            for method in methods:
                data = np.array(results[name][method]['time'])
                mean_val = np.mean(data) if data.size > 0 else np.nan
                std_val = np.std(data) if data.size > 0 else np.nan
                time_row += f" | {mean_val:8.1f} ± {std_val:6.1f} {' '*17}"
            print_and_log(time_row, log_file)
            
            print_and_log('-' * 120, log_file)
            
            # Winner analysis
            print_and_log(f"\n{'Performance Summary':<15}", log_file)
            best_method = None
            best_acc = -np.inf
            for method in methods:
                data = np.array(results[name][method]['accuracy'])
                mean_val = np.mean(data) if data.size > 0 else -np.inf
                if mean_val > best_acc:
                    best_acc = mean_val
                    best_method = method
            
            if best_method:
                print_and_log(f"  → Best performing method: {best_method.upper()} (Accuracy: {best_acc:.4f})", log_file)
                
                # Calculate improvements
                for method in methods:
                    if method != best_method:
                        data = np.array(results[name][method]['accuracy'])
                        mean_val = np.mean(data) if data.size > 0 else 0
                        improvement = ((best_acc - mean_val) / mean_val * 100) if mean_val > 0 else 0
                        print_and_log(f"  → {best_method.upper()} vs {method.upper()}: {improvement:+.2f}% improvement", log_file)

        print_and_log('\n' + '=' * 120, log_file)
        print_and_log(f"Benchmark Configuration: Budget={args.budget} trials/method | Seeds={seeds} | GPU={'Enabled' if args.gpu else 'Disabled'}", log_file)
        print_and_log('=' * 120, log_file)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
