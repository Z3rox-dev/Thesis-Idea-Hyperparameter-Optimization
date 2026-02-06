from __future__ import annotations

import argparse
import time
import os
from datetime import datetime
import csv
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import random

# Local import of LGS v3 optimizer
from hpo_lgs_v3 import HPOptimizer as LGSv3HPO

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


def format_hparams(fun_name: str, x_norm: np.ndarray) -> str:
    """Return a concise, human-readable string with x_norm and mapped hyperparams.

    Supports both the legacy 3-D search space and the extended 8-D space.
    """
    x_norm = np.asarray(x_norm, dtype=float)
    try:
        x_str = np.array2string(x_norm, precision=3, separator=',')
    except Exception:
        x_str = str(list(map(float, x_norm)))

    if fun_name == 'resnet18_cifar10':
        if len(x_norm) >= 8:
            # Extended 8D space: lr, momentum, weight_decay, batch_size, step_size, gamma, label_smoothing, mixup_alpha
            bounds = [
                (-4, -1),      # log10 lr
                (0.8, 0.99),   # momentum
                (-6, -2),      # log10 weight decay
                (128, 256),    # batch size (min 128 for speed)
                (2, 8),        # StepLR step_size (epochs)
                (0.85, 0.99),  # gamma (StepLR)
                (0.0, 0.2),    # label smoothing
                (0.0, 1.0),    # mixup alpha (0 disables)
            ]
            hp_raw = map_to_domain(x_norm[:8], bounds)
            lr = 10 ** float(hp_raw[0])
            momentum = float(hp_raw[1])
            weight_decay = 10 ** float(hp_raw[2])
            batch_size = int(max(128, min(256, round(float(hp_raw[3]) / 32) * 32)))
            step_size = int(round(float(hp_raw[4])))
            gamma = float(hp_raw[5])
            label_smoothing = float(hp_raw[6])
            mixup_alpha = float(hp_raw[7])
            return (f"x={x_str} | hp: lr={lr:.2e}, mom={momentum:.3f}, wd={weight_decay:.1e}, batch={batch_size}, "
                    f"step={step_size}, gamma={gamma:.3f}, ls={label_smoothing:.3f}, mixup={mixup_alpha:.2f}")
        else:
            bounds = [(-4, -1), (0.8, 0.99), (128, 256)]
            hp_raw = map_to_domain(x_norm[:3], bounds)
            lr = 10 ** float(hp_raw[0])
            momentum = float(hp_raw[1])
            batch_size = int(max(128, min(256, round(float(hp_raw[2]) / 32) * 32)))
            return f"x={x_str} | hp: lr={lr:.2e}, momentum={momentum:.3f}, batch={batch_size}"

    return f"x={x_str}"


# ------------------------------ ML benchmark -------------------------------

# Global dataset cache to avoid re-downloading
_CIFAR10_CACHE = {'train': None, 'val': None, 'test': None}

# Global validation DataLoader cache (batch_size is fixed at 512, so we can reuse it)
_VAL_LOADER_CACHE = {'cpu': None, 'gpu': None}


def _cleanup_cuda():
    """Cleanup CUDA memory and synchronize to prevent memory leaks and errors."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        import gc
        gc.collect()

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

    # Determine whether we're in 3D or 8D mode based on length of x_norm
    extended = len(x_norm) >= 8
    if extended:
        # Extended space: lr, momentum, weight_decay, batch_size, step_size, gamma, label_smoothing, mixup_alpha
        bounds = [
            (-4, -1),      # log10 lr
            (0.8, 0.99),   # momentum
            (-6, -2),      # log10 weight decay
            (128, 256),    # batch size (min 128 for speed)
            (2, 8),        # StepLR step size (epochs)
            (0.85, 0.99),  # gamma
            (0.0, 0.2),    # label smoothing
            (0.0, 1.0),    # mixup alpha
        ]
        hp_raw = map_to_domain(x_norm[:8], bounds)
        hp = {
            'lr': 10 ** hp_raw[0],
            'momentum': hp_raw[1],
            'weight_decay': 10 ** hp_raw[2],
            'batch_size': int(max(128, min(256, round(hp_raw[3] / 32) * 32))),
            'step_size': int(round(hp_raw[4])),
            'gamma': hp_raw[5],
            'label_smoothing': hp_raw[6],
            'mixup_alpha': hp_raw[7],
        }
    else:
        bounds = [
            (-4, -1),       # log10(learning_rate): 10^-4 to 10^-1
            (0.8, 0.99),    # momentum
            (128, 256),     # batch_size (min 128 for speed)
        ]
        hp_raw = map_to_domain(x_norm[:3], bounds)
        hp = {
            'lr': 10 ** hp_raw[0],
            'momentum': hp_raw[1],
            'batch_size': int(max(128, min(256, round(hp_raw[2] / 32) * 32))),
            # Defaults for extended hparams (inactive)
            'weight_decay': 0.0,
            'step_size': 5,
            'gamma': 0.95,
            'label_smoothing': 0.0,
            'mixup_alpha': 0.0,
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
    
    criterion = nn.CrossEntropyLoss(label_smoothing=hp['label_smoothing'])
    optimizer = optim.SGD(
        model.parameters(),
        lr=hp['lr'],
        momentum=hp['momentum'],
        weight_decay=hp['weight_decay'],
        foreach=True
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp['step_size'], gamma=hp['gamma']) if extended else None

    # Mixed Precision Training with AMP
    # DISABLED due to CUDA errors with GradScaler - training without AMP for stability
    use_amp = False  # Disabled: use_gpu and torch.cuda.is_available()
    scaler = None  # Disabled: torch.amp.GradScaler('cuda') if use_amp else None

    # ============== TRAINING LOOP: 7 epochs on train split (reduced for speed) ==============
    model.train()
    for epoch in range(7):  # Reduced to 7 epochs for faster benchmark
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Convert inputs to channels_last if using GPU
            if use_gpu and torch.cuda.is_available():
                inputs = inputs.to(memory_format=torch.channels_last)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Optional mixup augmentation in loss computation
            if extended and hp['mixup_alpha'] > 0.0:
                lam = np.random.beta(hp['mixup_alpha'], hp['mixup_alpha'])
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size, device=device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                targets_a, targets_b = labels, labels[index]

                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(mixed_inputs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    loss.backward()
                    optimizer.step()
            else:
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

        if scheduler is not None:
            scheduler.step()

    # Mixup is applied inline inside training; nothing special to do post-training.

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
    
    # Cleanup CUDA memory after each trial
    del model, optimizer, criterion, trainloader
    if scheduler is not None:
        del scheduler
    if scaler is not None:
        del scaler
    _cleanup_cuda()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def resnet18_cifar10_safe(x_norm: np.ndarray, use_gpu: bool, trial_seed: int = 97) -> Dict[str, float]:
    """Safe wrapper for resnet18_cifar10 with retry logic for CUDA errors."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return resnet18_cifar10(x_norm, use_gpu, trial_seed)
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'out of memory' in str(e).lower():
                print(f"CUDA error on attempt {attempt + 1}/{max_retries}: {e}")
                _cleanup_cuda()
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)  # Wait before retry
                    continue
            raise
    # If all retries failed, return default metrics
    return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


ML_FUNS: Dict[str, Tuple[Callable[[np.ndarray, bool, int], Dict[str, float]], int]] = {
    # Default now set to 8-D (extended). Can be overridden via --hp-dim.
    # Using safe wrapper with retry logic for CUDA error handling
    'resnet18_cifar10': (resnet18_cifar10_safe, 8),
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
    
    extended = len(x_norm) >= 8
    if extended:
        bounds = [
            (-4, -1),      # log10 lr
            (0.8, 0.99),   # momentum
            (-6, -2),      # log10 weight decay
            (128, 256),    # batch size (min 128 for speed)
            (2, 8),        # step size
            (0.85, 0.99),  # gamma
            (0.0, 0.2),    # label smoothing
            (0.0, 1.0),    # mixup alpha
        ]
        hp_raw = map_to_domain(x_norm[:8], bounds)
        hp = {
            'lr': 10 ** hp_raw[0],
            'momentum': hp_raw[1],
            'weight_decay': 10 ** hp_raw[2],
            'batch_size': int(max(128, min(256, round(hp_raw[3] / 32) * 32))),
            'step_size': int(round(hp_raw[4])),
            'gamma': hp_raw[5],
            'label_smoothing': hp_raw[6],
            'mixup_alpha': hp_raw[7],
        }
    else:
        bounds = [(-4, -1), (0.8, 0.99), (128, 256)]
        hp_raw = map_to_domain(x_norm[:3], bounds)
        hp = {
            'lr': 10 ** hp_raw[0],
            'momentum': hp_raw[1],
            'batch_size': int(max(128, min(256, round(hp_raw[2] / 32) * 32))),
            'weight_decay': 0.0,
            'step_size': 5,
            'gamma': 0.95,
            'label_smoothing': 0.0,
            'mixup_alpha': 0.0,
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
    
    criterion = nn.CrossEntropyLoss(label_smoothing=hp['label_smoothing'])
    optimizer = optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'], weight_decay=hp['weight_decay'], foreach=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp['step_size'], gamma=hp['gamma']) if extended else None
    # DISABLED AMP due to CUDA errors
    use_amp = False
    scaler = None
    
    model.train()
    for epoch in range(7):  # Reduced to 7 epochs for speed (consistent with HPO trials)
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


        if scheduler is not None:
            scheduler.step()

    # No EMA in new space
    
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
    
    # Cleanup CUDA memory
    del model, optimizer, criterion, trainloader, testloader
    if scheduler is not None:
        del scheduler
    if scaler is not None:
        del scaler
    _cleanup_cuda()
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_on_test_set_safe(x_norm: np.ndarray, use_gpu: bool, seed: int = 99999) -> Dict[str, float]:
    """Safe wrapper for evaluate_on_test_set with retry logic for CUDA errors."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return evaluate_on_test_set(x_norm, use_gpu, seed)
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'out of memory' in str(e).lower():
                print(f"CUDA error in test eval, attempt {attempt + 1}/{max_retries}: {e}")
                _cleanup_cuda()
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
            raise
    return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


# ------------------------------ optimization --------------------------------

def run_optimizer(
    optimizer: str,
    fun_name: str,
    seed: int,
    budget: int,
    use_gpu: bool,
    verbose: bool = False,
    trial_logger: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
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

    if optimizer == 'lgs_v3':
        # Track best metrics
        best_metrics = {'accuracy': -np.inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        best_x_norm: Optional[np.ndarray] = None
        trial_idx_counter = [0]

        def wrapped_objective(x_norm: np.ndarray) -> float:
            """Objective wrapper called by LGS v3 for each trial."""
            nonlocal best_metrics, best_x_norm
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            # Update best
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)

            trial_idx_counter[0] += 1

            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | v3 trial {trial_idx_counter[0]:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)
            # LGS v3 with maximize=True expects higher values = better
            return metrics['accuracy']

        hpo = LGSv3HPO(bounds=bounds, maximize=True, seed=seed)
        hpo.optimize(wrapped_objective, budget=budget)
        return best_metrics, best_x_norm

    elif optimizer == 'optuna':
        if not OPTUNA_AVAILABLE:
            return ({'accuracy': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan')}, None)
        
        best_metrics = {'accuracy': -np.inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        best_x_norm: Optional[np.ndarray] = None
        
        def objective_optuna(trial: optuna.trial.Trial) -> float:
            nonlocal best_metrics, best_x_norm
            x_norm = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)
            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | optuna trial {trial.number:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)
            return metrics['accuracy']
        
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True) #multivariate on off
        study = optuna.create_study(direction='maximize' if maximize else 'minimize', sampler=sampler)
        study.optimize(objective_optuna, n_trials=budget, show_progress_bar=False)
        return best_metrics, best_x_norm

    elif optimizer == 'random':
        rng = np.random.default_rng(seed)
        best_metrics = {'accuracy': -np.inf, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        best_x_norm: Optional[np.ndarray] = None
        
        for i in range(budget):
            x_norm = rng.random(dim)
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            if metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)
            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | rand trial {i+1:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)
        
        return best_metrics, best_x_norm

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
    parser.add_argument('--test-topk', type=int, default=10, help='Evaluate top-K validation winners on the true test set')
    parser.add_argument('--hp-dim', type=int, default=8, choices=[3, 8], help='Hyperparameter search dimensionality (default: 8)')
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

    # Override search dimensionality if requested
    if 'resnet18_cifar10' in names:
        ML_FUNS['resnet18_cifar10'] = (resnet18_cifar10, args.hp_dim)

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
        print_and_log(f'HP dimension: {ML_FUNS.get("resnet18_cifar10", (None, None))[1]}', log_file)
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

        winners: List[Dict[str, Any]] = []  # collect per-run bests
        for name in names:
            for seed in seeds:
                for method in methods:
                    t0 = time.time()
                    method_map = {'v3': 'lgs_v3', 'lgs_v3': 'lgs_v3', 'curv': 'lgs_v3', 'opt': 'optuna', 'optuna': 'optuna', 'rand': 'random', 'random': 'random'}
                    if method not in method_map:
                        continue
                    metrics, best_x = run_optimizer(
                        method_map[method],
                        name,
                        seed,
                        args.budget,
                        args.gpu,
                        args.verbose,
                        trial_logger=lambda s: print_and_log(s, log_file),
                    )
                    elapsed = time.time() - t0

                    results[name][method]['accuracy'].append(metrics['accuracy'])
                    results[name][method]['precision'].append(metrics['precision'])
                    results[name][method]['recall'].append(metrics['recall'])
                    results[name][method]['f1'].append(metrics['f1'])
                    results[name][method]['time'].append(elapsed)

                    winners.append({
                        'function': name,
                        'method': method,
                        'seed': seed,
                        'val_metrics': metrics,
                        'x_norm': best_x,
                        'elapsed': elapsed
                    })

                    if args.verbose:
                        msg = (f"{name:20s} | seed {seed:2d} | {method:8s} | "
                               f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                               f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s")
                        print_and_log(msg, log_file)

        # --- Enhanced Results Table ---
        print_and_log('\n' + '=' * 120, log_file)
        print_and_log('COMPREHENSIVE BENCHMARK RESULTS (VALIDATION SPLIT ONLY) - ResNet-18 on CIFAR-10', log_file)
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

        # --- Per-method Top-5 evaluation on TEST ---
        print_and_log('\n' + '=' * 120, log_file)
        per_method_k = 5
        print_and_log(f'PER-METHOD TOP-{per_method_k} TEST EVALUATION', log_file)
        print_and_log('=' * 120, log_file)

        # Sort winners globally once by validation accuracy
        winners_sorted = sorted(winners, key=lambda w: w['val_metrics']['accuracy'], reverse=True)

        per_method_results: Dict[str, Dict[str, float]] = {}

        for method in methods:
            method_winners = [w for w in winners_sorted if w['method'] == method]
            if not method_winners:
                continue

            method_topk = method_winners[:per_method_k]
            print_and_log(f"\nMethod: {method}", log_file)
            print_and_log('-' * 120, log_file)

            header_pm = f"{'Rank':<5} {'Function':<20} {'Seed':<6} {'ValAcc':<8} | {'TestAcc':<8} {'TestPrec':<9} {'TestRec':<8} {'TestF1':<8}"
            print_and_log(header_pm, log_file)
            print_and_log('-' * len(header_pm), log_file)

            test_accs: List[float] = []
            for idx, win in enumerate(method_topk, start=1):
                x_norm = win['x_norm']
                test_metrics = {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan}
                if x_norm is not None:
                    try:
                        test_metrics = evaluate_on_test_set_safe(x_norm, args.gpu, seed=win['seed'] + 777)
                    except Exception as e:
                        print_and_log(f"Warning: per-method test eval failed for method {method}, rank {idx} ({e})", log_file)

                test_acc = float(test_metrics['accuracy']) if not np.isnan(test_metrics['accuracy']) else np.nan
                test_accs.append(test_acc)

                line_pm = (f"{idx:<5} {win['function']:<20} {win['seed']:<6d} "
                           f"{win['val_metrics']['accuracy']:<8.4f} | "
                           f"{test_metrics['accuracy']:<8.4f} {test_metrics['precision']:<9.4f} "
                           f"{test_metrics['recall']:<8.4f} {test_metrics['f1']:<8.4f}")
                print_and_log(line_pm, log_file)

            test_arr_pm = np.array([x for x in test_accs if not np.isnan(x)], dtype=float)
            if test_arr_pm.size > 0:
                mean_acc = float(np.nanmean(test_arr_pm))
                std_acc = float(np.nanstd(test_arr_pm))
                per_method_results[method] = {'mean_test_acc': mean_acc, 'std_test_acc': std_acc}
                print_and_log(f"Summary {method}: mean test acc over Top-{test_arr_pm.size} = {mean_acc:.4f} ± {std_acc:.4f}", log_file)

        if per_method_results:
            print_and_log('\n' + '#' * 120, log_file)
            print_and_log('PER-METHOD TOP-5 TEST SUMMARY', log_file)
            print_and_log('#' * 120, log_file)
            for method, stats in per_method_results.items():
                msg = (f"{method:10s} | mean test acc (Top-{per_method_k}): "
                       f"{stats['mean_test_acc']:.4f} ± {stats['std_test_acc']:.4f}")
                print_and_log(msg, log_file)

        print_and_log('\n' + '=' * 120, log_file)
        print_and_log(f"Benchmark Configuration: Budget={args.budget} trials/method | Seeds={seeds} | GPU={'Enabled' if args.gpu else 'Disabled'} | TopPerMethod={per_method_k}", log_file)
        print_and_log('=' * 120, log_file)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
