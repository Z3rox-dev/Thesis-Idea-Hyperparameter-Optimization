
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ------------------------------- utilities ---------------------------------

def map_to_domain(x_norm: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    x_norm = np.asarray(x_norm, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + x_norm * (hi - lo)


# ------------------------------ ML benchmark -------------------------------

def resnet18_cifar10(x_norm: np.ndarray, use_gpu: bool) -> float:
    """ResNet-18 on CIFAR-10. Returns validation accuracy (maximize)."""
    if not TORCH_AVAILABLE:
        return 0.0

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    bounds = [
        (1e-4, 1e-1),   # learning_rate (log scale)
        (0.8, 0.99),    # momentum
        (16, 128),      # batch_size
    ]
    hp_raw = map_to_domain(x_norm, bounds)
    hp = {
        'lr': 10**(-4 * hp_raw[0]), # Log uniform mapping
        'momentum': hp_raw[1],
        'batch_size': int(hp_raw[2]),
    }

    # CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Use a subset for faster training during HPO
    train_indices = list(range(0, len(trainset), 5)) # 1/5 of training data
    train_subset = Subset(trainset, train_indices)
    trainloader = DataLoader(train_subset, batch_size=hp['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model, loss, optimizer
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'])

    # Training loop (abbreviated for HPO)
    model.train()
    for epoch in range(10):  # Short training for benchmark
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return float(accuracy)


ML_FUNS: Dict[str, Tuple[Callable[[np.ndarray, bool], float], int]] = {
    'resnet18_cifar10': (resnet18_cifar10, 3),
}


# ------------------------------ optimization --------------------------------

def run_optimizer(
    optimizer: str, fun_name: str, seed: int, budget: int, use_gpu: bool
) -> float:
    func, dim = ML_FUNS[fun_name]
    bounds = [(0.0, 1.0)] * dim
    maximize = True

    def objective_wrapper(x_norm: np.ndarray) -> float:
        return func(x_norm, use_gpu)

    if optimizer == 'curvature':
        hpo = QuadHPO(bounds=bounds, maximize=maximize, rng_seed=seed)
        hpo.optimize(lambda x, e=1: objective_wrapper(x), budget=budget)
        return float(hpo.sign * hpo.best_score_global)

    elif optimizer == 'optuna':
        if not OPTUNA_AVAILABLE:
            return float('nan')
        def objective_optuna(trial: optuna.trial.Trial) -> float:
            x_norm = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
            return objective_wrapper(x_norm)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize' if maximize else 'minimize', sampler=sampler)
        study.optimize(objective_optuna, n_trials=budget, show_progress_bar=False)
        return float(study.best_value)

    elif optimizer == 'random':
        rng = np.random.default_rng(seed)
        best_val = -np.inf
        for _ in range(budget):
            x_norm = rng.random(dim)
            val = objective_wrapper(x_norm)
            if val > best_val:
                best_val = val
        return float(best_val)

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
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    names = [n for n in args.functions.split(',') if n in ML_FUNS]
    methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]

    print('=' * 78)
    print('ADVANCED BENCHMARK: Curvature vs Optuna vs Random (Deep Learning)')
    print('=' * 78)
    print(f'Seeds: {seeds}')
    print(f'Budget per method: {args.budget}')
    print(f'GPU requested: {args.gpu}')
    print(f'Functions: {", ".join(names)}')
    print(f'Methods: {", ".join(methods)}')
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

    results: Dict[str, Dict[str, List[float]]] = {name: {m: [] for m in methods} for name in names}

    for name in names:
        for seed in seeds:
            for method in methods:
                t0 = time.time()
                method_map = {'curv': 'curvature', 'opt': 'optuna', 'optuna': 'optuna', 'rand': 'random', 'random': 'random'}
                if method not in method_map: continue

                val = run_optimizer(method_map[method], name, seed, args.budget, args.gpu)
                results[name][method].append(val)
                if args.verbose:
                    print(f"{name:20s} | seed {seed:2d} | {method:8s} | val: {val:.4f} | t: {time.time()-t0:.1f}s")

    # --- Results Table ---
    print('\n' + '=' * 78)
    print('Benchmark Results (mean ± std)')
    print('-' * 78)

    headers = ['Function'] + [f'{m}_mean' for m in methods] + [f'{m}_std' for m in methods]
    print(f"{'Function':<20}" + "".join([f"{h.split('_')[0]:>12}" for h in methods]))
    print(f"{'':<20}" + "".join([f"{'(mean)':>12}" for h in methods]))


    for name in names:
        row = [f"{name:<20}"]
        means = []
        for method in methods:
            data = np.array(results[name][method])
            mean = np.mean(data) if data.size > 0 else np.nan
            means.append(mean)
            std = np.std(data) if data.size > 0 else np.nan
            row.append(f"{mean:12.4f} ± {std:<8.4f}")
        print(" ".join(row))

    print('=' * 78)


if __name__ == '__main__':
    main()
