#!/usr/bin/env python3
"""
Mini-benchmark: Coherence vs Optuna con Training REALE su CIFAR-10.

Obiettivo: Verificare che Coherence eccelle su landscape reali e smooth
(non discretizzati da surrogati XGBoost).

Il training reale produce gradienti continui → Coherence dovrebbe sfruttarli.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add parent for ALBA
sys.path.insert(0, str(Path(__file__).parent.parent))
from thesis.ALBA_V1 import ALBA

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Simple CNN for CIFAR-10 (small and fast)
# ============================================================================
class SmallCNN(nn.Module):
    """Small CNN for fast training (~30s per eval on GPU)."""
    
    def __init__(self, hidden_channels=32, dropout=0.3, activation='relu'):
        super().__init__()
        
        act_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
        }[activation]
        
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            act_fn(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# Training function
# ============================================================================
def train_and_eval(config: dict, train_loader, val_loader, device, epochs=5):
    """
    Train a small CNN and return validation accuracy.
    
    Config keys:
    - lr: learning rate [1e-4, 1e-1] log scale
    - weight_decay: [1e-6, 1e-2] log scale
    - dropout: [0.0, 0.5]
    - hidden_channels: [16, 64] int
    - activation: ['relu', 'gelu', 'silu', 'tanh']
    - batch_size: already in loader
    """
    model = SmallCNN(
        hidden_channels=int(config['hidden_channels']),
        dropout=config['dropout'],
        activation=config['activation'],
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    return accuracy


# ============================================================================
# ALBA Coherence objective
# ============================================================================
def create_alba_objective(train_loader, val_loader, device, epochs):
    """Create objective function for ALBA."""
    
    def objective(x: np.ndarray) -> float:
        """
        x is [4] dimensional: [lr_log, wd_log, dropout, hidden_ch_norm]
        We fix activation to 'relu' for simplicity.
        """
        # Decode parameters
        lr = 10 ** (x[0] * 3 - 4)  # [0,1] -> [1e-4, 1e-1]
        weight_decay = 10 ** (x[1] * 4 - 6)  # [0,1] -> [1e-6, 1e-2]
        dropout = x[2] * 0.5  # [0,1] -> [0, 0.5]
        hidden_channels = int(16 + x[3] * 48)  # [0,1] -> [16, 64]
        
        config = {
            'lr': lr,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'hidden_channels': hidden_channels,
            'activation': 'relu',
        }
        
        acc = train_and_eval(config, train_loader, val_loader, device, epochs)
        return acc  # Maximize accuracy
    
    return objective


# ============================================================================
# Optuna objective
# ============================================================================
def create_optuna_objective(train_loader, val_loader, device, epochs):
    """Create objective function for Optuna."""
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        hidden_channels = trial.suggest_int('hidden_channels', 16, 64)
        activation = trial.suggest_categorical('activation', ['relu'])  # Fixed for fair comparison
        
        config = {
            'lr': lr,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'hidden_channels': hidden_channels,
            'activation': activation,
        }
        
        acc = train_and_eval(config, train_loader, val_loader, device, epochs)
        return acc
    
    return objective


# ============================================================================
# Run single comparison
# ============================================================================
def run_comparison(seed: int, budget: int, train_loader, val_loader, device, epochs):
    """Run ALBA vs Optuna with same budget."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    results = {'seed': seed, 'budget': budget}
    
    # --- ALBA Coherence ---
    print(f"    [ALBA] Starting with budget={budget}...", flush=True)
    alba_obj = create_alba_objective(train_loader, val_loader, device, epochs)
    
    # ALBA uses [(low, high), ...] format for bounds
    bounds_alba = [(0.0, 1.0) for _ in range(4)]
    
    alba_opt = ALBA(
        bounds=bounds_alba,
        maximize=True,  # We want to maximize accuracy
        seed=seed,
        total_budget=budget,
    )
    
    t0 = time.time()
    best_x, alba_best = alba_opt.optimize(alba_obj, budget=budget)
    alba_time = time.time() - t0
    
    results['alba_best'] = float(alba_best)
    results['alba_time'] = alba_time
    print(f"    [ALBA] Best={alba_best:.4f} in {alba_time:.1f}s", flush=True)
    
    # --- Optuna TPE ---
    print(f"    [Optuna] Starting with budget={budget}...", flush=True)
    optuna_obj = create_optuna_objective(train_loader, val_loader, device, epochs)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    
    t0 = time.time()
    study.optimize(optuna_obj, n_trials=budget, show_progress_bar=False)
    optuna_time = time.time() - t0
    
    optuna_best = study.best_value
    results['optuna_best'] = optuna_best
    results['optuna_time'] = optuna_time
    print(f"    [Optuna] Best={optuna_best:.4f} in {optuna_time:.1f}s", flush=True)
    
    # Winner
    results['winner'] = 'alba' if alba_best > optuna_best else ('tie' if alba_best == optuna_best else 'optuna')
    
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Real Training: Coherence vs Optuna')
    parser.add_argument('--budget', type=int, default=30, help='Number of evaluations per optimizer')
    parser.add_argument('--seeds', type=str, default='0-2', help='Seed range (e.g., 0-4)')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs per eval')
    parser.add_argument('--subset', type=int, default=5000, help='Subset of training data')
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    print("=" * 70)
    print("REAL TRAINING BENCHMARK: Coherence vs Optuna")
    print("=" * 70)
    print(f"Budget: {args.budget} evaluations")
    print(f"Seeds: {seeds}")
    print(f"Epochs per eval: {args.epochs}")
    print(f"Training subset: {args.subset} samples")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load CIFAR-10
    print("Loading CIFAR-10...", flush=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    train_full = datasets.CIFAR10(
        root='/mnt/workspace/data',
        train=True,
        download=True,
        transform=transform,
    )
    
    val_full = datasets.CIFAR10(
        root='/mnt/workspace/data',
        train=False,
        download=True,
        transform=transform,
    )
    
    # Use subset for speed
    train_subset = Subset(train_full, range(args.subset))
    val_subset = Subset(val_full, range(2000))  # 2000 for validation
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
    print()
    
    # Quick timing estimate
    print("Estimating time per eval...", flush=True)
    test_config = {'lr': 0.01, 'weight_decay': 1e-4, 'dropout': 0.2, 'hidden_channels': 32, 'activation': 'relu'}
    t0 = time.time()
    _ = train_and_eval(test_config, train_loader, val_loader, device, args.epochs)
    time_per_eval = time.time() - t0
    total_evals = args.budget * 2 * len(seeds)  # 2 optimizers
    print(f"Time per eval: {time_per_eval:.1f}s")
    print(f"Estimated total time: {time_per_eval * total_evals / 60:.1f} minutes")
    print()
    
    # Run comparisons
    all_results = []
    alba_wins = 0
    optuna_wins = 0
    
    for i, seed in enumerate(seeds):
        print(f"\n[Seed {seed}] ({i+1}/{len(seeds)})")
        result = run_comparison(seed, args.budget, train_loader, val_loader, device, args.epochs)
        all_results.append(result)
        
        if result['winner'] == 'alba':
            alba_wins += 1
        elif result['winner'] == 'optuna':
            optuna_wins += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    alba_bests = [r['alba_best'] for r in all_results]
    optuna_bests = [r['optuna_best'] for r in all_results]
    
    print(f"ALBA avg: {np.mean(alba_bests):.4f} ± {np.std(alba_bests):.4f}")
    print(f"Optuna avg: {np.mean(optuna_bests):.4f} ± {np.std(optuna_bests):.4f}")
    print()
    print(f"ALBA wins: {alba_wins}/{len(seeds)} ({100*alba_wins/len(seeds):.1f}%)")
    print(f"Optuna wins: {optuna_wins}/{len(seeds)} ({100*optuna_wins/len(seeds):.1f}%)")
    
    # Save results
    os.makedirs('/mnt/workspace/thesis/benchmark_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = f'/mnt/workspace/thesis/benchmark_results/real_training_{timestamp}.json'
    
    with open(outfile, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': all_results,
            'summary': {
                'alba_wins': alba_wins,
                'optuna_wins': optuna_wins,
                'alba_avg': float(np.mean(alba_bests)),
                'optuna_avg': float(np.mean(optuna_bests)),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
