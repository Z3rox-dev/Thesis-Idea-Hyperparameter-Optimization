# Guida JAHS-Bench-201

> **Ambiente richiesto**: Python 3.9 (`/mnt/workspace/miniconda3/envs/py39`)

## Installazione

```bash
/mnt/workspace/miniconda3/envs/py39/bin/pip install jahs-bench
```

**Nota**: L'installazione richiede versioni specifiche di alcune librerie:
- ConfigSpace 0.4.x (non 1.x)
- scikit-learn 1.0.2
- pandas 1.3.5

## Task Disponibili

```python
from jahs_bench.api import BenchmarkTasks

# 3 task disponibili:
# - CIFAR10: 'cifar10'
# - ColorectalHistology: 'colorectal_histology'  
# - FashionMNIST: 'fashion_mnist'
```

## Tipi di Benchmark

| Tipo | Descrizione |
|------|-------------|
| `surrogate` | Usa modello surrogato (XGBoost) per predire le performance - **VELOCE** |
| `tabular` | Query su dataset tabulare pre-computato |
| `live` | Training reale della rete - **LENTO** |

## Spazio di Configurazione (13 dimensioni)

### Iperparametri Continui (log-scale)
| Nome | Range | Default |
|------|-------|---------|
| `LearningRate` | [0.001, 1.0] | 0.1 |
| `WeightDecay` | [1e-5, 0.01] | 0.0005 |

### Iperparametri Ordinali
| Nome | Valori | Descrizione |
|------|--------|-------------|
| `N` | {1, 3, 5} | Profondità rete |
| `W` | {4, 8, 16} | Larghezza canali |
| `Resolution` | {0.25, 0.5, 1.0} | Risoluzione input |

### Iperparametri Categorici
| Nome | Valori |
|------|--------|
| `Activation` | {ReLU, Hardswish, Mish} |
| `TrivialAugment` | {True, False} |
| `Optimizer` | {SGD} (fisso) |

### Operazioni NAS (architettura)
| Nome | Valori | Descrizione |
|------|--------|-------------|
| `Op1` - `Op6` | {0, 1, 2, 3, 4} | Operazioni nei 6 edge del cell |

Operazioni:
- 0: Identity
- 1: 3x3 Conv
- 2: 3x3 Avg Pool
- 3: 3x3 Max Pool
- 4: Zero (skip)

## Uso Base

```python
from jahs_bench import Benchmark

# Inizializza (scarica dati automaticamente alla prima esecuzione)
bench = Benchmark(
    task='cifar10',           # Task da usare
    kind='surrogate',         # Tipo di benchmark
    download=True,            # Scarica se non presente
    save_dir='/mnt/workspace/jahs_bench_data',  # Directory dati
    metrics=['valid-acc']     # Metriche da predire (riduce memoria)
)

# Campiona una configurazione random
config = bench.sample_config()
print(config)
# {'Activation': 'ReLU', 'LearningRate': 0.48, 'N': 1, 'Op1': 1, ...}

# Valuta la configurazione
result = bench(config)
print(result)
# {200: {'valid-acc': 84.1, ...}}
```

## Metriche Disponibili

La query ritorna un dizionario `{epoch: {metrics}}`:

```python
result = bench(config)
# result[200] contiene:
# - 'valid-acc': Validation accuracy (%)
# - 'test-acc': Test accuracy (%)
# - 'train-acc': Training accuracy (%)
# - 'FLOPS': Floating point operations
# - 'latency': Latenza inferenza
# - 'size_MB': Dimensione modello in MB
# - 'runtime': Tempo di training
```

## Esempio Completo: HPO Loop

```python
import numpy as np
from jahs_bench import Benchmark

# Setup
bench = Benchmark(task='cifar10', kind='surrogate', 
                  save_dir='/mnt/workspace/jahs_bench_data')

# Funzione obiettivo (minimizza errore)
def objective(config):
    result = bench(config)
    valid_acc = result[200]['valid-acc']  # Epoch 200
    return 1.0 - valid_acc / 100.0  # Converti in errore

# Random search semplice
best_error = float('inf')
best_config = None

for i in range(100):
    config = bench.sample_config()
    error = objective(config)
    
    if error < best_error:
        best_error = error
        best_config = config
        print(f"Iter {i}: New best error = {error:.4f}")

print(f"\nBest error: {best_error:.4f}")
print(f"Best config: {best_config}")
```

## Accesso al ConfigSpace

```python
import jahs_bench.lib.core.configspace as cs_module

# Ottieni lo spazio di configurazione
config_space = cs_module.joint_config_space

# Lista iperparametri
for hp in config_space.get_hyperparameters():
    print(f"{hp.name}: {type(hp).__name__}")
```

## Note Importanti

1. **Prima esecuzione lenta**: Scarica ~200MB di modelli surrogati
2. **Dati salvati in**: `/mnt/workspace/jahs_bench_data/`
3. **Epoch fisso**: Di default valuta a epoch=200 (training completo)
4. **Surrogato veloce**: ~10-50 query/secondo
5. **Range accuracy**: CIFAR-10 surrogate raggiunge ~92-94% val acc (best)

## Struttura Dati Scaricati

```
/mnt/workspace/jahs_bench_data/
├── assembled_surrogates/
│   ├── cifar10/
│   ├── colorectal_histology/
│   └── fashion_mnist/
└── ...
```

## Benchmark Script

Lo script `/mnt/workspace/thesis/benchmark_jahs.py` confronta 4 ottimizzatori:
- Random Search
- Optuna TPE
- TuRBO
- HPOptimizerV5sFixed (ALBA)

```bash
# Esegui benchmark completo
cd /mnt/workspace/thesis
/mnt/workspace/miniconda3/envs/py39/bin/python benchmark_jahs.py \
    --task cifar10 \
    --n_evals 200 \
    --n_seeds 10

# Output in: /mnt/workspace/thesis/results/jahs_benchmark_*.txt
```

## Risultati Tipici (CIFAR-10, 200 evals, 10 seeds)

| Optimizer | Mean Error | Std |
|-----------|-----------|-----|
| Optuna TPE | 0.0879 | 0.0037 |
| ALBA (v5s Fixed) | 0.0904 | 0.0057 |
| TuRBO | 0.0940 | 0.0106 |
| Random | 0.1024 | 0.0093 |
