# ALBA Framework — Guida rapida all'installazione e all'uso

## Prerequisiti

- **Python ≥ 3.9** (testato con 3.9, 3.10, 3.11, 3.12)
- `pip` (incluso in qualsiasi installazione Python moderna)

> Non servono CUDA, GPU, né dataset esterni: il framework è self-contained.

---

## Opzione 1 — Installazione da archivio (consigliata)

Se hai ricevuto un file **`alba_framework-1.0.0.tar.gz`** oppure **`alba_framework-1.0.0-py3-none-any.whl`**:

```bash
pip install alba_framework-1.0.0-py3-none-any.whl
# oppure
pip install alba_framework-1.0.0.tar.gz
```

Fatto! Tutte le dipendenze (numpy, scipy) vengono installate automaticamente.

---

## Opzione 2 — Installazione dalla cartella sorgente

Se hai la cartella `alba_framework_potential/`:

```bash
cd alba_framework_potential
pip install .
```

Per avere anche `matplotlib` (serve solo per i grafici dell'esempio):

```bash
pip install ".[examples]"
```

---

## Opzione 3 — Senza installazione (solo import diretto)

Se non vuoi installare nulla, basta avere numpy e scipy:

```bash
pip install numpy scipy
```

Poi dal livello superiore alla cartella `alba_framework_potential/`:

```python
import sys
sys.path.insert(0, "/percorso/alla/cartella/che/contiene/alba_framework_potential")

from alba_framework_potential import ALBA
```

---

## Eseguire il demo

```bash
# Dopo l'installazione (Opzione 1 o 2):
python -m alba_framework_potential.examples.quick_demo

# Oppure direttamente:
python examples/quick_demo.py
```

L'output atteso è qualcosa come:

```
==============================================================================
  ALBA Framework — Quick Demo
==============================================================================

  Benchmark             |  Details
  ------------------------------------------------------------------------
  Sphere 5-D            |  dim=5  budget=200  best_y=    0.000123  leaves=  8  time=0.45s
  Rosenbrock 5-D        |  dim=5  budget=200  best_y=    0.234567  leaves= 10  time=0.52s
  ...

  Done!  All benchmarks completed successfully.
==============================================================================
```

---

## Uso rapido nel proprio codice

### Minimizzare una funzione con bounds continui

```python
from alba_framework_potential import ALBA

bounds = [(-5, 5)] * 10  # 10 dimensioni

opt = ALBA(bounds=bounds, maximize=False, seed=42, total_budget=300)
best_x, best_y = opt.optimize(my_objective, budget=300)
print(f"Best: {best_y:.6f}")
```

### Ottimizzare iperparametri (spazio misto continuo + categoriale)

```python
from alba_framework_potential import ALBA

param_space = {
    "learning_rate": (1e-4, 1e-1, "log"),   # continuo, scala log
    "n_layers":      (1, 8, "int"),          # intero
    "hidden_size":   (32, 512, "int"),       # intero
    "activation":    ["relu", "tanh", "gelu"],  # categoriale
    "dropout":       (0.0, 0.5),             # continuo
}

opt = ALBA(param_space=param_space, maximize=False, seed=42, total_budget=200)

for i in range(200):
    config = opt.ask()       # → dict {"learning_rate": 0.003, ...}
    loss = train_model(**config)
    opt.tell(config, loss)

best_config, best_loss = opt.decode(opt.best_x), opt.best_y
print(f"Best config: {best_config}")
print(f"Best loss:   {best_loss:.6f}")
```

### Loop ask/tell (più controllo)

```python
opt = ALBA(bounds=bounds, maximize=False, total_budget=200)

for i in range(200):
    x = opt.ask()         # np.ndarray
    y = my_function(x)
    opt.tell(x, y)

print(f"Best: {opt.best_y:.6f}")
print(f"Stats: {opt.get_statistics()}")
```

---

## Struttura del pacchetto

```
alba_framework_potential/
├── __init__.py          # Esporta ALBA e componenti
├── optimizer.py         # Classe principale ALBA
├── cube.py              # Partizionamento adattivo dello spazio
├── param_space.py       # Gestione spazi misti (log, int, cat)
├── categorical.py       # Sampling categoriale
├── lgs.py               # Local Gradient Surrogate
├── gamma.py             # Soglia dinamica gamma
├── leaf_selection.py    # Selezione foglie (Thompson, UCB, Potential)
├── candidates.py        # Generazione candidati
├── acquisition.py       # Funzione di acquisizione UCB
├── splitting.py         # Politiche di split
├── local_search.py      # Ricerca locale (Gaussian, Covariance)
├── coherence.py         # Coerenza geometrica dei gradienti
├── drilling.py          # Drilling locale
├── lgs_fixed.py         # Variante LGS stabilizzata
├── examples/
│   └── quick_demo.py    # Demo pronta all'uso
└── pyproject.toml       # Metadata per pip install
```

---

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `ModuleNotFoundError: No module named 'alba_framework_potential'` | Assicurati di aver fatto `pip install .` dalla cartella corretta, oppure usa `sys.path.insert()` |
| `ImportError: numpy` | `pip install numpy scipy` |
| Il plot non appare | Installa matplotlib: `pip install matplotlib` |
| Python < 3.9 | Aggiorna Python (3.9+ richiesto) |
