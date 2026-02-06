# Come usare il Surrogato in modo Intelligente

## Stato attuale (V4)
- **Split axis**: varianza dei punti buoni (data-driven) ✓
- **Split point**: mediana dei buoni ✓
- **Cube selection**: good_ratio + UCB exploration ✓
- **Sampling**: perturba punti buoni ✓
- **Surrogato**: NON USATO

V4 = 30% wins vs Optuna, ma non sfrutta il surrogato quadratico!

## Idee per usare il Surrogato

### 1. **Sampling guidato dal surrogato**
Invece di campionare random vicino ai buoni, usa il surrogato per:
- Predire dove è il massimo locale nel cubo
- Campionare con probabilità proporzionale a μ(x) + β*σ(x) (UCB)
- Combinare: 50% vicino ai buoni, 50% da UCB del surrogato

### 2. **Early stopping sulle foglie**
Usa il surrogato per decidere se continuare a esplorare un cubo:
- Se max_predicted(cubo) < best_global - margine → potatura
- Evita di sprecare trial in zone sicuramente subottimali

### 3. **Split point ottimale**
Invece della mediana dei buoni, usa il surrogato per:
- Trovare l'ottimo quadratico x* = -b/(2a) per ogni dimensione
- Splittare dove il surrogato predice il massimo gradiente

### 4. **Confidence-aware selection**
Quando selezioni la foglia, considera anche l'incertezza:
- Foglie con alta varianza σ²(x) meritano esplorazione
- Combina good_ratio con uncertainty del surrogato

### 5. **PCA sulle predizioni del surrogato**
- Campiona una griglia nel cubo
- Calcola predizioni μ(x) per ogni punto
- Fai PCA sui punti con alte predizioni
- Questo trova le direzioni "promettenti" secondo il modello

### 6. **Surrogato per l'esplorazione, dati per exploitation**
- **Exploration**: usa surrogato + UCB per esplorare zone incerte
- **Exploitation**: usa varianza buoni per raffinare zone promettenti
- Alterna in base al progresso

## Cosa NON ha funzionato
- Curvatura per split axis (V3, V5): troppo rumorosa
- PCA globale per sampling (V2): limita troppo

## Prossimo esperimento
Iniziamo con l'idea più promettente: **Sampling UCB dal surrogato**
