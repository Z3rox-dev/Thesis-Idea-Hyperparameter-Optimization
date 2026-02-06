# Scoperta: Coherence e Discretizzazione del Surrogato

**Data:** 22 Gennaio 2026

## 🎯 Insight Principale

> **"Non è la funzione vera che conta, ma come la 'vediamo' attraverso il surrogato."**

La performance di Coherence NON dipende da:
- Tipo di parametri (continui vs discreti)
- Smoothness della funzione vera sottostante
- Multimodalità della funzione

**Dipende invece da:** come il surrogato "discretizza" il landscape in output.

---

## 📊 Evidenze Sperimentali

### Benchmark 1: HPO Reali (Contraddizione Iniziale)

| Benchmark | % Params Continui | Coherence Winrate |
|-----------|-------------------|-------------------|
| ParamNet | 87.5% | 23.3% ❌ |
| XGBoost Tabular | ~50% | 29.0% |
| JAHS-Bench-201 | ~60% | 46.7% |
| NN Tabular | 0% | **74.2%** ✅ |

**Paradosso:** NN Tabular (0% continui) batte ParamNet (87.5% continui)!

### Benchmark 2: Smooth vs Discretized (Stessa Funzione!)

**SPHERE (dim=10, budget=200, 50 seeds):**

| Discretizzazione | COH Winrate | Delta vs Smooth |
|------------------|-------------|-----------------|
| SMOOTH | **56.0%** | - |
| BINS_10 | 0% | -56% |
| BINS_25 | 0% | -56% |
| BINS_50 | 10% | -46% |
| BINS_100 | 10% | -46% |
| BINS_200 | 40% | -16% |
| BINS_500 | 40% | -16% |
| BINS_1000 | 50% | -6% |
| BINS_2000 | **58.0%** | ≈0% (convergenza!) |

**ROSENBROCK (dim=10, budget=200, 50 seeds):**

| Discretizzazione | COH Winrate | Delta vs Smooth |
|------------------|-------------|-----------------|
| SMOOTH | **52.0%** | - |
| BINS_10 | 10% | -42% |
| BINS_25 | 10% | -42% |
| BINS_50 | 10% | -42% |
| BINS_100 | 10% | -42% |
| BINS_2000 | **4.0%** | -48% (NON converge!) |

---

## 🔬 Spiegazione Meccanicistica

### Perché Coherence fallisce su landscape discretizzati:

```
SMOOTH:                    DISCRETIZED (pochi bins):
                          
    \                          ____
     \                        |    |
      \                       |    |____
       \    ← gradiente       |         |____
        \     continuo        |              |
         \                    ← plateau + cliff
          \.                    (no gradiente locale!)
```

1. **Coherence usa i gradienti locali** per capire la "direzione giusta"
2. Su un plateau, **tutti i punti hanno lo stesso valore** → gradiente = 0
3. Sui cliff (bordi dei gradini), il gradiente è **infinito ma non informativo**
4. Il k-NN graph costruito su plateau ha **similarità coseno undefined/random**

### Soglia di Convergenza

- **SPHERE:** ~200-500 bins per convergere a smooth
- **ROSENBROCK:** >2000 bins non bastano (valle stretta richiede risoluzione enorme)

---

## 💡 Implicazioni Pratiche

### Per JAHS-Bench-201:
- Usa XGBoost con 500 trees come surrogato
- Questo crea un landscape a ~500 "gradini" effettivi
- Spiega il winrate mediocre (46.7%) nonostante parametri continui

### Per HPOBench NN Tabular:
- Potrebbe usare valutazioni dirette o surrogato più smooth
- Spiega l'eccellente winrate (74.2%)

### Raccomandazioni:
1. **Coherence funziona meglio** con surrogati GP o valutazioni dirette
2. **Coherence soffre** con surrogati tree-based (RF, XGBoost, LightGBM)
3. Per benchmark con surrogati tree-based, considerare:
   - Aumentare esplorazione
   - Usare smoothing del surrogato
   - Disabilitare gating su plateau

---

## 🔮 Direzioni Future

1. **GP Wrapper:** Avvolgere surrogato XGBoost con GP per smoothing
2. **Plateau Detection:** Rilevare quando siamo su un plateau e switchare strategia
3. **Local Perturbation:** Aggiungere rumore controllato per "rompere" i plateau

---

---

## 🔬 Verifica Empirica su JAHS (22 Gen 2026)

### Test: Perturbazioni Minime su Parametri Continui

**LearningRate:**
```
LR=0.100000 -> valid-acc=86.1691360474
LR=0.100010 -> valid-acc=86.1691360474  ← IDENTICO (plateau)
LR=0.100100 -> valid-acc=86.1691360474  ← IDENTICO (plateau)
LR=0.101000 -> valid-acc=86.1911010742  ← Finalmente cambia
```

**Resolution (caso estremo):**
```
Res=1.000 -> valid-acc=86.1691360474
Res=0.999 -> valid-acc=77.5862426758  ← CLIFF! (-8.5 punti!)
Res=0.990 -> valid-acc=77.5862426758  ← PLATEAU
Res=0.950 -> valid-acc=77.5862426758  ← PLATEAU
Res=0.900 -> valid-acc=77.5862426758  ← PLATEAU
```

### Conclusione Empirica

L'output XGBoost **NON è smooth**, anche se restituisce float:
- **Plateau ampi** dove δx piccoli → δy = 0
- **Cliff improvvisi** dove δx minimo → δy enorme

---

## 🎯 Training Reale vs Surrogato: Perché Coherence Dovrebbe Eccellere

### Il Training Reale è SMOOTH per Natura

Quando fai training reale di una NN:

```python
# Cambio LR da 0.1 a 0.1001
config1 = {'lr': 0.100}  → train() → val_loss = 0.4523
config2 = {'lr': 0.1001} → train() → val_loss = 0.4521  # Leggermente diverso!
```

La loss function è **continua e differenziabile** rispetto agli iperparametri perché:

1. **Il processo di training** converge a pesi leggermente diversi
2. **La validation loss** riflette queste differenze continue
3. **Non c'è discretizzazione** artificiale

### Visualizzazione

```
SURROGATO (XGBoost):              TRAINING REALE:
                                  
     ____                              \
    |    |____                          \
    |         |____                      \
    |              |                      \.
                                           
   plateau + cliff                    gradiente continuo
   (gradiente = 0 o ∞)                (Coherence può usarlo!)
```

### Perché Coherence Eccelle su Smooth

Coherence calcola la **similarità coseno** tra gradienti locali:

```
Su PLATEAU (surrogato):
  punto A: y=86.17, gradiente ≈ [0, 0, 0, ...]
  punto B: y=86.17, gradiente ≈ [0, 0, 0, ...]
  → cos(0, 0) = undefined! Coherence non sa cosa fare
  
Su SMOOTH (training reale):
  punto A: y=0.4523, gradiente ≈ [-0.02, 0.01, -0.03, ...]
  punto B: y=0.4521, gradiente ≈ [-0.02, 0.01, -0.03, ...]
  → cos(g_A, g_B) ≈ 0.99 → Alta coerenza! → Exploita la direzione
```

### Previsione

| Scenario | Coherence Winrate Atteso |
|----------|--------------------------|
| Surrogato XGBoost (JAHS) | ~45-50% (misurato: 46.7%) |
| Surrogato GP (smooth) | ~60-70% |
| **Training Reale** | **70-80%+** |

---

## File Correlati

- Benchmark script: `thesis/benchmark_coherence_smooth_vs_discretized.py`
- Smoothing strategies: `thesis/benchmark_smoothing_strategies.py`
- Risultati JSON: `thesis/benchmark_results/coherence_smooth_vs_discretized_*.json`
- JAHS surrogato: `jahs_bench/surrogate/model.py` (usa XGBRegressor con n_estimators=500)
