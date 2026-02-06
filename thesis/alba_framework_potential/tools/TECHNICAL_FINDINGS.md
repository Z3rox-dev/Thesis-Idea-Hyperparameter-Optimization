# ALBA Framework - Technical Findings

Documento tecnico con i bug/limitazioni identificati durante il debugging intensivo.

---

## Finding 15: LGS Gradient Explosion Fixed
**Data:** Sessione precedente  
**Stato:** ✅ RISOLTO

Il Local Gradient Surrogate (LGS) poteva produrre gradienti con norma esplosiva
quando i rank weights erano molto sbilanciati. Fix applicato: normalizzazione
a norma unitaria post-fitting.

---

## Finding 16: Drilling Points Lost Fixed  
**Data:** Sessione precedente  
**Stato:** ✅ RISOLTO

I punti generati dal drilling optimizer venivano persi invece di essere
passati al sampler principale. Fix: drilling ora restituisce i punti che
vengono correttamente accumulati.

---

## Finding 17: Drilling Budget Monopoly Fixed
**Data:** Sessione precedente  
**Stato:** ✅ RISOLTO

Il drilling poteva consumare tutto il budget in una singola regione,
bloccando l'esplorazione. Fix: limite max budget per drilling call.

---

## Finding 18: Gradient Misalignment on Multimodal Functions
**Data:** Questa sessione  
**Stato:** ⚠️ LIMITAZIONE FONDAMENTALE (non bug)

Su Rastrigin (multimodale), 4/9 foglie hanno gradienti che puntano nella 
direzione OPPOSTA all'ottimo globale. Questo è intrinseco al comportamento
di LGS su funzioni con molti minimi locali:

```
Leaf gradient alignments vs global optimum:
  Leaf 0: alignment = +0.85 (✓ correct)
  Leaf 1: alignment = -0.72 (✗ opposite!)
  Leaf 2: alignment = +0.91 (✓ correct)
  Leaf 3: alignment = -0.65 (✗ opposite!)
  ...
```

**Implicazione:** ALBA può bloccarsi in minimi locali su funzioni multimodali
perché il gradiente locale è fuorviante.

---

## Finding 19: 10 Implicit Assumptions Verified
**Data:** Questa sessione  
**Stato:** ✅ TUTTI PASSATI

Test sistematico su 10 assunzioni implicite del framework:

| ID | Assunzione | Risultato |
|----|------------|-----------|
| A1 | Normalizzazione punti in [0,1]^d | ✓ Pass |
| A2 | Covarianza PD o semi-PD | ✓ Pass |
| A3 | Gradiente normalizzato norma 1 | ✓ Pass |
| A4 | UCB score finito | ✓ Pass |
| A5 | Samples dentro bounds cube | ✓ Pass |
| A6 | Rank weights in [0,1], somma ~1 | ✓ Pass |
| A7 | Gamma in [0,1] | ✓ Pass |
| A8 | Leaf count ≤ max_leaves | ✓ Pass |
| A9 | History points memorizzati | ✓ Pass |
| A10 | Coherence in [0,1] | ✓ Pass |

---

## Finding 20: ~~Coherence Ignora Varianza~~ → Coherence È Corretta!
**Data:** Questa sessione  
**Stato:** ✅ ANALISI RIVISTA - Nessun bug

### Analisi Iniziale (SBAGLIATA)

Inizialmente pensavo che la coherence fosse rotta perché:
- Usava solo la media degli alignment, ignorando la varianza
- Non distingueva Sphere da Rastrigin

### Correzione Dopo Discussione

L'utente ha sollevato un punto fondamentale:

> "Se ho un gradiente che nel figlio destra è negativo mentre in quello 
> sinistro è positivo, vuol dire che tra le due foglie c'è una valle, 
> quindi è coerente che quello di sinistra scenda e quello di destra salga?"

**ESATTO!** Gradienti opposti tra foglie vicine indicano una **VALLE** (minimo):

```
    Foglia L        Foglia R
       →               ←
       grad            grad
         \            /
          \  VALLE   /
           \_______/
            minimo
```

L'alignment = -1 (opposti) non significa "incoerenza", significa 
**CONVERGENZA verso un minimo comune**!

### Semantica Corretta

| Alignment | Significato |
|-----------|-------------|
| +1 | Gradienti paralleli → funzione monotona, minimo "oltre" |
| -1 | Gradienti opposti → **VALLE tra le foglie!** (buono) |
| 0 | Ortogonali → nessuna relazione chiara |

### Perché la Fix Proposta Era Sbagliata

Penalizzare la varianza avrebbe punito situazioni dove:
- Alcune coppie convergono (alignment -1) 
- Altre sono parallele (alignment +1)

Questa è una situazione **normale** su funzioni con struttura, non un bug!

### Vera Diagnosi del Problema Rastrigin

Il problema su Rastrigin **non è la coherence**, ma:
1. LGS trova correttamente gradienti verso **minimi locali**
2. Ma non c'è modo di sapere quale minimo è **globale**
3. ALBA si blocca in uno dei tanti minimi locali

### Conclusione

- ✅ La coherence funziona come progettata
- ✅ La fix proposta (penalizzare varianza) era **sbagliata** e avrebbe peggiorato
- ⚠️ Il limite su Rastrigin è **intrinseco** (tanti minimi locali)
- 💡 Soluzione: più esplorazione globale o multi-start, non modifiche a coherence

---

## Strumenti di Debugging Creati

| Tool | Scopo |
|------|-------|
| `tools/assumption_checker.py` | Verifica 10 assunzioni implicite |
| `tools/deep_trace.py` | Trace step-by-step di LGS fit |
| `tools/e2e_trace.py` | End-to-end verification |
| `tools/counter_examples.py` | 7 edge case per stress test |
| `tools/deep_investigation.py` | Analisi gradient alignment |
| `tools/coherence_bug_analysis.py` | Analisi bug coherence varianza |
| `tools/coherence_semantics_analysis.py` | Analisi semantica alignment |
| `tools/rastrigin_diagnosis.py` | Diagnosi comportamento su Rastrigin |
| `tools/rastrigin_deep_diagnosis.py` | Diagnosi approfondita |
| `tools/lgs_vs_true_gradient.py` | Confronto LGS vs gradiente analitico |

---

## Finding 21: LGS Stima Gradiente OPPOSTO su Rastrigin (Alcune Regioni)
**Data:** Questa sessione  
**Stato:** ⚠️ LIMITE STRUTTURALE IDENTIFICATO

### Scoperta

LGS stima il gradiente **opposto** a quello vero in alcune posizioni di Rastrigin:

| Posizione | Alignment LGS vs Vero |
|-----------|----------------------|
| (0.1, 0.1) | **+0.98** ✓ |
| (0.5, 0.5) | **-0.67** ✗ OPPOSTO |
| (1.0, 0.0) | **-0.77** ✗ OPPOSTO |
| (0.5, 0.0) | **-0.81** ✗ OPPOSTO |

### Causa

Il problema è la **scala delle oscillazioni**:

```
Rastrigin: f(x) = 10n + Σ(x² - 10·cos(2πx))

A (0.5, 0.0):
  f(0.3, 0.0) = 13.18  (Δf = -7.07 verso origine)
  f(0.5, 0.0) = 20.25  (centro)
  f(0.7, 0.0) = 13.58  (Δf = -6.67 lontano da origine)
```

In un raggio di 0.2:
- Il termine **cos(2πx)** può cambiare di **±20**
- Il termine **x²** cambia solo di **~0.1**

→ **Il rumore del coseno DOMINA il segnale del quadratico!**

### Perché LGS fallisce qui (non è un bug)

1. LGS vede campioni con valori simili (18-25) sparsi casualmente
2. Le differenze sono dominate dalle oscillazioni locali del coseno
3. Il trend globale (x² verso origine) è **invisibile** a scala locale
4. LGS impara dalle oscillazioni → gradiente sbagliato

### Perché funziona a (0.1, 0.1)

Vicino all'origine, il gradiente è **molto forte** (~37 vs ~1 altrove):
- Il segnale quadratico è amplificato vicino al minimo
- Domina le oscillazioni del coseno
- LGS riesce a catturarlo

### Classificazione

**NON È UN BUG**, è un limite fondamentale:
- Rastrigin ha oscillazioni ad alta frequenza
- Un modello locale lineare (LGS) non può catturare oscillazioni
- Il gradiente medio locale può puntare in qualsiasi direzione

### Implicazioni

1. ALBA (e qualsiasi metodo gradient-based) non è adatto a Rastrigin
2. Funzioni con oscillazioni ad alta frequenza richiedono approcci diversi
3. Possibili soluzioni:
   - Più esplorazione globale
   - Modelli non-lineari (GP con kernel appropriato)
   - Multi-start / population-based

---

## Finding 22: NaN in Potentials from Invalid good_ratio
**Data:** Questa sessione  
**Stato:** ✅ RISOLTO

### Problema

Quando `good_ratio()` restituisce `NaN` o `Inf`, i potentials calcolati 
diventavano tutti `NaN`, corrompendo le decisioni di exploitation.

```python
# Bug: nessuna validazione
leaf_good_ratios = np.array([leaves[i].good_ratio() for i in range(n)])
# Se uno è NaN, empirical_bonus diventa NaN, u_combined diventa NaN...
```

### Causa Root

Il calcolo del potential field usa `good_ratio` per ancorare i potenziali.
Se anche un solo valore è `NaN` o `Inf`:
1. `empirical_bonus` contiene NaN
2. `u_combined` diventa tutto NaN
3. `u_anchored` diventa tutto NaN
4. I potentials finali sono tutti NaN

### Fix Applicata

```python
# Sanitize good_ratios
valid_mask = np.isfinite(leaf_good_ratios)
if valid_mask.any():
    median_ratio = float(np.median(leaf_good_ratios[valid_mask]))
    leaf_good_ratios = np.where(valid_mask, leaf_good_ratios, median_ratio)
else:
    leaf_good_ratios = np.full(n, 0.5)
leaf_good_ratios = np.clip(leaf_good_ratios, 0.0, 1.0)
```

### Verifica

- Stress test con NaN, Inf, valori fuori [0,1]: tutti passano
- I potentials restano sempre in [0, 1]

---

## Finding 23: NaN in Candidates from Invalid gradient_dir
**Data:** Questa sessione  
**Stato:** ✅ RISOLTO

### Problema

Quando `gradient_dir` contiene `NaN` o `Inf`, ~13% dei candidati generati
(quelli dalla strategia "gradient") contengono `NaN`.

```python
# Bug: nessuna validazione di gradient_dir
elif strategy < 0.40 and model is not None and model["gradient_dir"] is not None:
    top_center = model["top_k_pts"].mean(axis=0)
    step = float(rng.uniform(self.step_min, self.step_max))
    x = top_center + step * model["gradient_dir"] * widths  # NaN se gradient_dir è NaN!
```

### Problema Aggiuntivo

Quando `top_k_pts` è vuoto ma la condizione `gradient_dir is not None` è vera:
```python
top_center = model["top_k_pts"].mean(axis=0)  # Warning: Mean of empty slice
# top_center diventa [nan, nan] → x diventa NaN
```

### Fix Applicata

```python
elif strategy < 0.40 and model is not None and model.get("gradient_dir") is not None:
    grad_dir = model["gradient_dir"]
    top_k_pts = model.get("top_k_pts", np.array([]))
    # BUG FIX: Skip gradient strategy if gradient contains NaN/Inf or top_k_pts is empty
    if not np.all(np.isfinite(grad_dir)) or len(top_k_pts) == 0:
        # Fallback to center perturbation
        x = center + rng.normal(0, self.sigma_center, dim) * widths
    else:
        top_center = top_k_pts.mean(axis=0)
        # ... resto del codice
```

### Verifica

- Test con NaN/Inf gradient: 0 candidati con NaN (vs 13% prima)
- Test con top_k_pts vuoto: nessun warning, candidati validi

---

## Finding 24: NaN in LGS Gradient from Invalid Scores
**Data:** Questa sessione  
**Stato:** ✅ RISOLTO

### Problema

Quando `all_scores` contiene `NaN` o `Inf`, il gradiente LGS diventa `NaN`.

```python
# Bug: nessuna validazione degli scores
y_mean = all_scores.mean()  # NaN se c'è un NaN!
y_std = all_scores.std() + 1e-6  # NaN!
y_centered = (all_scores - y_mean) / y_std  # Tutto NaN!
# → grad = NaN
```

### Root Cause

Il calcolo della normalizzazione usa `mean()` e `std()` che propagano NaN:
- Un singolo score NaN → `y_mean = NaN`
- `y_mean = NaN` → `y_centered = NaN` (tutto il vettore)
- `y_centered = NaN` → `grad = NaN`

### Fix Applicata

```python
# In lgs.py, dopo aver estratto all_pts e all_scores:

# BUG FIX: Remove NaN/Inf scores to prevent NaN propagation
valid_mask = np.isfinite(all_scores)
if not valid_mask.all():
    all_pts = all_pts[valid_mask]
    all_scores = all_scores[valid_mask]
    if len(all_scores) < dim + 2:
        return None
```

### Verifica

- Test con score NaN: gradiente calcolato correttamente sui punti validi
- Se troppi NaN (under threshold): ritorna None (safe fallback)

---

## Finding 25: NaN in Acquisition Selection
**Data:** Questa sessione  
**Stato:** ✅ RISOLTO

### Problema

Quando `mu` o `sigma` contengono `NaN` o `Inf`, il selettore poteva:
1. Selezionare candidati con predizioni invalide
2. Con tutti NaN, selezionare uniformemente (comportamento degno ma non documentato)

### Root Cause

```python
# Comportamento originale con NaN:
score = mu + beta * sigma  # NaN se mu o sigma sono NaN
score.std() > 1e-9  # NaN > x = False (quindi score_z = zeros)
# → selezione uniforme, ma candidati NaN potrebbero essere scelti
```

### Fix Applicata

```python
# BUG FIX: Handle NaN/Inf in mu and sigma
valid_mu = np.isfinite(mu)
valid_sigma = np.isfinite(sigma)

if not valid_mu.all():
    if valid_mu.any():
        mu = np.where(valid_mu, mu, np.median(mu[valid_mu]))
    else:
        mu = np.zeros_like(mu)  # All NaN → uniform selection

if not valid_sigma.all():
    if valid_sigma.any():
        sigma = np.where(valid_sigma, sigma, np.median(sigma[valid_sigma]))
    else:
        sigma = np.ones_like(sigma)  # All NaN → equal uncertainty

# Handle NaN/Inf novelty_weight
if not np.isfinite(beta):
    beta = 0.0  # Fall back to pure exploitation
```

### Verifica

- Test con alcuni NaN: candidati validi vengono preferiti
- Test con tutti NaN: selezione uniforme (comportamento documentato)
- Test con novelty_weight NaN/Inf: fallback a exploitation

---

## Finding 26: Split Axis Bias - Gradient Threshold Troppo Basso
**Data:** Gennaio 2026  
**Stato:** ✅ RISOLTO

### Problema Osservato

ALBA splittava sempre la stessa dimensione (tipicamente dim 0 o 3), creando
cubi con aspect ratio fino a 15-18x invece di ~2x.

```
Esempio pre-fix:
  Cube widths: [0.125, 1.0, 1.0, 0.25, 1.0]
  Aspect ratio: 8.0x (molto sbilanciato)
  Splits per dimension: dim0=5, dim1=0, dim2=0, dim3=2, dim4=0
```

### Causa Root

Il threshold del gradiente in `get_split_axis()` era **troppo basso** (0.3):

```python
# VECCHIO CODICE (BUGGATO)
if dominant_val > 0.3:  # Threshold fisso
    return dominant_dim
```

**Problema matematico:** In spazio normalizzato [0,1]^d, un gradiente unitario 
ha sempre almeno una componente ≥ 1/√d:

| Dimensione | 1/√d | Threshold 0.3 |
|------------|------|---------------|
| d=3 | 0.577 | ✗ sempre superato |
| d=5 | 0.447 | ✗ sempre superato |
| d=10 | 0.316 | ✗ sempre superato |

→ Il gradiente **SEMPRE** guidava la scelta, impedendo il bilanciamento!

### Effetto Catena

1. Early split su dim X basato su gradiente rumoroso
2. Dim X diventa stretta → gradiente in X diventa piccolo (normalizzato)
3. Prossimo split su dim Y → stesso problema
4. Feedback loop → 2-3 dimensioni monopolizzano gli split

### Fix Implementata

```python
def get_split_axis(self) -> int:
    dim = len(self.bounds)
    widths = np.array([hi - lo for lo, hi in self.bounds])
    
    # RULE 1: Se aspect ratio > 2.0, SEMPRE splittare dimensione più larga
    aspect_ratio = widths.max() / (widths.min() + 1e-9)
    if aspect_ratio > 2.0:
        return int(np.argmax(widths))
    
    # RULE 2: Per depth <= 2, sempre dimensione più larga (previene bias iniziale)
    if self.depth <= 2:
        return int(np.argmax(widths))
    
    # RULE 3: Gradient threshold MOLTO più alto (dim-adaptive)
    # In spazio normalizzato: threshold = 2/√d (difficile da superare)
    adaptive_threshold = 2.0 / np.sqrt(dim)
    grad_dir = self.model.get("gradient_dir") if self.model else None
    
    if grad_dir is not None and np.all(np.isfinite(grad_dir)):
        grad_scaled = np.abs(grad_dir) * widths
        grad_scaled_norm = grad_scaled / (np.linalg.norm(grad_scaled) + 1e-9)
        dominant_dim = int(np.argmax(grad_scaled_norm))
        dominant_val = grad_scaled_norm[dominant_dim]
        
        if dominant_val > adaptive_threshold:
            return dominant_dim
    
    # RULE 4: Fallback su varianza punti buoni
    # ... (resto invariato)
```

### Giustificazione Parametri

| Parametro | Valore | Giustificazione |
|-----------|--------|-----------------|
| `aspect_ratio > 2.0` | 2.0 | Limite oltre cui il cubo è troppo sbilanciato |
| `depth <= 2` | 2 | Prime 2-3 split devono bilanciare, non seguire rumore |
| `adaptive_threshold = 2/√d` | ~0.89 per d=5 | Richiede gradiente **MOLTO** dominante (raro) |

### Risultati Post-Fix

**Aspect Ratio:**
- Prima: 15-18x
- Dopo: 1.9-2.2x ✓

**Split Distribution (5 dimensioni):**
- Prima: [5, 0, 0, 2, 0] (2 dimensioni monopolizzano)
- Dopo: [2, 1, 1, 2, 1] (tutte le dimensioni usate) ✓

**Performance vs Random (100 trials, 10 seeds):**

| Funzione | Prima Fix | Dopo Fix |
|----------|-----------|----------|
| Sphere | +63.6% | **+80.7%** |
| Rosenbrock | +72.5% | **+85.9%** |
| Rastrigin | -5.2% | **+13.6%** |
| Ackley | +28.1% | **+33.7%** |
| Griewank | +8.4% | **+11.3%** |
| Levy | +52.3% | **+68.0%** |

→ Miglioramento su **TUTTE** le funzioni!

---

## Finding 27: UCB-Softmax vs Argmax - Softmax Salva LGS
**Data:** Gennaio 2026  
**Stato:** ✅ COMPORTAMENTO CORRETTO (non bug)

### Osservazione Iniziale

Test iniziali mostravano che UCB con argmax selezionava candidati con 
rank medio ~32 (peggio di random!). Questo sembrava indicare che LGS 
fosse "rotto".

### Analisi Approfondita

**Beta Sweep (argmax UCB):**

| Beta | Avg Rank | Top-10 Rate |
|------|----------|-------------|
| 0.00 | 26.6 | 21.5% |
| 0.10 | 25.3 | 19.5% ← Ottimo |
| 0.30 | 28.5 | 14.0% |
| 0.80 | 29.3 | 15.2% ← ALBA usa questo |
| 1.50 | 33.9 | 12.5% |

Con argmax, beta=0.1 sarebbe ottimale, ma ALBA usa beta=0.8.

### Scoperta Chiave

Testando il **vero** metodo di selezione ALBA (UCB-Softmax):

| Metodo | Avg Rank | Top-10 Rate |
|--------|----------|-------------|
| Argmax UCB (beta=0.8) | 29.3 | 15.2% |
| Random | 32.0 | 15.6% |
| **Softmax UCB (ALBA)** | **19.8** | **34.0%** ✓ |

→ Il **softmax con temperatura 3.0** trasforma il problema!

### Perché Softmax Funziona

1. **Argmax** seleziona sempre il candidato con score massimo
   - Se sigma è grande, seleziona punti incerti (spesso cattivi)
   - Beta alto amplifica questo problema

2. **Softmax** converte scores in probabilità:
   ```
   p_i = exp(score_i / temperature) / Σ exp(score_j / temperature)
   ```
   - Candidati buoni hanno probabilità alta ma non 100%
   - Aggiunge stocasticità benefica
   - Temperature 3.0 bilancia exploitation/exploration

3. **Effetto netto:** Anche se alcuni candidati hanno score UCB alto
   per sigma grande, il softmax permette di selezionare anche
   candidati con mu alta e sigma bassa.

### Conclusione

- ✅ LGS funziona bene (Spearman 0.4-0.6 localmente)
- ✅ UCB-Softmax è la scelta **giusta** per compensare sigma rumoroso
- ⚠️ Argmax con beta alto sarebbe problematico
- 💡 La combinazione (beta=0.8, temperature=3.0) è ben bilanciata

### Parametri Giustificati

| Parametro | Valore | Giustificazione |
|-----------|--------|-----------------|
| `beta_multiplier` | 2.0 | Con softmax, beta alto è OK (esplorazione) |
| `softmax_temperature` | 3.0 | Bilancia probabilità, evita determinismo |
| `novelty_weight` | 0.4 | beta_finale = 0.4 * 2.0 = 0.8 |

---

## Finding 28: Potential Field Anchor Metric - best_score vs good_ratio
**Data:** Gennaio 2026  
**Stato:** ✅ ANALISI COMPLETA - good_ratio è MEGLIO

### Problema Iniziale

Il Potential Field per la leaf selection usava `good_ratio()` come metrica
per ancorare il potenziale. Ho ipotizzato che `best_score` fosse migliore
perché ha correlazione perfetta (-1.0) con la qualità della foglia.

### Test di Correlazione (Iniziale - FUORVIANTE)

| Metrica | Correlazione con best_y | Varianza |
|---------|------------------------|----------|
| `best_score` | **-1.000** (perfetta!) | 0.000 |
| `good_ratio` | -0.733 | 0.232 |

Sembrava ovvio che `best_score` fosse migliore...

### Test A/B sui Risultati Finali (DECISIVO)

Ma il vero test è: quale metrica porta a risultati FINALI migliori?

| Funzione | best_score | good_ratio | Winner |
|----------|------------|------------|--------|
| Sphere | 1.28 | 1.38 | best_score |
| Rosenbrock | 255 | **158** | **good_ratio** |
| Rastrigin | 30.4 | **27.2** | **good_ratio** |
| Ackley | 3.52 | **3.29** | **good_ratio** |
| Griewank | 0.38 | **0.36** | **good_ratio** |
| Levy | 2.49 | **2.05** | **good_ratio** |
| Schwefel | 750 | **662** | **good_ratio** |
| Mixed | 13.1 | **13.0** | **good_ratio** |

**Totale: best_score 1, good_ratio 7**

### Perché good_ratio vince nonostante correlazione peggiore?

Il paradosso si spiega così:

1. **best_score può ingannare verso "fake valleys"**:
   - Una foglia può avere un singolo punto molto buono (best_score alto)
   - Ma quel punto potrebbe essere un minimo locale isolato
   - Concentrarsi lì spreca budget

2. **good_ratio misura la "densità di punti buoni"**:
   - Molti punti sopra gamma = regione generalmente promettente
   - Più robusto a outlier
   - Favorisce esplorazione di regioni con alta probabilità di successo

3. **L'esplorazione beneficia dalla stocasticità**:
   - good_ratio ha alta varianza → più esplorazione casuale
   - In problemi multimodali, questo aiuta a sfuggire minimi locali

### Lezione Appresa

**La correlazione con la metrica "vera" non è sufficiente!**

Bisogna sempre testare l'effetto sul risultato finale. Una metrica "noisy" può
essere migliore se quella noise favorisce esplorazione benefica.

### Decisione Finale

✅ **Manteniamo good_ratio** per l'ancoraggio del potential field.

```python
# CORRETTO (good_ratio)
leaf_good_ratios = np.array([leaves[i].good_ratio() for i in range(n)])
empirical_bonus = leaf_good_ratios * 2.0

# SBAGLIATO (best_score - sembrava migliore ma peggiora i risultati)
# leaf_best_scores = np.array([leaves[i].best_score for i in range(n)])
```

---

## Finding 31: CovarianceLocalSearchSampler Broken in High Dimensions
**Data:** Gennaio 2026  
**Stato:** ✅ RISOLTO

### Problema

Il `CovarianceLocalSearchSampler` performava peggio del semplice `GaussianLocalSearchSampler` 
su molte funzioni, specialmente in alta dimensione (10D+).

### Cause Identificate (attraverso debugging approfondito)

1. **Centering sbagliato**: Il sampler centrava i campioni su `mu_w` (media pesata 
   dei top-k punti) invece che su `best_x`. Questo causava deriva verso zone subottimali.

2. **Condizionamento che esplode in alta dimensione**:
   - Dim 3: condition number ~3,800
   - Dim 10: condition number ~6,500  
   - Dim 20: condition number ~17,000
   
   Con soglia 1000, quasi tutti i campioni in alta-D usavano il fallback Gaussiano!

3. **Troppi pochi punti per stimare la covarianza**: Con `top_k_fraction=0.15` fisso,
   in alta dimensione non c'erano abbastanza punti per una stima affidabile della
   matrice di covarianza (servono almeno O(dim²) punti).

4. **Scala troppo bassa**: Il moltiplicatore era 3.0, ma i test empirici mostrano
   che 5.0 funziona meglio per esplorare lo spazio delle direzioni.

### Fix Applicati

| Prima | Dopo | Perché |
|-------|------|--------|
| `top_k_fraction = 0.15` fisso | `fraction = min(0.5, 0.15 + 0.02*dim)` | Più punti in alta-D per stima affidabile |
| `eps = 1e-6` | `eps = 0.01 * (1 + 0.1*dim)` | Regolarizzazione proporzionale alla dimensione |
| `scale * 3.0` | `scale * 5.0` | Scala maggiore = più esplorazione |
| Centrato su `mu_w` | Centrato su `best_x` | Rimane vicino al vero migliore |
| Check condition > 1000 → fallback | Rimosso (la regolarizzazione risolve) | Usa sempre la covarianza |

### Risultati Dopo Fix

Test su 30 seed, 100 budget:

| Funzione | Dim | Cov wins | Improvement vs Gaussian |
|----------|-----|----------|------------------------|
| Rosenbrock | 3 | 23/30 (77%) | +29.5% |
| Rosenbrock | 5 | 17/30 (57%) | +8.5% |
| Rastrigin | 3 | 22/30 (73%) | +30.3% |
| Rastrigin | 5 | 18/30 (60%) | +11.5% |
| Rastrigin | 10 | 21/30 (70%) | +7.1% |
| Sphere | * | ~50% | ~0% (come atteso: isotropa) |

### Insight Teorico

Il sampler Cov funziona bene su **funzioni anisotrope** (Rosenbrock, Rastrigin, 
Cigar, Discus) perché impara la "forma" della valle. Su funzioni **isotrope** 
(Sphere) non c'è vantaggio perché non c'è struttura geometrica da sfruttare.

### Strumenti di Debug Creati

- `tools/debug_rosenbrock_cov.py` - Investigazione iniziale
- `tools/debug_rosenbrock_cov_multi_seed.py` - Test multi-seed
- `tools/debug_cov_trap.py` - Analisi volume e direzione
- `tools/deep_scale_research.py` - Ricerca scala ottimale
- `tools/deep_10d_investigation.py` - Perché 10D fallisce
- `tools/test_improved_cov.py` - Test configurazioni
- `tools/test_final_cov.py` - Validazione finale

---

*Ultimo aggiornamento: Gennaio 2026*


