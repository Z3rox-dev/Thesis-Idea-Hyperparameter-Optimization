# Bug Hunting Session Summary

## Sessione di Debug - Riepilogo

**Data:** Sessione corrente (Aggiornato)  
**Metodo:** Stress testing sistematico con edge cases estremi

---

## Bug Trovati e Fixati

### 1. Finding 22: NaN in Potentials (coherence.py)
- **Problema:** `good_ratio()` con NaN corrompeva tutti i potentials
- **Fix:** Sanitize good_ratios con `np.isfinite()` check
- **File:** `coherence.py` line ~373-385

### 2. Finding 23: NaN in Candidates (candidates.py)  
- **Problema:** `gradient_dir` con NaN produceva ~13% candidati invalidi
- **Fix:** Check `np.all(np.isfinite(grad_dir))` prima di usare gradient strategy
- **File:** `candidates.py` lines ~46-55

### 3. Finding 24: NaN in LGS Gradient (lgs.py)
- **Problema:** Score con NaN propagavano NaN al gradiente
- **Fix:** Filter out NaN/Inf scores prima del fitting
- **File:** `lgs.py` lines ~59-67

### 4. Finding 25: NaN in Acquisition (acquisition.py)
- **Problema:** mu/sigma con NaN potevano selezionare candidati invalidi
- **Fix:** Sanitize mu/sigma con median replacement per valori NaN
- **File:** `acquisition.py` lines ~45-62

### 5. Finding 26: NaN in Drilling (drilling.py) - LOW PRIORITY
- **Problema:** `start_x` con NaN propagava NaN in `ask()`
- **Fix:** Sanitize `start_x` con bounds center, `start_y` NaN → +Inf
- **File:** `drilling.py` lines ~37-56
- **Note:** Difesa in profondità, non critico nel contesto ALBA

### 6. Finding 27: NaN in Gamma (gamma.py) - HIGH PRIORITY
- **Problema:** `np.percentile(y_all)` restituisce NaN se y_all contiene NaN
- **Fix:** Filter NaN/Inf da y_all prima di calcolare percentile
- **File:** `gamma.py` lines ~49-62
- **Impact:** Corrupta lo stato core dell'algoritmo (gamma threshold)

### 7. Finding 28: Discretize out of range (categorical.py)
- **Problema:** `discretize(-0.5, 3) = -1` (indice negativo)
- **Fix:** Clamp input a [0,1] e output a [0, n_choices-1]
- **File:** `categorical.py` lines ~84-103

### 8. Finding 29: NaN in Local Search (local_search.py)
- **Problema:** NaN in `best_x` o `progress` propagava in output
- **Fix:** Sanitize best_x e progress prima del sampling
- **File:** `local_search.py` (entrambi i sampler)

### 9. Finding 30: NaN in Leaf Selection (leaf_selection.py)
- **Problema:** NaN da `good_ratio()` causava "probabilities contain NaN"
- **Fix:** Sanitize ratio e potential, fallback a selezione uniforme
- **File:** `leaf_selection.py` (entrambi i selector)

---

## Test Suites Create

1. `tools/deep_coherence_test.py` - Test unitari per coherence module
2. `tools/coherence_stress_test.py` - Edge cases numerici estremi
3. `tools/candidates_stress_test.py` - Test generazione candidati
4. `tools/lgs_stress_test.py` - Test Local Gradient Surrogate
5. `tools/cube_stress_test.py` - Test Cube splitting e geometry
6. `tools/acquisition_stress_test.py` - Test selezione UCB/softmax
7. `tools/drilling_stress_test.py` - Test (1+1)-CMA-ES drilling optimizer
8. `tools/optimizer_stress_test.py` - Test end-to-end ALBA optimizer
9. `tools/categorical_stress_test.py` - Test categorical sampling
10. `tools/local_search_stress_test.py` - Test local search samplers
11. `tools/leaf_selection_stress_test.py` - Test leaf selection strategies

---

## Moduli Verificati Senza Bug

- `cube.py` - Splitting e geometry corretti
- `splitting.py` - Semplice, usa cube.split()

---

## Pattern Comune dei Bug

Tutti i bug trovati seguono lo stesso pattern:
1. **Input invalidation mancante** per NaN/Inf
2. **NaN propagation** attraverso calcoli matematici
3. **No fallback** per gestire casi degenerati

La fix pattern è consistente:
```python
# Check validity
valid_mask = np.isfinite(values)
if not valid_mask.all():
    if valid_mask.any():
        # Replace with median of valid values
        values = np.where(valid_mask, values, np.median(values[valid_mask]))
    else:
        # All invalid → use neutral default
        values = np.full_like(values, default_value)
```

---

## Stato Finale

**Tutti i moduli core di ALBA framework ora sono robusti a NaN/Inf:**
- ✅ coherence.py (Finding 22)
- ✅ candidates.py (Finding 23)
- ✅ lgs.py (Finding 24)
- ✅ acquisition.py (Finding 25)
- ✅ drilling.py (Finding 26)
- ✅ gamma.py (Finding 27)
- ✅ categorical.py (Finding 28)
- ✅ local_search.py (Finding 29)
- ✅ leaf_selection.py (Finding 30)
- ✅ optimizer.py (uses fixed modules)
- ✅ cube.py (no issues found)
- ✅ splitting.py (no issues found)

**Test coverage:** 11 stress test suites, all passing
- ✅ cube.py
- ✅ splitting.py
