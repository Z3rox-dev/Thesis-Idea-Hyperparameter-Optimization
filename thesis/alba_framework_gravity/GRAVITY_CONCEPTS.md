# Concetti di Gravità in ALBA - Geo-Drift

## 1. MOTIVAZIONE

ALBA esplora lo spazio dei parametri usando:
- Partizionamento gerarchico (cube tree)
- Local search in cubi promettenti
- LGS per predizioni locali

**Problema**: Local search può vagare senza direzione quando LGS è rumoroso.

**Soluzione**: Usare un "campo gravitazionale" che attira le particelle verso 
regioni promettenti dello spazio, ispirandosi alla fisica newtoniana.

---

## 2. IMPLEMENTAZIONE ATTUALE

### 2.1 CubeGravity - Campo gravitazionale a livello di cubo

**Analogia fisica**:
- Ogni cubo è un "corpo celeste"
- Il **potenziale Φ_c** = EMA della loss osservata (minore = migliore)
- La **massa** = quanto il cubo è "attraente" = (Φ_max - Φ_c)
- La **posizione** = centro del cubo

**Formula per il drift**:
```
F(x) = Σ_c (Φ_max - Φ_c) · (center_c - x) / dist²
```

Dove:
- Cubi con potenziale basso (buoni) hanno massa alta → attraggono di più
- L'attrazione decresce con il quadrato della distanza (legge di Newton)
- Il drift finale è la somma vettoriale di tutte le forze

**Score per selezione cubo**:
```
score(c) = -Φ_c + λ·attraction(c) - μ·visits(c)
```
- `-Φ_c`: Preferisci cubi con basso potenziale (buoni risultati)
- `attraction`: Preferisci cubi attratti da altri buoni
- `-visits`: Penalizza cubi già visitati troppo (repulsione)

### 2.2 FreeGeometryEstimator - Geometria senza valutazioni extra

**Problema**: Il landscape può avere scale diverse per ogni dimensione.
Una dimensione "ripida" richiede passi più piccoli.

**Soluzione**: Stimare la sensibilità ∂f/∂x_i da coppie di punti già osservati.

**Formula**:
```
sensitivity[i] ≈ median( |f(x) - f(x')| / |x[i] - x'[i]| )
```

Per coppie (x, x') dove:
- La differenza è principalmente lungo la dimensione i
- alignment = |Δx[i]| / Σ|Δx| > 0.4

**Modulazione del drift**:
```python
scale[i] = 1 / sqrt(sensitivity[i] / avg_sensitivity)
drift_modulated = drift * scale
```
- Dimensioni sensibili → scale piccolo → drift piccolo
- Dimensioni piatte → scale grande → drift grande

### 2.3 Integrazione nel Local Search

```python
# In optimizer.py, local_search_step():
if self.geo_drift and self.cube_gravity:
    drift = self.cube_gravity.get_drift_vector(x, all_leaves)
    drift = self.geometry_estimator.modulate_drift(drift)
    
    x_drifted = x + drift_strength * drift
    x_drifted = np.clip(x_drifted, cube.lo, cube.hi)
    candidates.append(x_drifted)
```

---

## 3. RISULTATI SPERIMENTALI

Su benchmark YAHPO (3 task × 3 seed):
- **Win rate**: 70% (7/10 confronti ALBA vince)
- **Media miglioramento**: +33.3%
- **Config ottimale**: `geo_drift=True, novelty_weight=0, local_search_ratio=0.5`

---

## 4. ACCELERAZIONE GRAVITAZIONALE - NUOVE IDEE

### 4.1 Concetto fisico

Nella fisica classica, l'accelerazione gravitazionale è:
```
a = F/m = GM/r²
```

Ma soprattutto: **la velocità accumula nel tempo**
```
v(t) = v(0) + ∫ a dt
```

Attualmente il drift è un **vettore istantaneo** (velocità), non accumula storia.

### 4.2 Idea: Momentum Gravitazionale

**Proposta**: Mantenere uno stato di "velocità" per ogni particella che accumula drift.

```python
class GravitationalMomentum:
    def __init__(self, n_dims, friction=0.8):
        self.velocity = np.zeros(n_dims)
        self.friction = friction  # Attrito per stabilità
    
    def update(self, acceleration: np.ndarray) -> np.ndarray:
        """
        v(t+1) = friction * v(t) + acceleration
        """
        self.velocity = self.friction * self.velocity + acceleration
        return self.velocity
```

**Vantaggi**:
- **Inerzia**: Se stai andando verso una zona buona, continui anche se l'attrazione momentanea cambia
- **Smoothing**: Meno sensibile a rumore nelle stime di drift
- **Escape**: Può aiutare a uscire da minimi locali (come SGD con momentum)

### 4.3 Idea: Accelerazione adattiva per cubo

Invece di un drift uniforme, ogni cubo potrebbe avere la sua "accelerazione" 
basata sulla storia recente:

```python
class CubeAcceleration:
    def __init__(self):
        self.prev_best = None
        self.improvement_rate = 0.0  # "accelerazione" del miglioramento
    
    def update(self, new_best):
        if self.prev_best is not None:
            delta = self.prev_best - new_best  # Improvement
            # Se miglioriamo rapidamente, accelera
            self.improvement_rate = 0.8 * self.improvement_rate + 0.2 * delta
        self.prev_best = new_best
```

**Uso**: Cubi che stanno migliorando rapidamente ricevono più budget (sono in "caduta libera" verso l'ottimo).

### 4.4 Idea: Orbite stabili vs caduta libera

Nella fisica orbitale:
- **Orbita stabile**: La particella gira attorno al minimo senza caderci
- **Caduta libera**: La particella precipita verso il minimo

Potremmo classificare il comportamento in un cubo:

```python
def classify_trajectory(history):
    """
    Analizza se la traiettoria è:
    - CONVERGING: Si sta avvicinando al best (caduta libera)
    - ORBITING: Gira attorno al best (esplorazione)
    - ESCAPING: Si sta allontanando (repulsione)
    """
    if len(history) < 5:
        return "UNKNOWN"
    
    dists_to_best = [np.linalg.norm(x - best_x) for x, _ in history[-5:]]
    trend = np.polyfit(range(5), dists_to_best, 1)[0]
    
    if trend < -0.1:
        return "CONVERGING"  # Aumenta exploitation
    elif trend > 0.1:
        return "ESCAPING"    # Cambia zona
    else:
        return "ORBITING"    # Continua esplorazione
```

### 4.5 Idea: Campo gravitazionale con "massa variabile"

Nella fisica la massa è costante. Ma qui potremmo far dipendere la "massa" 
di un cubo dalla sua **confidenza**:

```python
confidence_mass = visits_in_cube / (1 + variance_in_cube)
```

- Tanti punti + bassa varianza → alta confidenza → massa stabile
- Pochi punti o alta varianza → massa instabile → meno influenza sul campo

### 4.6 Proposta concreta: Geo-Drift v2 con Momentum

```python
# Nuovo parametro
momentum_friction: float = 0.7  # 0=no momentum, 1=full momentum

# Stato persistente
self.drift_velocity = {}  # cube_id -> velocity vector

def local_search_step_v2(self, cube, x):
    cube_id = id(cube)
    
    # Calcola accelerazione (forza / massa)
    acceleration = self.cube_gravity.get_drift_vector(x, all_leaves)
    acceleration = self.geometry_estimator.modulate_drift(acceleration)
    
    # Applica momentum
    if cube_id not in self.drift_velocity:
        self.drift_velocity[cube_id] = np.zeros_like(x)
    
    self.drift_velocity[cube_id] = (
        self.momentum_friction * self.drift_velocity[cube_id] + 
        (1 - self.momentum_friction) * acceleration
    )
    
    velocity = self.drift_velocity[cube_id]
    
    # Applica velocità
    x_new = x + drift_strength * velocity
    return np.clip(x_new, cube.lo, cube.hi)
```

---

## 5. CONSIDERAZIONI IMPLEMENTATIVE

### Pro del momentum:
1. **Smoother trajectories**: Meno salti erratici
2. **Faster convergence**: Accumula velocità verso ottimi
3. **Better escape**: Inerzia può superare barriere locali

### Contro del momentum:
1. **Overshoot**: Può superare l'ottimo se friction troppo basso
2. **Stale velocity**: In spazi che cambiano (split), la velocity può essere obsoleta
3. **Complexity**: Più stato da mantenere

### Quando usarlo:
- **Funzioni smooth**: Momentum aiuta molto
- **Funzioni rumorose**: Meglio friction alto (0.9) o disattivare
- **Early optimization**: Momentum accelera convergenza
- **Late optimization**: Meglio ridurre momentum per fine-tuning

---

## 6. NEXT STEPS SUGGERITI

1. **Implementare `GravitationalMomentum`** con friction configurabile
2. **Testare su YAHPO/JAHS** confrontando:
   - `geo_drift=True, momentum=0` (attuale)
   - `geo_drift=True, momentum=0.7`
   - `geo_drift=True, momentum=0.9`
3. **Analizzare traiettorie** per vedere se momentum effettivamente:
   - Produce curve più smooth
   - Converge più velocemente
   - Evita oscillazioni

---

## 7. DEFORMAZIONE DELLO SPAZIO (Analisi Approfondita)

Oltre alla gravità Newtoniana, la **Relatività Generale** offre concetti interessanti.

### 7.1 Concetti dalla Relatività Generale

| Concetto | Fisica | Applicazione ALBA |
|----------|--------|-------------------|
| **Metrica Adattiva** | ds² = g_ij dx^i dx^j | Distanze che dipendono dalla qualità locale |
| **Geodetiche** | Percorsi più brevi in spazio curvo | Traiettorie che passano per zone buone |
| **Curvatura** | Tensore di Riemann | Step size adattivo alla complessità |
| **Lensing** | Luce deviata dalla massa | Direzione ricerca deviata da attrattori |

### 7.2 Gravitational Lensing (TESTATO ✓)

**Idea**: La direzione di ricerca viene "deviata" dai punti buoni.

```python
def apply_lensing(direction, current_pos, attractors, strength=0.1):
    for attr in attractors:
        to_attr = attr - current_pos
        dist = np.linalg.norm(to_attr) + 0.01
        direction += strength * to_attr / (dist**2)
    return direction / np.linalg.norm(direction)
```

**Risultati test (Rosenbrock, 150 iter, 15 seeds)**:
| Lensing | Mean f | Std |
|---------|--------|-----|
| 0.0 | 118.5 | 156.8 |
| **0.1** | **7.7** | 5.9 |
| 0.3 | 21.7 | 16.6 |
| 0.5 | 28.8 | 43.2 |

→ **Lensing 0.1 riduce l'errore del 93%!**

### 7.3 Warped Space Sampling (TESTATO ✓)

**Idea**: Campionare in uno spazio deformato dove zone buone sono "espanse".

```python
def warp_point(x, good_points, warp_strength):
    warped = x.copy()
    for gp in good_points:
        diff = gp - x
        dist = np.linalg.norm(diff) + 0.01
        warped += warp_strength * np.exp(-dist**2 / 0.3) * diff
    return warped
```

**Risultati test (Rosenbrock)**:
| Warp | Mean f |
|------|--------|
| 0.0 | 4.0 |
| 0.4 | 3.7 |
| 0.6 | 2.9 |
| **0.8** | **1.8** |

→ Warped space funziona bene ma è più instabile (std alta).

### 7.4 Curvatura Adattiva (TESTATO ✗)

**Idea**: Stimare la curvatura locale e ridurre step in zone "ripide".

**Risultati**: Effetto **negativo** su Rosenbrock (-36%), modesto su Sphere (+9%).

**Motivo**: La stima della Hessiana richiede molti punti locali. Con pochi campioni 
la stima è rumorosa e porta a step troppo piccoli.

### 7.5 Confronto Finale

| Tecnica | Efficacia | Complessità | Raccomandazione |
|---------|-----------|-------------|-----------------|
| **Lensing** | ★★★★★ | Bassa | **IMPLEMENTARE** |
| **Warped Space** | ★★★☆☆ | Media | Considerare |
| **Curvatura** | ★☆☆☆☆ | Alta | Evitare |
| **Momentum** | ★★☆☆☆ | Bassa | Non necessario |

---

## 8. FORMULA GEO-DRIFT FINALE

Combinando i risultati dei test, la formula ottimale è:

```
# 1. Calcola drift gravitazionale base (verso centroidi buoni cubi)
drift = cube_gravity.get_drift_vector(x, all_leaves)

# 2. Modula per geometria stimata
drift = geometry_estimator.modulate_drift(drift)

# 3. Applica lensing sulla direzione random (NUOVO)
random_dir = normalize(rng.normal(0, 1, n_dims))
lensed_dir = apply_lensing(random_dir, x, top_attractors, strength=0.1)

# 4. Combina drift e lensed direction
final_direction = 0.5 * drift + 0.5 * lensed_dir

# 5. Genera candidato
x_new = x + step_size * final_direction
```

**Parametri consigliati**:
- `drift_strength`: 0.15 (come attuale)
- `lensing_strength`: 0.1
- `n_attractors`: 5 (top punti buoni)
---

## 9. TEST SU BENCHMARK REALISTICI (HPO)

### 9.1 Funzioni Sintetiche vs HPO Reali

I test iniziali erano su **funzioni sintetiche** (Rosenbrock, Sphere, etc.):
- Smooth, deterministiche
- Ottimo globale unico e ben definito
- **Lensing molto efficace** (+53% improvement medio)

### 9.2 Test su Scenari HPO Simulati

Testato su 6 problemi HPO realistici che simulano tuning di:
- NN Learning Rate Schedule (5D)
- Transformer HPO (6D)
- CNN Architecture (6D)
- XGBoost HPO (6D)
- Random Forest HPO (5D)
- SVM HPO (4D)

**Caratteristiche dei problemi HPO**:
- Rumore (noise ~0.01)
- Plateau e zone piatte
- Interazioni complesse tra parametri
- Scale logaritmiche (learning rate, regularization)

### 9.3 Risultati Completi

| Benchmark | ALBA base | Lens 0.1 | Lens 0.2 | Optuna | Winner |
|-----------|-----------|----------|----------|--------|--------|
| NN_LR_Schedule (5D) | **0.440** | 0.447 | 0.457 | 0.446 | ALBA base |
| Transformer_HPO (6D) | **0.412** | 0.428 | 0.423 | 0.427 | ALBA base |
| CNN_Architecture (6D) | 0.373 | 0.374 | 0.378 | **0.364** | Optuna |
| XGBoost_HPO (6D) | 0.247 | 0.251 | 0.256 | **0.245** | Optuna |

**WINS (4 benchmark)**:
- ALBA base: 2/4 (50%)
- Optuna: 2/4 (50%)
- Lensing: 0/4 (0%)

### 9.4 Conclusioni sui Benchmark HPO

**Lensing NON migliora su problemi HPO realistici**:

1. **Rumore**: Il rumore nei dati HPO confonde gli attrattori.
   I "migliori punti" possono essere outlier fortunati.

2. **Landscape complesso**: Gli HPO hanno interazioni non-lineari
   tra parametri. Lensing assume landscape più semplici.

3. **Plateau**: Molte regioni hanno loss simili. Lensing
   attrae verso punti che non sono significativamente migliori.

4. **ALBA base già buono**: Su questi problemi, local search
   standard funziona bene senza bisogno di bias direzionale.

### 9.5 Quando usare Lensing

| Scenario | Lensing utile? |
|----------|----------------|
| Funzioni smooth (Rosenbrock, Sphere) | ✓ Sì (+50-80%) |
| Funzioni multimodali leggere (Ackley) | ✓ Sì (+40-60%) |
| HPO con rumore | ✗ No |
| Problemi con molti plateau | ✗ No |
| NAS / architettura search | ? Da testare |

### 9.6 Raccomandazione Finale

**Per ALBA framework**:
1. **Mantenere geo-drift** (funziona bene)
2. **Lensing opzionale**: `lensing_strength=0.0` di default
3. **Attivare lensing** solo se:
   - Funzione deterministica (no noise)
   - Ottimo globale ben definito
   - Early convergence lenta

**Parametro suggerito**:
```python
lensing_strength: float = 0.0  # Default off
# Attivare con 0.1-0.2 per funzioni smooth
```

---

## 10. LOCAL LINEAR REGRESSION GRADIENT (LLR) ⭐

### 10.1 Motivazione

L'idea della **deformazione dello spazio** continua ad essere promettente.
La chiave è stimare un "gradiente" senza avere accesso al gradiente vero.

**Problema con Lensing**: Attrae verso i "migliori punti" osservati, ma:
- Su funzioni rumorose, i "migliori" possono essere outlier
- Non considera la struttura locale del landscape

**Soluzione: LLR Gradient**
- Fitta un piano locale ai K vicini
- Il gradiente del piano è la direzione di discesa stimata
- Più robusto perché usa TUTTI i vicini, non solo i migliori

### 10.2 Formula LLR Gradient

Per stimare il gradiente in un punto x:

1. **Trova K vicini**: {(x_i, f_i)} con K ≥ dim + 2

2. **Calcola pesi gaussiani**:
   ```
   w_i = exp(-|x_i - x|² / 2σ²)
   dove σ = median(distances)
   ```

3. **Fit piano locale pesato**:
   ```
   min Σ_i w_i · (f_i - a·x_i - b)²
   ```

4. **Estrai gradiente**:
   ```
   gradient = a  (coefficienti del fit)
   descent_direction = -a / |a|
   ```

5. **Combina con random**:
   ```
   direction = (1-gw) · random + gw · descent_direction
   ```

### 10.3 Risultati Test

| Funzione | Random | LLR (gw=0.7) | Improvement |
|----------|--------|--------------|-------------|
| Rosenbrock 5D | 1.200 | 1.166 | +2.8% |
| Sphere 5D | 0.000 | 0.000 | +68.2% |
| Rastrigin 5D | 0.094 | **0.022** | **+76.3%** ⭐ |
| Ackley 5D | 0.046 | **0.021** | **+54.0%** ⭐ |
| Levy 5D | 0.000 | 0.000 | +68.3% |

**WIN RATE: 100% (5/5)**
**MEAN IMPROVEMENT: +53.9%**

### 10.4 Confronto Approcci

| Approccio | Win Rate | Best Impr | Note |
|-----------|----------|-----------|------|
| **LLR Gradient** | **100%** | **+76%** | ⭐ Migliore |
| Pseudo-Gradient | 75% | +40% | Alternativa leggera |
| Lensing | 80%* | +80% | *Fallisce su HPO rumoroso |
| Random | 0% | 0% | Baseline |

### 10.5 Vantaggi LLR vs Lensing

| Aspetto | Lensing | LLR Gradient |
|---------|---------|--------------|
| Usa | Solo posizioni best | Tutti i vicini + valori f() |
| Robusto al rumore | ✗ No | ✓ Sì |
| Matematicamente | Euristico | Least squares (fondato) |
| Computazione | O(n_attractors) | O(K·dim²) |

### 10.6 Implementazione per ALBA

```python
def estimate_llr_gradient(x: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          n_neighbors: int = None) -> np.ndarray:
    """
    Stima la direzione di discesa via Local Linear Regression.
    
    Args:
        x: Punto corrente
        X: Tutti i punti osservati (N x dim)
        y: Valori obiettivo (N,)
        n_neighbors: Numero vicini (default: 2*dim)
    
    Returns:
        Direzione di discesa normalizzata (dim,)
    """
    dim = len(x)
    if n_neighbors is None:
        n_neighbors = max(2 * dim, dim + 2)
    
    if len(X) < n_neighbors:
        return np.zeros(dim)
    
    # Trova K vicini
    distances = np.linalg.norm(X - x, axis=1)
    nearest_idx = np.argsort(distances)[:n_neighbors]
    
    X_local = X[nearest_idx]
    y_local = y[nearest_idx]
    d_local = distances[nearest_idx]
    
    # Pesi gaussiani
    sigma = np.median(d_local) + 1e-8
    weights = np.exp(-d_local**2 / (2 * sigma**2))
    
    # Centra per stabilità
    X_centered = X_local - x
    y_mean = np.average(y_local, weights=weights)
    y_centered = y_local - y_mean
    
    # Weighted least squares con regularizzazione
    W = np.diag(weights)
    XtWX = X_centered.T @ W @ X_centered + 1e-6 * np.eye(dim)
    XtWy = X_centered.T @ W @ y_centered
    
    gradient = np.linalg.solve(XtWX, XtWy)
    
    # Normalizza e ritorna direzione di discesa
    norm = np.linalg.norm(gradient)
    if norm > 1e-8:
        return -gradient / norm
    return np.zeros(dim)
```

### 10.7 Integrazione in local_search_step()

```python
def local_search_step_with_llr(self, cube, x, observed_X, observed_y):
    # 1. Direzione random
    random_dir = self.rng.standard_normal(self.dim)
    random_dir = random_dir / np.linalg.norm(random_dir)
    
    # 2. Stima LLR gradient
    llr_dir = estimate_llr_gradient(x, observed_X, observed_y)
    
    # 3. Combina (gw=0.5-0.7)
    gw = 0.6
    if np.linalg.norm(llr_dir) > 1e-8:
        direction = (1 - gw) * random_dir + gw * llr_dir
        direction = direction / np.linalg.norm(direction)
    else:
        direction = random_dir
    
    # 4. Opzionalmente aggiungi geo-drift
    if self.geo_drift:
        drift = self.cube_gravity.get_drift_vector(x, all_leaves)
        direction = 0.7 * direction + 0.3 * drift
        direction = direction / np.linalg.norm(direction)
    
    # 5. Step
    step_size = self.rng.uniform(0.3, 1.0) * self.radius
    x_new = x + step_size * direction
    return np.clip(x_new, cube.lo, cube.hi)
```

### 10.8 Parametri Consigliati

| Parametro | Valore | Note |
|-----------|--------|------|
| `gradient_weight` | 0.5-0.7 | 0.7 migliore su funzioni smooth |
| `n_neighbors` | 2 × dim | Minimo dim+2 per fit valido |
| `sigma` | median(distances) | Auto-adattivo |
| Combina con geo-drift | Sì | 0.7 LLR + 0.3 drift |