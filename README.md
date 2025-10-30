# Hyperparameter Optimization via Adaptive QuadTree with PCA-Based Anisotropic Splitting

## Teorema dello Split Basato su Anisotropia e Curvatura

Questo repository implementa un algoritmo di ottimizzazione degli iperparametri basato su QuadTree adattivo che utilizza due principi fondamentali per partizionare lo spazio di ricerca:

### 1. **Split Basato su Anisotropia (PCA)**

#### Principio

L'algoritmo utilizza la **Principal Component Analysis (PCA)** per determinare se la funzione obiettivo mostra una direzione preferenziale nello spazio degli iperparametri.

#### Meccanismo

1. **Calcolo degli Assi Principali**: 
   - Seleziona i migliori punti testati (top 30% per default)
   - Calcola la matrice di covarianza di questi punti
   - Estrae autovalori λ₁ ≥ λ₂ ≥ ... ≥ λₙ e autovettori corrispondenti

2. **Misura dell'Anisotropia**:
   ```
   ratio = λ₁ / mean(λ₂, ..., λₙ)
   ```
   - Se `ratio ≥ threshold` (default 1.4): La funzione è **anisotropica**
   - Altrimenti: La funzione è **isotropica**

3. **Decisione**:
   - **Anisotropia elevata**: Split lungo gli assi PCA (direzioni di massima varianza)
   - **Anisotropia bassa**: Split lungo gli assi originali (coordinate native)

#### Rationale Matematico

Quando λ₁ >> λ₂, ..., λₙ, significa che:
- La funzione obiettivo varia molto più rapidamente lungo il primo componente principale
- Esiste una "valle" o "cresta" dominante nella funzione
- Dividere lungo questa direzione separa efficacemente regioni con performance diverse

**Esempio**: In un problema di tuning learning rate (lr) e momentum (m), se i migliori risultati formano una linea diagonale nel piano (lr, m), PCA identifica questa direzione e lo split avviene lungo di essa.

### 2. **Split Dove la Funzione Curva di Più (Quadratic Fitting)**

#### Principio

Una volta determinata la direzione di split (da PCA o assi originali), il **punto di taglio** è scelto dove la funzione obiettivo presenta la **massima curvatura**.

#### Meccanismo

1. **Proiezione dei Punti**:
   - Proietta tutti i punti testati sull'asse di split scelto
   - Ottiene coppie (t, y) dove t è la coordinata proiettata e y è il valore della funzione

2. **Fit Quadratico**:
   ```
   y ≈ a + b·t + (c/2)·t²
   ```
   - Utilizza ridge regression per robustezza: (ΦᵀΦ + αI)w = Φᵀy

3. **Punto Stazionario**:
   ```
   t* = -b/c
   ```
   - Questo è il punto dove la derivata dy/dt = b + c·t è zero
   - Rappresenta il minimo (se c > 0) o massimo (if c < 0) della parabola

4. **Interpretazione**:
   - **c > 0 (convessa)**: t* è un minimo → split separa le regioni discendenti dai due lati
   - **c < 0 (concava)**: t* è un massimo → split separa le regioni ascendenti/discendenti
   - **|c| grande**: La funzione cambia rapidamente → split efficace
   - **|c| piccolo**: La funzione è quasi lineare → split meno informativo

#### Fallback

Se il fit quadratico fallisce (insufficienti dati, c ≤ 0, o condizionamento numerico):
- Usa la **mediana** delle proiezioni dei migliori punti (top 40%)
- Oppure il **punto medio** dell'intervallo se i dati sono scarsi

#### Esempio Geometrico

Considera una funzione obiettivo 1D: `y = -(x - 0.5)² + 1` (parabola concava centrata in 0.5)

- Il fit quadratico identifica c < 0 (concava) e t* = 0.5
- Lo split avviene in x = 0.5, separando:
  - Regione sinistra (x < 0.5): funzione crescente
  - Regione destra (x > 0.5): funzione decrescente

Questo massimizza la "omogeneità" locale all'interno di ciascuna regione figlia.

## Integrazione dei Due Teoremi

L'algoritmo combina entrambi i principi in modo gerarchico:

### Policy di Split

1. **Verifica Precondizioni**:
   - Profondità massima non superata (`depth < max_depth`)
   - Ampiezza minima rispettata (`width > min_width`)
   - Sufficiente evidenza (`n_trials ≥ min_trials` o `n_points ≥ min_points`)

2. **Scelta Tipo di Split**:
   - Se `n_points ≥ 10` e anisotropia OK → **Quad Split** (4-way lungo PC1 e PC2)
   - Altrimenti → **Binary Split** (2-way lungo asse più largo)

3. **Validazione Split**:
   - Simula lo split e fitta surrogati locali nei figli
   - Calcola riduzione di varianza: Δvar = var_parent - Σ (nᵢ/n_total) · varᵢ
   - Accetta split solo se Δvar ≥ γ (default 0.02)

### Algoritmo di Split Quadruplo (split4)

Quando l'anisotropia è sufficiente:

```
1. Calcola PCA → ottieni R (rotazione) e μ (centro) dai punti migliori
2. Identifica assi PC1 (massima varianza) e PC2 (seconda massima varianza)
3. Per ciascun asse:
   - Proietta i punti: T = Rᵀ(X - μ)
   - Fitta quadratica lungo PC1: trova t₁* (punto di massima curvatura)
   - Fitta quadratica lungo PC2: trova t₂* (punto di massima curvatura)
4. Crea 4 quadranti usando (t₁*, t₂*) come centro di divisione:
   - Q1: t₁ < t₁* AND t₂ < t₂*
   - Q2: t₁ ≥ t₁* AND t₂ < t₂*
   - Q3: t₁ < t₁* AND t₂ ≥ t₂*
   - Q4: t₁ ≥ t₁* AND t₂ ≥ t₂*
5. Assegna i punti storici ai quadranti appropriati
```

## Vantaggi dell'Approccio

### 1. **Adattività Geometrica**
- Non assume assi fissi: si adatta alla geometria intrinseca della funzione
- Particolarmente efficace per funzioni con "valli" o "creste" non allineate agli assi originali

### 2. **Efficienza del Partizionamento**
- Split dove la funzione cambia di più → massima informazione guadagnata
- Evita split inutili in direzioni quasi costanti

### 3. **Robustezza**
- Fallback intelligenti quando PCA o fit quadratico falliscono
- Regularizzazione (ridge) per stabilità numerica
- Soglie conservative per evitare split prematuri

### 4. **Scalabilità**
- Gate di profondità e ampiezza prevengono esplosione esponenziale
- Validazione via riduzione varianza garantisce solo split benefici

## Esempi di Utilizzo

### Caso 1: Funzione Isotropica (Ridge)

Se la funzione obiettivo è una sfera: `f(x) = -||x - c||²`

- PCA trova λ₁ ≈ λ₂ ≈ ... ≈ λₙ → ratio basso → **isotropica**
- Split avviene lungo assi originali (equamente utili)
- Quadratic fitting trova il centro c come punto di massima curvatura

### Caso 2: Funzione Anisotropica (Valle Diagonale)

Se la funzione ha una valle lungo x + y = 1:

```python
f(x, y) = -(x + y - 1)² - 0.1*(x - y)²
```

- PCA trova che PC1 è orientato lungo (1, 1)/√2 (direzione valle)
- λ₁ >> λ₂ → ratio alto → **anisotropica**
- Split avviene lungo PC1 (direzione della valle)
- Quadratic fitting trova il minimo della parabola lungo questa direzione

### Caso 3: Funzione a Sella (Saddle Point)

Per una funzione con punto di sella: `f(x, y) = x² - y²`

- PCA identifica PC1 lungo y (discesa) e PC2 lungo x (ascesa)
- Split lungo PC1 separa la "valle" in y
- Split lungo PC2 separa la "cresta" in x
- Quad split cattura la struttura a sella completa

## Parametri Chiave

### Anisotropia PCA
- `anisotropy_threshold` (default 1.4): Soglia ratio per considerare anisotropia
- `pca_min_points` (default 10): Punti minimi per calcolare PCA affidabile
- `q_good` (default 0.3): Frazione di punti migliori per PCA

### Curvatura Quadratica
- `ridge_alpha` (default 1e-3): Regolarizzazione per fit quadratico
- Fallback a mediana top 40% se fit fallisce

### Split Policy
- `gamma` (default 0.02): Minima riduzione varianza per accettare split
- `max_depth` (default 4): Profondità massima albero
- `min_trials` (default 5): Trial finali minimi prima di split
- `min_points` (default 10): Punti testati minimi per quad split

## Riferimenti Teorici

Questo approccio combina idee da:
- **Bayesian Optimization**: Surrogate modeling locale (quadratico) per guidare splits
- **Adaptive Mesh Refinement**: Split dove l'errore locale è massimo
- **Manifold Learning**: PCA per scoprire struttura intrinseca
- **Decision Trees**: Divisione binaria/quad per partizionamento gerarchico

## Implementazione

Vedere `HPO_QuadTree_v1.py` per l'implementazione completa con:
- `_principal_axes()`: Calcolo PCA e anisotropia
- `_quad_cut_along_axis()`: Fit quadratico e punto di massima curvatura
- `should_split()`: Policy integrata di split
- `split2()` / `split4()`: Esecuzione split binario/quad con redistribuzione punti

---

**Autore**: Implementazione per tesi su Hyperparameter Optimization  
**Licenza**: Da specificare
