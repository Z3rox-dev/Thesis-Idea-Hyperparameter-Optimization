# ALBA — esposizione dell’algoritmo (stile tesi PhD)

Questa cartella contiene un’implementazione di ricerca del framework **ALBA** (Adaptive Local Bayesian Algorithm), pensata per ottimizzazione black-box e, in particolare, per **Hyperparameter Optimization (HPO)** in spazi misti continui/categoriali.

L’obiettivo di questo documento è descrivere l’algoritmo con un livello di formalizzazione adatto a una tesi di dottorato: prima una panoramica end‑to‑end, poi la scomposizione per componenti (con riferimenti puntuali all’implementazione nelle sezioni successive).

## Overview (visione end‑to‑end, senza dettagli implementativi)

ALBA è un algoritmo di **ottimizzazione black‑box sequenziale**: genera una sequenza di punti $\{x_t\}_{t=1}^B$ nel dominio $\mathcal{X}$, osserva $y_t=f(x_t)$ e aggiorna la propria strategia usando esclusivamente la storia delle valutazioni. L’obiettivo è trovare, entro un budget finito $B$, un incumbent $x^\star$ che (in minimizzazione) renda $f(x^\star)$ il più piccolo possibile.

L’intuizione guida può essere descritta con una metafora geometrica: lo spazio di ricerca è un paesaggio ad alta dimensionalità. ALBA evita di costruire una “mappa globale” unica (costosa e fragile in alta‑D) e sceglie invece di **cartografare localmente** il paesaggio in molte regioni, stimare in ciascuna una direzione di discesa, e poi **ricomporre** queste informazioni locali in un segnale globale che indirizzi il campionamento.

**Contributo distintivo (in una frase).** ALBA costruisce un segnale globale “economico” a partire da modelli locali (un **campo di potenziale**) e stima quando questi modelli sono affidabili (una **coerenza geometrica**); insieme, questi due segnali modulano in modo probabilistico il compromesso esplorazione/sfruttamento senza richiedere un surrogate globale monolitico.

### 0.1 Rappresentazione dello spazio (continui + categoriali)

ALBA opera su una rappresentazione vettoriale $x \in \mathbb{R}^d$ soggetta a vincoli per‑dimensione. Variabili categoriali e discrete vengono trattate tramite un’**encoding** che consenta sia:
1) campionamento nello spazio continuo di lavoro, sia  
2) proiezione coerente su un insieme finito di scelte.

L’algoritmo mantiene statistiche specifiche per queste dimensioni, poiché esplorare combinazioni discrete richiede segnali diversi rispetto a quelli per variabili continue.

### 0.2 La struttura: partizionamento adattivo in regioni

Il dominio $\mathcal{X}$ viene concettualmente rappresentato come una regione “radice” che viene **suddivisa ricorsivamente**: ogni suddivisione genera due (o più, in generale) sotto‑regioni figlie. Ripetendo questo processo si ottiene un **albero di partizione**, in cui le foglie (leaves) rappresentano la discretizzazione corrente dello spazio.

Durante l’ottimizzazione, ALBA mantiene quindi una partizione del dominio in un insieme di regioni (celle) $\{C_i\}_{i=1}^K$, ognuna iper‑rettangolare. Per ogni cella $C_i$ si mantiene:
- un sottoinsieme di osservazioni $\mathcal{D}_i=\{(x,s)\}$ assegnate alla cella;
- statistiche riassuntive (numero di punti, miglior valore locale, ecc.);
- un modello locale (surrogato) che descrive il comportamento della funzione *all’interno* della cella.

L’idea è allocare capacità di modellazione dove arrivano dati: regioni promettenti vengono raffinate (splitting), regioni poco esplorate restano più grosse.

### 0.3 “Punti buoni” e soglia dinamica $\gamma_t$

Una volta che lo spazio è suddiviso in celle, ALBA deve allocare il budget tra regioni diverse. Farlo usando i valori grezzi di $f$ (o anche dello score $s$) può essere fragile: la scala può essere poco informativa nelle prime iterazioni, cambiare drasticamente nel tempo, o essere dominata da rumore e outlier. Per questo ALBA introduce un concetto *relativo* di successo: un punto è “buono” se rientra tra i migliori osservati fino a quel momento.

Si introduce quindi una soglia dinamica $\gamma_t$ (dipendente da $t$) costruita dalla storia degli score $\{s_\tau\}_{\tau\le t}$ (ad esempio come un quantile che può variare con $t$), e si definisce:
$$
z_t = \mathbb{I}[s_t \ge \gamma_t], \qquad \text{“good point”} \iff z_t=1,
$$
dove $s_t$ è lo score interno del punto $x_t$, definito in modo da avere **massimizzazione interna** anche quando il problema originario è di minimizzazione. La soglia è *globale* (comune a tutte le celle): un “successo” ha quindi lo stesso significato ovunque nel dominio.

Questa etichetta “good” svolge tre ruoli complementari:
1) **normalizza** la nozione di qualità nel tempo (è una misura d’ordinamento relativo), rendendo più stabile il confronto tra valutazioni anche quando la scala di $f$ è instabile;  
2) trasforma ogni valutazione in un esito Bernoulli (successo/insuccesso) aggregabile per cella, che alimenta una stima della “probabilità di successo” locale;  
3) fornisce un’ancora robusta per le scelte strutturali (splitting e statistiche su categoriali), perché consente di ragionare su “dove si concentrano i punti buoni” più che su valori assoluti.

### 0.4 Decisione a due livelli: scegliere *dove* e scegliere *cosa*

Ogni iterazione può essere vista come una decisione a due livelli:

**(A) Selezione della cella (dove campionare).**  
Ogni cella viene trattata come un “braccio” con una probabilità latente di produrre un esito $z=1$, cioè un *good point* secondo la soglia $\gamma_t$. Ogni valutazione assegnata alla cella aggiorna conteggi successi/insuccessi e, di conseguenza, una posterior Beta‑Bernoulli; campionando da questa posterior in stile Thompson Sampling (e aggiungendo un bonus per celle poco esplorate) si ottiene una selezione stocastica ma biasata verso regioni promettenti.

**(B) Campionamento nella cella (che punto scegliere).**  
Dentro la cella selezionata, ALBA alterna:
- **sfruttamento**: propone punti guidati da un surrogato locale e seleziona con una regola tipo Upper Confidence Bound (mean + bonus d’incertezza), ad esempio
  $$
  a(x)=\mu(x)+\beta\,\sigma(x);
  $$
- **esplorazione**: propone punti più “neutri” (casuali o debolmente strutturati) per mantenere copertura e prevenire over‑exploitation di stime locali rumorose.

### 0.5 Surrogati locali: stimare una direzione e un’incertezza

In ogni cella, il surrogato locale è intenzionalmente semplice: un modello lineare pesato (ridge) in coordinate normalizzate della cella. Questo produce:
- un vettore direzionale che indica come variare $x$ per aumentare lo score atteso;
- una stima di incertezza che cresce quando ci si allontana dalle regioni meglio osservate o quando il fit è instabile.

Questo tipo di surrogate non pretende di rappresentare la funzione globalmente: serve solo come *strumento locale* per generare candidati sensati.

### 0.6 Coerenza geometrica: quando fidarsi dei gradienti locali

In alta dimensionalità, un gradiente locale stimato con pochi punti può essere rumoroso o addirittura fuorviante. ALBA introduce quindi una misura di **coerenza**:

- si costruisce un grafo di vicinanza tra celle (tipicamente k‑nearest neighbors sui centri);
- si misura quanto le direzioni stimate in celle vicine sono allineate;
- se l’allineamento è basso (campo “non conservativo” / incoerente), ALBA riduce l’uso del surrogato e aumenta l’esplorazione.

La coerenza diventa quindi un gating data‑driven: non è un parametro arbitrario ma un modo per valutare la qualità geometrica delle informazioni locali.

### 0.7 Campo di potenziale globale: ricomporre l’informazione locale

Oltre alla coerenza, ALBA ricostruisce un **campo scalare** (potenziale) sulle celle che funge da “mappa energetica” globale:
- ad ogni arco $(i,j)$ del grafo si associa una variazione prevista lungo la direzione locale stimata nella cella $i$;
- si risolve un problema di minimi quadrati che trova valori $\{u_i\}$ coerenti con queste differenze lungo tutti gli archi, ad esempio
  $$
  \min_{u\in\mathbb{R}^K}\sum_{(i,j)\in E} w_{ij}\,\big(u_j-u_i-d_{ij}\big)^2,
  \qquad d_{ij}\approx g_i^\top(c_j-c_i);
  $$
- i valori $u_i$ vengono normalizzati in $\phi_i \in [0,1]$ con semantica:
  $$
  \phi_i \approx 0 \Rightarrow \text{cella promettente}, \qquad
  \phi_i \approx 1 \Rightarrow \text{cella poco promettente}.
  $$

Questo potenziale non sostituisce lo score reale, ma fornisce un segnale globale (economico) per decidere *quanto aggressivamente* sfruttare una cella: ALBA rende lo sfruttamento più probabile dove il potenziale è basso e più raro dove è alto.

### 0.8 Gestione delle variabili categoriali (curiosità + posteriori)

Per dimensioni categoriali, ALBA mantiene conteggi per categoria (a livello di cella) e un tracciamento globale delle combinazioni visitate. Il campionamento discreto combina:
- posteriori Beta (Thompson) per preferire categorie che sembrano produrre score migliori;
- un termine di “curiosità” che premia combinazioni rare/non viste;
- una dinamica tipo crossover tra configurazioni élite per ricombinare scelte discrete promettenti.

### 0.9 Adattamento nel tempo: esplorazione, splitting, refinement

Il comportamento di ALBA evolve durante il budget:
- nella prima fase si privilegia la copertura e la costruzione di struttura (splitting + surrogate locali + segnali globali);
- nella fase finale si aumenta il refinement attorno all’incumbent con perturbazioni locali a raggio decrescente e, opzionalmente, brevi episodi di ricerca locale “aggressiva” quando emerge un nuovo best.

### 0.10 Pseudocodice concettuale

```text
Input: dominio X, budget B
Init: partizione iniziale in 1 cella; storia D0 := ∅; incumbent x* := None

for t = 1..B:
  Aggiorna soglia γ_t (definizione di “good”) dalla storia corrente
  Aggiorna statistiche good/bad per cella
  Seleziona una cella C_t con Thompson Sampling + bonus esplorativo (good/bad)
  (periodicamente) aggiorna coerenza e potenziale su grafo tra celle

  Dentro C_t:
    calcola probabilità di sfruttamento usando coerenza e potenziale
    con tale probabilità:
      genera candidati; valuta UCB con surrogate locale; scegli un candidato
    altrimenti:
      esplora con campioni neutrali nella cella
    applica un campionamento dedicato alle variabili categoriali

  osserva y_t = f(x_t) e calcola score s_t
  aggiorna D e incumbent x*
  aggiorna la cella con (x_t, s_t) e l’esito z_t = I[s_t >= γ_t]; rifitta il surrogato locale
  se la cella ha abbastanza dati: splitta e aggiorna la partizione

Output: incumbent x*
```

---

## 1. Problema, convenzioni e notazione

Consideriamo il problema di ottimizzazione black-box:

$$
\min_{x \in \mathcal{X}} f(x)
$$

dove $\mathcal{X}$ è un dominio iper-rettangolare (eventualmente con dimensioni categoriali codificate) e la valutazione di $f(\cdot)$ è costosa.

### 1.1 Massimizzazione interna

L’implementazione adotta la convenzione “**higher-is-better**” internamente:
- se `maximize=False` (default), ALBA trasforma $y = f(x)$ in **fitness** $s(x) = -y$;
- se `maximize=True`, allora $s(x)=y$.

Questa trasformazione è realizzata in `optimizer.py` tramite:
- `ALBA._to_internal(y_raw)` e `ALBA._to_raw(y_internal)`.

Nel seguito useremo:
- $s_i$ per indicare lo score interno (da massimizzare),
- $y_i$ per indicare il valore originale dell’obiettivo.

### 1.2 Spazio e rappresentazione dei parametri

ALBA opera su un vettore $x \in \mathbb{R}^d$ entro bounds per-dimensione:
$$
 x_k \in [\ell_k, u_k].
$$

Due modalità d’uso:
1. **`bounds=`**: l’utente fornisce direttamente intervalli reali (`optimizer.py`).
2. **`param_space=`**: l’utente fornisce specifiche “tipizzate”; la codifica/decodifica verso uno spazio interno normalizzato è gestita da `ParamSpaceHandler` (`param_space.py`).

In modalità `param_space`, gli intervalli interni sono sempre $[0,1]$ e le dimensioni categoriali sono codificate come coordinate reali in $[0,1]$.

---

## 2. Architettura generale: “Think locally, act globally”

ALBA combina quattro ingredienti:

1. **Partizionamento adattivo** dello spazio in regioni iper-rettangolari (“cube leaves”) tramite un albero tipo k-d tree (`cube.py`, `splitting.py`).
2. **Surrogati locali** per ogni leaf: **Local Gradient Surrogate (LGS)**, un modello lineare pesato che stima una direzione di gradiente e un’incertezza tipo bayesiana (`lgs.py`).
3. **Campo di potenziale globale** ricostruito integrando (in senso least-squares) il campo di gradienti locali su un grafo kNN dei centri delle leaves (`coherence.py`).
4. **Politica di campionamento** che modula la probabilità di sfruttamento vs esplorazione in funzione di:
   - affidabilità geometrica delle stime (coerenza),
   - “energia” globale della regione (potenziale),
   - dinamiche di stagnazione, presenza di categoriche, ecc. (`optimizer.py`, `leaf_selection.py`, `categorical.py`).

Il ciclo operativo è orchestrato dalla classe `ALBA` in `optimizer.py` con interfaccia `ask()/tell()`.

---

## 3. Partizionamento adattivo in cubes

### 3.1 La struttura `Cube`

Ogni leaf $C$ (classe `Cube` in `cube.py`) memorizza:
- bounds per dimensione,
- numero di valutazioni `n_trials`,
- conteggio “buone” `n_good` rispetto a una soglia dinamica $\gamma$ (Sez. 6),
- miglior punto locale `best_x` e miglior score locale,
- lista `tested_pairs = {(x_i, s_i)}`,
- modello LGS `lgs_model` (opzionale),
- statistiche categoriali `cat_stats` per le dimensioni discrete.

### 3.2 Quando si splitta (SplitDecider)

La decisione “quando splittare” è separata dalla modalità “come splittare”:
- **quando**: `ThresholdSplitDecider` (`splitting.py`) impone soglie su:
  - minimo numero di trial,
  - profondità massima dell’albero,
  - soglia di split in funzione della dimensionalità (tramite `n_trials >= split_trials_factor * dim + split_trials_offset`).
- **come**: `CubeIntrinsicSplitPolicy` delega a `Cube.split()` (`cube.py`).

L’implementazione in `optimizer.py` applica anche *scaling laws* di default (Sez. 11) quando `split_depth_max` o `split_trials_factor` non sono specificati.

### 3.3 Asse e punto di split

`Cube.get_split_axis()` (`cube.py`) seleziona l’asse combinando:
- controllo dell’aspect ratio (evitare cubes troppo allungati),
- regole conservative per split “early”,
- uso della direzione di gradiente LGS se sufficientemente dominante,
- fallback su ampiezza o varianza dei punti migliori.

Dato l’asse $k$, `Cube.split()` sceglie il cut tramite:
1. **mediana pesata** delle coordinate dei punti “buoni” $s_i \ge \gamma$ (peso proporzionale a $s_i-\gamma$),
2. in caso di pochi punti buoni, media,
3. altrimenti midpoint.

È applicato un margine per evitare cuts degeneri vicino ai bordi.

---

## 4. Local Gradient Surrogate (LGS)

### 4.1 Modello e normalizzazione locale

Il modello LGS viene fittato in `lgs.py` (funzione `fit_lgs_model`), usando coordinate normalizzate rispetto alla leaf:

$$
z_i = \frac{x_i - c}{w}, \qquad c=\text{center}(C),\; w=\text{widths}(C),
$$

dove la divisione è elemento per elemento.

Si standardizza lo score interno:
$$
\tilde{s}_i = \frac{s_i - \mu_s}{\sigma_s}.
$$

### 4.2 Regressione pesata + regolarizzazione

Si usa una regressione lineare pesata (ridge) per stimare un vettore $g$:

$$
g = \arg\min_g \sum_i w_i\,( \tilde{s}_i - z_i^\top g )^2 + \lambda \lVert g \rVert^2
$$

con pesi $w_i$ dati da:
- una componente gaussiana in funzione della distanza $\lVert z_i\rVert^2$,
- un “rank weight” che favorisce punti con score più alto.

In forma chiusa:

$$
g = (Z^\top W Z + \lambda I)^{-1} Z^\top W \tilde{s}.
$$

`lgs.py` implementa:
- regolarizzazione adattiva in funzione di $d$ e del numero di punti,
- check del condizionamento per aumentare $\lambda$ in caso di instabilità,
- “parent backfill”: se una leaf ha pochi punti, recupera campioni dal padre (solo se contenuti nella leaf) per stabilizzare il fit.

### 4.3 Predizione (media + incertezza)

La predizione per un candidato $x$ usa:

$$
\mu(x) = \mu_s + (z^\top g)\,\sigma_s
$$

e una varianza tipo bayesiana:

$$
\sigma^2(x) = \sigma_s^2 \cdot \nu \cdot \left(1 + z^\top (Z^\top W Z + \lambda I)^{-1} z\right),
$$

dove $\nu$ è una stima della varianza residua (in spazio normalizzato).

In codice:
- fit: `fit_lgs_model(...)`
- predizione: `predict_bayesian(model, candidates)`.

---

## 5. Generazione candidati e acquisizione (UCB-softmax)

### 5.1 Candidate generation (mixture)

`MixtureCandidateGenerator` (`candidates.py`) genera candidati in una leaf con una mistura di strategie:
- perturbazione di punti top-k del modello,
- step lungo la direzione di gradiente (con rumore),
- perturbazione del centro,
- uniform sampling nei bounds della leaf.

### 5.2 Selezione del candidato: UCB + softmax

`UCBSoftmaxSelector` (`acquisition.py`) seleziona un candidato tra $n$ punti con una UCB:

$$
\text{UCB}(x) = \mu(x) + \beta\,\sigma(x), \qquad \beta = 2 \cdot \texttt{novelty\_weight}.
$$

Per mantenere stocasticità (utile contro overfitting locale e per robustezza), si applica un softmax sulla UCB standardizzata (z-score) con temperatura.

Questa scelta implementa un compromesso pratico:
- i candidati migliori restano più probabili,
- ma rimane una massa non nulla su candidati “vicini”.

---

## 6. Soglia dinamica $\gamma$ e definizione di “punti buoni”

ALBA usa una soglia $\gamma$ per etichettare “good points” ($s_i \ge \gamma$) e guidare:
- statistica `n_good` per leaf,
- split cut (mediana pesata sui buoni),
- statistiche categoriali per leaf.

La schedulazione è in `gamma.py` (`QuantileAnnealedGammaScheduler`):
- per pochi dati, $\gamma \leftarrow 0$,
- altrimenti $\gamma$ è un percentile di `y_all` (interno), con quantile che anneala dall’inizio dell’esplorazione verso un target finale.

Intuizione: nelle fasi iniziali la selezione dei “buoni” deve essere più conservativa/robusta, poi può diventare più permissiva.

---

## 7. Selezione della leaf (policy gerarchica)

Durante i passi “tree-guided” (non global-random e non puro local search), ALBA seleziona una leaf $C$.

Per default, in `optimizer.py`, la policy è `ThompsonSamplingLeafSelector` (`leaf_selection.py`):
- si campiona una probabilità di successo dalla posterior Beta:
  $$
  p_C \sim \text{Beta}(n_\text{good}+1,\; n_\text{bad}+1),
  $$
  dove $n_\text{bad}=n_\text{trials}-n_\text{good}$.
- si aggiunge un bonus di esplorazione decrescente con $n_\text{trials}$,
- si aggiunge un bonus se la leaf ha un modello LGS “sufficientemente informato”.

Infine si seleziona via softmax (temperatura diversa se in stagnazione).

Nota: esiste anche `PotentialAwareLeafSelector` (stesso file) per incorporare esplicitamente il potenziale nella selezione della leaf (attivabile passando `leaf_selector=`).

---

## 8. Coerenza geometrica e campo di potenziale globale

Questa sezione descrive il contributo del campo di potenziale globale.

### 8.1 Grafo kNN sui centri delle leaves

Dato l’insieme delle leaves $\{C_i\}_{i=1}^K$, si calcolano i centri $c_i$ e si costruisce un grafo kNN con $k=6$ (`coherence.py`, `_build_knn_graph`).

Le distanze sono normalizzate per range globali per evitare dominanza di una dimensione.

### 8.2 Predizioni locali lungo gli archi

Per un arco diretto $(i,j)$, si considera:
- il delta tra centri $\Delta_{ij}=c_j-c_i$,
- il gradiente locale (in spazio normalizzato) $g_i$ stimato da LGS.

Si mascherano le dimensioni categoriali (gradiente e delta posti a zero per quelle coordinate).

Si calcolano due quantità:
1. **drop predetto normalizzato**
   $$
   d_{ij} = \langle \hat{g}_i,\; \widehat{\Delta}_{ij} \rangle
   $$
   dove i cappelli indicano vettori unitari (scale-invariant);
2. **allineamento tra gradienti**
   $$
   a_{ij} = \langle \hat{g}_i,\; \hat{g}_j \rangle \in [-1,1].
   $$

### 8.3 Coerenza come conservatività locale

L’idea: un campo di gradienti “ben comportato” dovrebbe essere localmente consistente.

La **coerenza di una leaf** è calcolata come media degli allineamenti $a_{ij}$ degli archi incidenti, rimappata in $[0,1]$:

$$
\text{coh}(i)=\frac{1+\mathbb{E}[a_{ij}]}{2}.
$$

Il tracker (`CoherenceTracker`) stima inoltre soglie percentili (Q60 e Q80) per un gating data-driven.

### 8.4 Ricostruzione del potenziale via least-squares

Si cercano potenziali scalari $\{u_i\}$ che rispettino, sui bordi del grafo, le differenze previste dai gradienti:

$$
u_j - u_i \approx d_{ij}.
$$

In pratica si risolve un problema LS pesato:

$$
\min_u \sum_{(i,j)\in E} w_{ij}\,(u_j-u_i-d_{ij})^2
$$

con gauge fixing $u_0=0$ (risolto con LSQR su matrice sparsa; `coherence.py`, `_solve_potential_least_squares`).
I pesi $w_{ij}$ derivano dalla qualità di allineamento (più allineati → più affidabili).

### 8.5 Correzione del segno (dettaglio critico)

Dato che ALBA ottimizza *internamente* lo score $s=-f$ (nel caso di minimizzazione), il gradiente stimato da LGS è, in generale:

$$
\nabla s(x) = -\nabla f(x),
$$

quindi punta **verso** il minimo (invece che “allontanarsene” come $\nabla f$). Per garantire la semantica:

> **basso potenziale = regione buona / vicina al minimo**

l’implementazione inverte il potenziale ricostruito (`u_inverted = -u`) e lo combina con un segnale empirico di densità (Sez. 8.6).

### 8.6 Bonus di densità e normalizzazione finale

Il potenziale viene reso più robusto combinandolo con un segnale empirico:

$$
\text{dens}(i) = \frac{n_\text{good}(i)}{\text{vol}(C_i)}.
$$

Si normalizza la densità in $[0,1]$ e la si usa come “bonus” (alta densità → potenziale più basso).

Infine:
- si ancora il potenziale ponendo a zero la leaf “migliore”,
- si normalizza in $[0,1]$,
- in caso di potenziale non informativo (varianza troppo bassa) si ripiega sulla sola densità.

La cache viene aggiornata periodicamente (configurabile via `coherence_update_interval` nel costruttore di `ALBA`).

---

## 9. Campionamento modulato dal potenziale (componente globale di ALBA)

Il campionamento dentro una leaf avviene in `ALBA._sample_in_cube()` (`optimizer.py`).

### 9.1 Gating con coerenza e potenziale

Dopo un warmup iniziale (quando `_sample_in_cube()` è invocata e `self.iteration < 15`, la proposta è uniform nei bounds della leaf), per una leaf $C$ si calcola:
- potenziale normalizzato $\phi(C)\in[0,1]$ (se `use_potential_field=True`, altrimenti $\phi(C)=0.5$ come caso neutro),
- una decisione binaria di “coerenza sufficiente” `coherence_ok` (se `use_coherence_gating=True`).

Dettaglio implementativo: il potenziale grezzo da `CoherenceTracker.get_potential(...)` viene anche **smorzato** verso $0.5$ in funzione della `global_coherence` (se bassa, il potenziale è considerato poco affidabile e quindi pesa poco).

La probabilità di sfruttamento (uso di LGS + acquisizione) è:

$$
p_\text{exploit}(\phi)=0.95-0.65\,\phi \;\;\in[0.30, 0.95].
$$

Se la coerenza è bassa, questa probabilità viene **cappata** a 0.5 (massimo 50% di sfruttamento), forzando più esplorazione quando le stime di gradiente sono ritenute inaffidabili.

### 9.2 Exploit: campionamento guidato da LGS

Se si sceglie exploit, si usa:
- `CandidateGenerator.generate(...)` per generare candidati,
- `Cube.predict_bayesian(...)` per $(\mu,\sigma)$,
- `AcquisitionSelector.select(...)` per scegliere il punto finale.

### 9.3 Explore: strategie semplici ma robuste

Se si sceglie explore (o se LGS non è pronto), si usa `_sample_explore_mode()` (`optimizer.py`) con una mistura:
- uniforme nei bounds della leaf,
- sampling attorno al centro con jitter,
- perturbazione di `best_x` locale se disponibile.

### 9.4 Gestione delle variabili categoriali

Dopo aver proposto un candidato continuo, `CategoricalSampler.sample(...)` (`categorical.py`) applica:
- elite crossover con probabilità adattiva (più alta in stagnazione),
- Thompson sampling per categoria usando statistiche per-leaf,
- scelta curiosa (bonus inverso alle visite) su più candidati categoriali.

Questo step è cruciale per evitare che l’ottimizzazione in spazi misti collassi in pochi pattern categoriali.

---

## 10. Fase di local search e “drilling” (refinement)

ALBA divide il budget in:
- **exploration phase**: primi `exploration_budget` step,
- **local search phase**: ultimi `local_search_budget` step (`local_search_ratio`).

Durante la seconda fase:
- con probabilità crescente (da 0.5 fino a ~0.9) si campiona attorno al best globale con un local search sampler (`local_search.py`),
- altrimenti si continua con passi tree-guided (utile per non perdere regioni alternative).

Opzionalmente, ALBA può attivare `DrillingOptimizer` (`drilling.py`) quando viene trovato un nuovo best (dopo una warmup e con un budget massimo dedicato): è un raffinamento iterativo stile **(1+1)-CMA-ES semplificato** che “scava” localmente attorno all’incumbent.

---

## 11. Scaling laws implementate nei default (motivazione empirica)

Per controllare la degenerazione in alta dimensione (dove ogni leaf deve contenere abbastanza punti per fittare un LGS stabile), `optimizer.py` imposta alcuni default adattivi:

1. **Massima profondità vs dimensionalità**:
   - se `split_depth_max is None`: `max(4, int(40 / dim))`.
2. **Soglia di split vs budget e dimensionalità**:
   - se `split_trials_factor is None`: base più alto in alta-D e scaling logaritmico col budget:
     - `base_factor = 6` se `dim > 10`, altrimenti `3`;
     - `budget_scale = log(1 + total_budget/500)`;
     - `split_trials_factor = base_factor * max(1, budget_scale)`.

Queste scelte mirano a evitare l’“endless split paradox”: splits troppo frequenti mantengono la densità per leaf costante e insufficiente, rendendo il gradiente locale rumoroso e impedendo di sfruttare davvero l’aumento di budget.

---

## 12. Complessità computazionale (stima)

Sia $K$ il numero di leaves e $d$ la dimensionalità.

- **Fit LGS per leaf**: $O(n_C d^2 + d^3)$ (inversione di matrice $d\times d$), dove $n_C$ è il numero di punti nella leaf (o backfill).
- **Predizione su $m$ candidati**: $O(m d^2)$ (termine $C_{\text{norm}}\,\text{inv\_cov}$).
- **Selezione leaf**: $O(K)$.
- **Coerenza/potenziale** (naive kNN): $O(K^2 d)$ per distanze + LSQR su grafo sparso ($|E|\approx Kk$).

Nel regime tipico (budget moderati e controllo della profondità), $K$ resta gestibile; l’obiettivo progettuale è evitare il costo $O(N^3)$ dei surrogate globali (es. GP) mantenendo guidance globale.

---

## 13. Pseudocodice (ask/tell)

```text
Input: bounds or param_space, budget B, parametri di split, novelty_weight, ...
Init: root cube C0 := bounds, leaves := {C0}, history := ∅, gamma := 0

for t = 1..B:
  if drilling attivo:
     x := driller.ask()
  else if rand() < global_random_prob:
     x := Uniform(bounds)
  else if t <= exploration_budget:
     gamma := GammaSchedule(history, t)
     if t mod 5 == 0:
        Fit LGS su tutte le leaves
        Update CoherenceTracker (cache coerenza/potenziale)
     C := LeafSelector(leaves, stagnation)
     x := SampleInCube(C)    # (potenziale/coerenza → exploit vs explore)
  else:
     con prob p(progress): x := LocalSearch(best_x)
     altrimenti:           x := tree-guided come sopra

  y := f(x)  (o evaluate(config) in modalità param_space)
  tell(x, y):
    aggiorna best e stagnation
    (opz.) attiva drilling su nuovo best
    assegna x alla leaf contenente C
    aggiorna statistiche e cat_stats
    fit LGS per C
    se SplitDecider(C): split C e aggiorna leaves (+ coherence cache)

Output: best_x, best_y
```

---

## 14. Mappa dei file (per lettura della tesi “dal codice”)

- `optimizer.py`: classe `ALBA`, loop `ask()/tell()`, due fasi, split, integrazione coerenza/potenziale.
- `cube.py`: struttura dati `Cube`, statistiche, split axis/cut, fit/predict del modello locale.
- `lgs.py`: matematica del surrogate locale (regressione pesata + incertezza).
- `coherence.py`: kNN graph, coerenza, ricostruzione del potenziale (LSQR), cache e gating.
- `leaf_selection.py`: politiche di selezione leaf (UCB-softmax, Thompson, potential-aware).
- `candidates.py`: generazione candidati (mixture).
- `acquisition.py`: acquisizione UCB + softmax.
- `categorical.py`: handling delle dimensioni categoriali (Thompson + curiosity + crossover).
- `local_search.py`: campionatori di local search (gaussiano o covariance-adaptive).
- `drilling.py`: raffinamento locale (1+1)-CMA-ES semplificato.
- `gamma.py`: scheduler della soglia $\gamma$.

---

## 15. Esempio d’uso minimale

Nota pratica (import): nel repository il package vive in `thesis/alba_framework_potential/`. Per importarlo come `alba_framework_potential`, esegui gli script dalla cartella `thesis/` oppure imposta `PYTHONPATH=thesis`.

### 15.1 Modalità `param_space`

```python
from alba_framework_potential import ALBA

param_space = {
    "learning_rate": (1e-4, 1e-1, "log"),
    "hidden_size": (32, 512, "int"),
    "activation": ["relu", "tanh", "gelu"],
}

opt = ALBA(param_space=param_space, maximize=False, seed=42, use_potential_field=True)
for _ in range(100):
    cfg = opt.ask()
    loss = train_and_eval(**cfg)
    opt.tell(cfg, loss)

print(opt.best_y, opt.decode(opt.best_x))
```

### 15.2 Modalità `bounds`

```python
from alba_framework_potential import ALBA

bounds = [(0.0, 1.0)] * 10
opt = ALBA(bounds=bounds, maximize=False, seed=0, use_potential_field=True)

for _ in range(200):
    x = opt.ask()
    y = objective(x)
    opt.tell(x, y)
```
