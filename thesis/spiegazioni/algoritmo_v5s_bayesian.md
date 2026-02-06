# HPOptimizerV5s: Algoritmo Ibrido Bayesiano con Trust Region

Questo documento descrive il funzionamento dell'algoritmo `HPOptimizerV5s` (implementato in `hpo_v5s_more_novelty_standalone.py`), una soluzione avanzata per l'ottimizzazione degli iperparametri che combina partizionamento dello spazio, modelli surrogati locali e regressione Bayesiana.

## 1. Panoramica
L'algoritmo è progettato per risolvere problemi di ottimizzazione "Black-Box" costosi, dove non si conosce la forma della funzione obiettivo e ogni valutazione richiede tempo.

I pilastri principali sono:
1.  **Partizionamento Gerarchico (Cubes)**: Divide lo spazio di ricerca in sottoregioni ("Cubi") per gestire la complessità e la non-convessità.
2.  **Local Gradient Surrogate (LGS)**: Ogni cubo apprende un modello locale del gradiente per guidare la ricerca.
3.  **Regressione Lineare Bayesiana**: Stima non solo la direzione di miglioramento ($\mu$) ma anche l'incertezza ($\sigma$) per bilanciare esplorazione ed estrazione.
4.  **Trust Region Weighting**: Pesa le osservazioni in base alla distanza dal centro del cubo, rendendo il modello lineare robusto anche in spazi curvi.

---

## 2. Componenti Chiave

### 2.1 Partizionamento dello Spazio (Class `Cube`)
L'algoritmo inizia con un singolo cubo che copre l'intero spazio di ricerca. Man mano che vengono raccolti dati:
-   Se un cubo accumula abbastanza valutazioni (`n_trials`), viene diviso in due figli.
-   Il taglio avviene lungo la dimensione più promettente (basata sul gradiente stimato) o quella più larga.
-   Questo approccio "Divide et Impera" permette di isolare le regioni promettenti e di adattare la scala della ricerca.

### 2.2 Local Gradient Surrogate (LGS) con Trust Region
All'interno di ogni cubo, viene addestrato un modello lineare locale per approssimare la funzione obiettivo.

**Problema**: I modelli lineari falliscono se lo spazio è molto curvo o vasto.
**Soluzione (Trust Region)**:
Invece di usare tutti i punti con peso uguale, l'algoritmo applica un peso esponenziale decrescente basato sulla distanza dal centro del cubo:
$$ w_i = \exp\left(-\frac{\|x_i - c\|^2}{2\sigma^2}\right) $$
Dove $c$ è il centro del cubo e $\sigma^2$ è la distanza media dei punti.
Questo focalizza il modello sulla regione locale, ignorando i punti lontani che potrebbero "inquinare" la stima del gradiente.

### 2.3 Regressione Bayesiana (Weighted Ridge Regression)
Per stimare la direzione di discesa e l'incertezza, usiamo una Ridge Regression pesata:

1.  **Stima dei Coefficienti (Gradiente)**:
    $$ \beta = (X^T W X + \lambda I)^{-1} X^T W y $$
    Dove $W$ è la matrice diagonale dei pesi della Trust Region.

2.  **Stima dell'Incertezza (Varianza)**:
    La varianza predittiva per un nuovo punto $x$ è data da:
    $$ \sigma^2(x) = \sigma_n^2 (1 + x^T (X^T W X + \lambda I)^{-1} x) $$
    Questo ci dice quanto siamo "sicuri" della predizione in quel punto. Punti lontani dai dati osservati avranno un'incertezza maggiore.

### 2.4 Funzione di Acquisizione (UCB)
Per scegliere il prossimo punto da valutare all'interno di un cubo, l'algoritmo genera `n_candidates` e sceglie il migliore usando un criterio Upper Confidence Bound (UCB):
$$ \text{Score}(x) = \mu(x) + \beta \cdot \sigma(x) $$
-   $\mu(x)$: Miglioramento previsto (sfruttamento).
-   $\sigma(x)$: Incertezza del modello (esplorazione).
-   $\beta$: Fattore di bilanciamento (legato al parametro `novelty_weight`).

Questo permette all'algoritmo di esplorare zone ignote del cubo se il modello è incerto, o di convergere rapidamente se la direzione è chiara.

---

## 3. Flusso di Esecuzione (`ask` & `tell`)

1.  **Selezione del Cubo**: L'algoritmo sceglie quale cubo esplorare basandosi su:
    -   Tasso di successo storico (`good_ratio`).
    -   Termine di esplorazione (per non trascurare cubi poco visitati).
    -   Bonus se il cubo ha un modello LGS valido.

2.  **Campionamento (Ask)**:
    -   Genera candidati nel cubo (casuali, perturbazioni dei migliori, o lungo il gradiente).
    -   Valuta i candidati con il modello Bayesiano (UCB).
    -   Restituisce il candidato con lo score più alto.

3.  **Aggiornamento (Tell)**:
    -   Riceve il vero valore della funzione obiettivo.
    -   Aggiorna i dati del cubo.
    -   Ricalcola il modello LGS (con i nuovi pesi Trust Region).
    -   Se necessario, divide il cubo in due.

4.  **Fase Finale (Local Search)**:
    -   Verso la fine del budget, l'algoritmo passa a una ricerca locale pura attorno al miglior punto trovato finora, riducendo progressivamente il raggio di ricerca per raffinare la soluzione.

---

## 4. Perché funziona bene?
-   **Robustezza**: La Trust Region impedisce al modello lineare di "impazzire" in spazi non lineari.
-   **Efficienza**: La stima Bayesiana del gradiente guida la ricerca molto meglio del campionamento casuale.
-   **Scalabilità**: Il partizionamento permette di gestire spazi complessi scomponendoli in problemi più semplici.
-   **Esplorazione Intelligente**: L'uso dell'incertezza ($\sigma$) evita di rimanere bloccati in minimi locali o di sovrastimare regioni poco esplorate.

## 5. Risultati dei Benchmark
L'algoritmo è stato testato su:
-   **Tabular NN**: Competitivo con Optuna, batte Random Search.
-   **XGBoost**: Risultati solidi e consistenti.
-   **ParamNet (Adult, Higgs, Letter, Mnist, Optdigits)**: Vince su 5 dataset su 6 contro Optuna e Random Search, dimostrando un'ottima capacità di generalizzazione.
