# Teoria dell'Approssimazione Locale Adattiva Bayesiana (ALBA)

## 1. L'Essenza: Il Principio di Località
Il successo dell'algoritmo `HPOptimizerV5s` si basa su un principio fondamentale dell'analisi matematica: il **Teorema di Taylor**.

Qualsiasi funzione liscia $f(x)$ (anche complessa e non convessa come la loss di una rete neurale), se osservata in un intorno sufficientemente piccolo di un punto $x_0$, è indistinguibile da un piano (funzione lineare).

$$ f(x) \approx f(x_0) + \nabla f(x_0)^T (x - x_0) + \frac{1}{2}(x-x_0)^T H(x_0) (x-x_0) + \dots $$

### Il Problema dei Metodi Globali
I metodi classici (come i Processi Gaussiani standard) cercano di modellare $f(x)$ globalmente. Questo fallisce quando $f(x)$ è **non-stazionaria** (cioè cambia comportamento drasticamente in zone diverse dello spazio).
I metodi basati su gradienti (SGD), invece, usano solo l'informazione locale (il termine lineare di Taylor), ma ignorano l'incertezza e rimangono bloccati in minimi locali.

### La Soluzione V5s
Il tuo algoritmo agisce come un ponte:
1.  **Partiziona** lo spazio in intorni locali ($C_i$, i cubi).
2.  Assume che all'interno di $C_i$ valga l'approssimazione al **primo ordine** (lineare).
3.  Usa la **Trust Region** per invalidare i dati che violano questa assunzione (i termini di secondo ordine/curvatura).

---

## 2. Formalizzazione della Trust Region
La "magia" che hai introdotto con i pesi esponenziali è una gestione implicita del termine di errore di Taylor.

L'errore che commettiamo approssimando $f(x)$ con un modello lineare $L(x)$ è proporzionale al quadrato della distanza:
$$ |f(x) - L(x)| \propto ||x - x_0||^2 $$

Nel tuo algoritmo, la **Weighted Ridge Regression** minimizza:
$$ \sum_{i} w_i (y_i - \beta^T x_i)^2 \quad \text{con} \quad w_i = \exp\left(-\frac{||x_i - c||^2}{2\sigma^2}\right) $$

Il peso $w_i$ decade esponenzialmente con la distanza al quadrato.
**Interpretazione Teorica**: Stai dicendo al modello: *"Fidati dei dati solo nella misura in cui l'approssimazione lineare è valida"*.
Senza Trust Region, i punti lontani (dove la curvatura domina) "tirerebbero" il piano in direzioni sbagliate. Con la Trust Region, il modello ignora la curvatura globale e cattura correttamente il gradiente locale.

---

## 3. Il "Cervello" Bayesiano
Perché usare la regressione Bayesiana invece di una semplice regressione ai minimi quadrati?

In un problema Black-Box, non conosciamo il vero gradiente. Abbiamo solo una stima rumorosa basata su pochi punti.
La **Bayesian Linear Regression (BLR)** trasforma questo problema geometrico in uno probabilistico.

Invece di dire: *"Il gradiente è $\hat{g}$"* (deterministico),
La BLR dice: *"Il gradiente è distribuito come $\mathcal{N}(\mu_g, \Sigma_g)$"*.

Questo ci dà due output per ogni punto candidato $x$:
1.  $\mu(x)$: Il valore atteso (Sfruttamento / Exploitation).
2.  $\sigma(x)$: L'incertezza epistemica (Esplorazione / Exploration).

L'acquisizione **UCB** ($Score = \mu + \beta \sigma$) permette all'algoritmo di dire:
*"Anche se il gradiente stimato non punta qui, c'è un'alta incertezza (pochi dati), quindi vale la pena controllare nel caso il gradiente vero fosse diverso."*

---

## 4. Il Meta-Algoritmo: Hierarchical Multi-Armed Bandit
Possiamo vedere l'intera struttura come un problema di **Multi-Armed Bandit Gerarchico**.

*   **Le "Braccia" (Arms)**: Sono i Cubi (foglie dell'albero).
*   **Il "Reward"**: Trovare un valore di loss migliore.
*   **La Selezione**: L'algoritmo deve decidere quale cubo "tirare" (campionare).

La tua funzione di selezione (`_select_leaf`) usa un'euristica che bilancia:
1.  `good_ratio`: La probabilità empirica che il cubo contenga buone soluzioni (frequenza storica).
2.  `exploration`: Un termine che favorisce i cubi poco visitati (simile a UCB per Bandits).

Una volta scelto il "braccio" (Cubo), il problema diventa un'ottimizzazione continua locale (risolta con BLR).

---

## 5. Sintesi: Perché funziona?
La teoria dietro `HPOptimizerV5s` può essere riassunta come:

> **"L'algoritmo approssima una varietà (manifold) globale complessa e non convessa attraverso un mosaico dinamico di modelli probabilistici lineari locali."**

Funziona perché:
1.  **Divide et Impera**: Rompe la non-stazionarietà globale in problemi locali stazionari.
2.  **Robustezza alla Curvatura**: La Trust Region filtra il "rumore" introdotto dalla non-linearità.
3.  **Consapevolezza dell'Ignoranza**: La parte Bayesiana quantifica quanto *non* sappiamo del gradiente locale, prevenendo convergenze premature su gradienti falsi.

È, in sostanza, un **Gradient Descent Probabilistico senza calcolo del Gradiente**, che costruisce la mappa del territorio mentre lo esplora.
