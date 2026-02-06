# Teoria dell'Approssimazione Locale Adattiva Bayesiana (ALBA)

## 1. Introduzione: Il Problema dell'Ottimizzazione Black-Box
L'algoritmo `HPOptimizerV5s` affronta il problema della minimizzazione globale di una funzione obiettivo $f: \mathcal{X} \to \mathbb{R}$, dove $\mathcal{X} \subset \mathbb{R}^d$ è un dominio limitato (iper-rettangolo).
Le caratteristiche del problema sono:
1.  **Black-Box**: Non abbiamo accesso alla forma analitica di $f$, né ai suoi gradienti $\nabla f$.
2.  **Costosa**: Ogni valutazione $y_i = f(x_i)$ richiede risorse significative (tempo, calcolo).
3.  **Non-Convessa e Non-Stazionaria**: $f$ può avere molti minimi locali e il suo comportamento (scala, variabilità) può cambiare drasticamente in diverse regioni dello spazio.

L'approccio proposto, che denominiamo **ALBA (Adaptive Local Bayesian Approximation)**, si fonda sulla decomposizione del problema globale complesso in una serie di sottoproblemi locali più semplici, modellati probabilisticamente.

---

## 2. Il Principio di Località e l'Espansione di Taylor
Il fondamento teorico dell'algoritmo è il **Teorema di Taylor**. Per una funzione $f$ sufficientemente liscia ($C^2$), il valore in un punto $x$ vicino a un punto di riferimento $c$ (centro di una regione locale) può essere approssimato come:

$$ f(x) = f(c) + \nabla f(c)^T (x - c) + \frac{1}{2}(x - c)^T H(c) (x - c) + O(||x - c||^3) $$

Dove $H(c)$ è la matrice Hessiana.
I metodi di ottimizzazione basati su gradienti (come SGD) utilizzano solo il termine lineare. Tuttavia, in un contesto Black-Box, non conosciamo $\nabla f(c)$. Dobbiamo stimarlo dai dati.

### Il Limite dei Modelli Globali
Un modello surrogato globale (es. un Processo Gaussiano standard con kernel stazionario) assume che le proprietà di $f$ (come la lunghezza di correlazione) siano costanti ovunque. Se $f$ è molto frastagliata in una zona e piatta in un'altra, il modello globale fallisce.
ALBA risolve questo problema assumendo che **localmente** (in un intorno $U_c$ di $c$), il termine quadratico e quelli superiori siano trascurabili o trattabili come "rumore", permettendo l'uso di un modello lineare:

$$ f(x) \approx \beta_0 + \beta^T (x - c) \quad \forall x \in U_c $$

---

## 3. Trust Region e Weighted Ridge Regression
L'approssimazione lineare è valida solo se il termine di errore (dominato dalla curvatura $\frac{1}{2}(x-c)^T H (x-c)$) è piccolo.
L'errore di approssimazione $\epsilon(x)$ cresce quadraticamente con la distanza dal centro:
$$ |\epsilon(x)| \propto ||x - c||^2 $$

Per mitigare questo effetto senza dover stimare esplicitamente l'Hessiana (che richiederebbe $O(d^2)$ valutazioni), l'algoritmo introduce una **Trust Region Soft** tramite pesi esponenziali.

### Formulazione Matematica
Invece di una regressione standard, risolviamo un problema di **Weighted Ridge Regression**. Dato un insieme di osservazioni locali $\mathcal{D}_L = \{(x_i, y_i)\}$, cerchiamo i coefficienti $\beta$ che minimizzano:

$$ J(\beta) = \sum_{(x_i, y_i) \in \mathcal{D}_L} w_i (y_i - \beta^T \tilde{x}_i)^2 + \lambda ||\beta||^2 $$

Dove $\tilde{x}_i = x_i - c$ e i pesi sono definiti da un kernel Gaussiano:
$$ w_i = \exp\left(-\frac{||x_i - c||^2}{2\sigma_{dist}^2}\right) $$

**Interpretazione**:
Il peso $w_i$ agisce come un filtro passa-basso spaziale.
-   Punti vicini a $c$ ($||x_i - c|| \to 0$) hanno $w_i \approx 1$: Il modello deve fittarli bene.
-   Punti lontani ($||x_i - c|| \gg \sigma_{dist}$) hanno $w_i \to 0$: Il loro contributo all'errore è soppresso.
Questo permette al modello lineare di ignorare la curvatura globale e catturare correttamente il gradiente locale $\nabla f(c) \approx \beta$.

---

## 4. Inferenza Bayesiana Locale
Poiché le valutazioni sono poche e rumorose, una stima puntuale di $\beta$ è rischiosa. ALBA adotta un approccio **Bayesiano** per quantificare l'incertezza epistemica.

Assumiamo un modello generativo lineare con rumore Gaussiano eteroschedastico (indotto dai pesi):
$$ y = \phi(x)^T \mathbf{w} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2 W^{-1}) $$

Con un prior Gaussiano sui pesi $\mathbf{w} \sim \mathcal{N}(0, \Sigma_p)$.
La distribuzione a posteriori dei pesi $p(\mathbf{w} | \mathcal{D})$ è anch'essa Gaussiana:
$$ \mathbf{w} | \mathcal{D} \sim \mathcal{N}(\bar{\mathbf{w}}, A^{-1}) $$

Dove:
-   $A = \sigma_n^{-2} X^T W X + \Sigma_p^{-1}$ (Matrice di Precisione)
-   $\bar{\mathbf{w}} = \sigma_n^{-2} A^{-1} X^T W y$ (Media a posteriori)

### Predizione e Incertezza
Per un nuovo punto candidato $x_*$, la distribuzione predittiva è:
$$ p(y_* | x_*, \mathcal{D}) = \mathcal{N}(\mu(x_*), \sigma^2(x_*)) $$

1.  **Media (Direzione di Discesa)**:
    $$ \mu(x_*) = \bar{\mathbf{w}}^T \phi(x_*) $$
    Rappresenta la miglior stima del valore della funzione.

2.  **Varianza (Incertezza)**:
    $$ \sigma^2(x_*) = \phi(x_*)^T A^{-1} \phi(x_*) + \sigma_n^2 $$
    Questo termine è cruciale. Esso cresce man mano che ci si allontana dai dati osservati (lungo direzioni non esplorate dello spazio delle feature).

---

## 5. Partizionamento Gerarchico dello Spazio
L'algoritmo non si affida a un'unica Trust Region. Utilizza una struttura dati ad albero (simile a un *k-d tree*) per gestire la **non-stazionarietà**.

-   **Adattamento alla Scala**: Regioni diverse dello spazio possono avere "densità di informazione" diverse. L'albero divide ricorsivamente i cubi che contengono molte valutazioni promettenti.
-   **Isolamento dei Minimi**: Se la funzione ha più bacini di attrazione, il partizionamento tende a isolarli in cubi separati. Ogni cubo sviluppa il proprio modello locale indipendente, evitando che i gradienti di un bacino interferiscano con quelli di un altro.

La decisione di "splittare" un cubo è basata sulla densità di campionamento, garantendo che il modello locale lineare non venga forzato a coprire un'area troppo vasta dove l'approssimazione fallirebbe.

---

## 6. Strategia di Acquisizione (UCB)
All'interno di ogni cubo, la scelta del prossimo punto $x_{next}$ è guidata dalla funzione di acquisizione **Upper Confidence Bound (UCB)**:

$$ \alpha(x) = \mu(x) + \kappa \cdot \sigma(x) $$

-   **Exploitation ($\mu(x)$)**: Segue il gradiente stimato verso il minimo locale previsto.
-   **Exploration ($\sigma(x)$)**: Spinge la ricerca verso zone del cubo dove il modello è incerto (dove la matrice $A^{-1}$ è "grande").

Il parametro $\kappa$ (legato a `novelty_weight` nel codice) controlla il trade-off.
In ALBA, questo meccanismo è doppiamente adattivo:
1.  **Adattivo nello Spazio**: Grazie al partizionamento, $\sigma(x)$ è calcolato rispetto ai dati locali.
2.  **Adattivo nella Curvatura**: Grazie alla Trust Region, $\mu(x)$ è affidabile solo dove la linearità regge.

---

## 7. Conclusione
`HPOptimizerV5s` non è una semplice euristica, ma un'implementazione approssimata di un **Processo Gaussiano Non-Stazionario con Kernel a Risoluzione Variabile**.

Invece di definire un kernel globale complesso (difficile da addestrare), l'algoritmo costruisce implicitamente la superficie di risposta incollando insieme "pezze" lineari probabilistiche (i modelli nei cubi). La Trust Region assicura che queste pezze siano localmente coerenti, mentre l'inferenza Bayesiana guida l'esplorazione per raffinare la mappa dove serve di più.
