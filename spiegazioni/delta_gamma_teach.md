# 🔥 Delta e Gamma — Spiegazione Semplice e Tecnica

## ⭐ Versione Semplice — Metafora del Cuoco

Immagina che ogni **cubo** del tuo algoritmo sia una *zona della ricetta* da esplorare.

Dentro ogni cubo costruisci una **mini-collina** (il surrogato quadratico) che rappresenta come pensi varia il punteggio in quell’area.

Poi ti chiedi:

> *“Mi conviene dividere questo cubo in cubetti più piccoli?”*

Per decidere:

1. **Simuli lo split** (ma senza farlo davvero).
2. Crei la collinetta locale per ogni figlio.
3. Valuti *quanto migliorano* (o peggiorano) le collinette dei figli rispetto al padre.

### 👉 Che cos’è **delta**?

Delta è:

delta = varianza_padre − varianza_figli

Interpretazione:

- **delta > 0** → Splittare migliora la precisione (meno incertezza).
- **delta = 0** → Splittare non cambia nulla.
- **delta < 0** → Splittare peggiora (più rumore).

### 👉 Che cos’è **gamma**?

Gamma è la **soglia minima** di miglioramento necessaria per decidere di splittare.

- Se l’aumento di qualità è **più piccolo** di gamma → *non splittare*.
- Se è **maggiore o uguale** → *splitta*.

In altre parole:

> Gamma = “quanto devono migliorare i figli prima che valga la pena dividere”.

---

## ⭐ Versione Tecnica

Il surrogato del cubo padre ha varianza residua:

σ²_parent


Dopo la simulazione dello split ottieni k figli con varianze:

σ²_child1, σ²_child2, ..., σ²_childk


### Varianza media post-split:

σ²_post = Σ (n_child / n_total) * σ²_child


### Definizione formale di **delta**:

delta = σ²_parent − σ²_post

Interpretazione tecnica:

- **delta > 0** → lo split *riduce l’errore* → buono.
- **delta = 0** → nessun cambiamento.
- **delta < 0** → surrogati peggiori → split da evitare.

### Criterio di split:

Nel tuo codice:

if delta < gamma:
blocca lo split

Quindi:

- con `gamma = 0.02` richiedi un miglioramento assoluto del 2% → spesso troppo alto.
- con `gamma = 0.0` accetti lo split se non peggiora (`delta ≥ 0`).

Gamma controlla la **sensibilità allo split**:

- **gamma basso** → splitti spesso.
- **gamma alto** → non splitti quasi mai.
- **gamma costante assoluto** → pericoloso perché dipende dalla scala della varianza.

---

## ⭐ Riassunto Finale

- **delta** = quanto lo split riduce la varianza del surrogato.  
- **gamma** = miglioramento minimo richiesto per splittare.

Regole:

- `delta < 0` → NON splittare  
- `0 ≤ delta < gamma` → miglioramento troppo debole, NON splittare  
- `delta ≥ gamma` → SPLIT  
