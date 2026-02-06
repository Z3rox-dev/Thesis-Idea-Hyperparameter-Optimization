# Piano: griglia + heatmap per-leaf con scoring LGS a batch

## Obiettivo
Sostituire la generazione candidati “a miscuglio” nella folder `alba_framework` con una strategia **geometrica** basata su:

- **Griglia per-leaf (cubo)**: candidati ordinati dentro i bounds del leaf.
- **Scoring economico**: LGS calcola `mu/sigma` su tanti candidati (anche milioni) **senza consumare budget objective**.
- **Valutazione costosa**: a ogni iterazione si valuta **un solo candidato** (quello scelto) con l’objective.
- **Heatmap per-leaf**: si aggiorna solo su `tell()` con l’esito reale (buono/cattivo o statistiche continue).
- **Ereditarietà allo split**: quando un leaf viene splittato, la griglia/heatmap viene “tramandata” ai figli, con **zoom** (griglia più fitta) e **riscalamento/redistribuzione** dei segnali esistenti.

Questo documento descrive il piano tecnico e le decisioni di design per implementarlo.

---

## Glossario
- **Leaf / Cubo**: regione iper-rettangolare dell’albero (bounds locali).
- **Grid**: discretizzazione regolare (o quasi) dello spazio del leaf.
- **Cella**: un “bin” multidimensionale della grid.
- **Heatmap**: mapping cella → statistiche (visite, best, mean, good/bad, ecc.).
- **LGS**: Local Gradient Surrogate, usato per predire `mu(x)` e `sigma(x)` su candidati.

---

## Vincoli chiave (da rispettare)
1. **1 eval objective per iterazione**: non si valuta tutta la griglia.
2. **Molti candidati scorati da LGS**: OK, ma a batch per non saturare RAM.
3. **La griglia ha senso solo nei bounds del leaf**: ogni leaf mantiene la propria.
4. **Split = zoom**: nei figli la discretizzazione può diventare più fitta.
5. **Heatmap deve essere trasferibile**: i segnali del padre devono finire nel figlio giusto e (se aumenta risoluzione) devono essere “riscalati”/ridistribuiti.

---

## Architettura proposta

### 1) Struttura dati per leaf: `LeafGridState`
Per ogni leaf (cubo) introduci uno stato persistente:

- `bounds`: bounds correnti del leaf.
- `B`: numero di bin per dimensione (o una strategia per determinarlo, es. fisso o adattivo).
- `stats`: dizionario sparse `cell_id -> CellStats`.
- `total_visits`: contatore globale per leaf.
- (opzionale) `last_refresh_iter`: per logiche di “fading/schiarimento”.

Dove:

- `cell_id` può essere:
  - una tupla di indici `(i0, i1, ..., id-1)` (comoda ma più pesante), oppure
  - un intero ottenuto con encoding base-`B` (più compatto).

E `CellStats` contiene solo statistiche **additive** (facili da trasferire/ricostruire):
- `n`: quante visite
- `sum_y`: somma degli score osservati nella cella
- `sum_y2`: somma dei quadrati degli score osservati nella cella
- `n_good`: conteggio “buoni” rispetto alla `gamma` corrente (vedi nota sotto)

Nota: è **sparse**: esistono solo le celle visitate.

### 2) Mappatura punto → cella (binning)
Dato `x` (in coordinate normalizzate o nei bounds del leaf), la cella è:

- per dimensione `j`:
  - `t = (x[j] - lo[j]) / (hi[j] - lo[j])` clippato in `[0,1]`
  - `idx_j = floor(t * B)` clippato in `[0, B-1]`

Questo binning definisce la heatmap.

### 3) Generazione candidati “geometrica”
Due modalità, entrambe compatibili con batch + no-RAM-explosion:

- **(A) Griglia implicita + streaming**
  - Non materializzi `B^d` punti.
  - Produci le coordinate di cella in ordine (o quasi) e trasformi ogni cella in un punto rappresentativo (es. centro cella).
  - Per aumentare densità, campioni più punti per cella (es. jitter locale).

- **(B) Griglia esplicita su un sottoinsieme di dimensioni**
  - Se `d` è alto, scegli `k << d` dimensioni “attive” e fai una griglia su quelle.
  - Le altre dimensioni le fissi al centro del leaf o al best_x locale.

- **(C) Alta dimensionalità: densità ridotta ma su TUTTE le dimensioni (niente fissaggio)**
  - Se vuoi considerare tutte le dimensioni anche quando `d` è alto, evita la griglia cartesiana `B^d`.
  - Usa invece una sequenza “ordinata” e space-filling (es. low-discrepancy tipo Sobol/Halton) per generare punti in `[0,1]^d` e mapparli nei bounds del leaf.
  - La “densità” la controlli con `N_batch` (quanti punti scorare per iterazione), non con il prodotto cartesiano.
  - La heatmap resta definita dal binning su tutte le dimensioni (B può anche diminuire all’aumentare di `d`).

Nel piano base si parte da (A) perché è più generale; se l’obiettivo è **non fissare** dimensioni anche in alta dimensionalità, (C) è la variante consigliata.

### 4) Scoring LGS su candidati (a batch)
Per scegliere il candidato da valutare:

1. Generi un batch di `N_batch` candidati dalla griglia del leaf.
2. Calcoli `mu, sigma = cube.predict_bayesian(candidates)`.
3. Applichi una regola di selezione:
   - `argmax(mu)` (semplice)
   - oppure `argmax(mu + beta*sigma)` se vuoi exploration guidata
   - oppure softmax (stocastico)
4. Mantieni il best candidato visto finora.
5. Ripeti per i batch necessari.
6. Alla fine scegli **1** candidato e lo mandi all’objective.

Importante: questo non consuma budget objective.

### 5) Update heatmap su `tell()`
Quando arriva il vero `y`:

- Mappi il punto `x` alla cella del leaf.
- Aggiorni `CellStats` (additivo):
  - `n += 1`
  - `sum_y += y`
  - `sum_y2 += y*y`
  - `n_good` **non va considerato permanente** se `gamma` cambia nel tempo (vedi nota sotto)

Questo crea la “mappa” nel tempo.

Nota su `n_good` e aggiornamento di `gamma`
- In ALBA `gamma` cambia durante l’ottimizzazione.
- Se `gamma` cambia, un punto che ieri era “good” può diventare “bad” (o viceversa).
- Quindi: se vuoi usare `n_good` in modo corretto, deve essere **ricalcolato** quando `gamma` viene aggiornato.
- Scelta proposta (robusta e semplice): ricalcolare `n_good` per cella dai `tested_pairs` del leaf ogni volta che ALBA fa il “recount” dei good (stesso momento in cui già ricalcola `leaf.n_good`).

---

## Split: come tramandare griglia + heatmap (parte cruciale)

### Problema
Quando un leaf viene splittato in due figli:

- i punti/celle del padre devono “cadere” nel figlio corretto (in base ai bounds).
- nei figli la griglia può diventare **più fitta** (zoom), quindi:
  - una cella del padre può corrispondere a più celle del figlio.
  - la heatmap deve essere **riscalata/redistribuita**.
- vuoi anche che la heatmap si “schiarisca” ai bordi / nelle zone limitrofe (effetto smoothing/fading).

### Scelta di design: trasferire celle, non punti
Trasferiamo le **statistiche per cella** (sparse) dal padre ai figli.

Non serve conservare tutti i punti della griglia; basta conservare i `CellStats`.

### Step 1: definire la griglia del figlio
Quando nasce un figlio:

- stabilisci il suo `B_child`.
- tipicamente `B_child >= B_parent` (zoom più fitto) oppure costante.

Regola proposta (semplice):
- se vuoi densificare con la profondità: `B_child = min(B_max, B_parent + delta)`.

### Step 2: mappare una cella del padre nel figlio
Per ogni cella del padre (indicizzata dagli indici `idx_parent`):

1. Calcola il **bounding box** della cella nel continuo (nel sistema del padre).
2. Interseca quel box con i bounds del figlio.
3. Se intersezione vuota → quella cella non va nel figlio.
4. Se intersezione non vuota → devi distribuirla sulle celle del figlio che coprono l’intersezione.

### Step 3: redistribuzione (rescaling) delle statistiche
Tre strategie possibili (da scegliere):

#### Strategia S1 — “Conservativa (punto rappresentativo)”
- Rappresenti la cella del padre con un punto (es. centro cella).
- Lo assegni al figlio giusto.
- Lo binni nella griglia del figlio e copi/accumuli lì le statistiche.

Pro: implementazione semplicissima.
Contro: perdita di informazione geometrica quando `B_child` aumenta.

#### Strategia S2 — “Area/volume-weighted split” (consigliata)
- Consideri la cella del padre come un iper-rettangolo.
- Distribuisci le sue statistiche alle celle figlie proporzionalmente al volume di intersezione.

Esempio per `n` (visite):
- `n_child_cell += n_parent_cell * w` dove `w` è frazione di volume.

Nota: `n` diventa frazionario; puoi:
- tenerlo float (OK), oppure
- arrotondare in modo stocastico, oppure
- mantenere due variabili (`mass` float + `count` int).

Pro: vero rescaling con zoom.
Contro: più codice.

#### Strategia S3 — “Splatting + smoothing (schiarimento)” (per il tuo ‘schiarisce’)
- Come S2, ma dopo la redistribuzione applichi un filtro locale (tipo kernel) sulle celle del figlio:
  - parte della massa/statistiche si diffonde alle celle vicine.

Questo realizza l’effetto “schiarire”/sfumare ai bordi e nelle zone limitrofe.

Pro: implementa esattamente l’intuizione di “heatmap che schiarisce”.
Contro: introduce un iperparametro (raggio kernel, strength).

Nel piano: implementare prima S1 o S2, poi aggiungere S3 come optional.

### Step 4: coerenza con i dati reali
Nota importante:
- ALBA già tiene i `tested_pairs` per leaf.
- Dopo lo split i figli ricevono i `tested_pairs` filtrati.

Quindi la heatmap può sempre essere:
- trasferita come sopra, oppure
- ricostruita “esattamente” dai `tested_pairs`.

Decisione (MVP consigliato)
- **Su split: rebuild della heatmap dei figli dai `tested_pairs`**.
- Questo evita problemi semantici su statistiche non additive (es. `best`) e rende coerente anche `n_good` quando `gamma` cambia.
- Le strategie S1/S2/S3 restano utili eventualmente come ottimizzazioni o come smoothing, ma dopo aver stabilizzato la baseline.

---

## Scelta cella/candidato: come integrare heatmap e LGS
La tua idea prevede che la griglia sia ordinata e geometrica. LGS può:

- scorrere la griglia e trovare punti con `mu` alto (exploitation)
- o `mu + beta*sigma` alto (exploration)

La heatmap può entrare come:

- penalità per celle molto visitate
- priorità per celle mai visitate (copertura)
- check di affidabilità: se LGS propone sempre celle “storicamente cattive”, aumenti exploration o cambi scoring.

Regola base (minima):
- score_final = score_LGS - lambda * f(visits_in_cell)

Dove `f` cresce con le visite (es. `log(1+n)`).

---

## Integrazione nel codice (alto livello)

### Dove toccare
- `thesis/alba_framework/optimizer.py`
  - sostituire la parte di generazione candidati `_generate_candidates(...)` con una variante “grid streaming”.
  - mantenere `cube.predict_bayesian(...)` per scoring.

- `thesis/alba_framework/cube.py`
  - aggiungere un riferimento opzionale allo stato grid/heatmap (`grid_state`).
  - gestire copy/transfer su split.

- nuovo modulo (proposto): `thesis/alba_framework/grid.py`
  - `LeafGridState`, `CellStats`, binning, streaming di candidati, transfer allo split.

### Feature flag iniziale
Per non rompere tutto subito:
- introdurre una modalità (es. `sampling_mode="grid_lgs"`) o un parametro booleano.
- poi, quando stabilizzato, rimuovere la mixture.

---

## Parametri minimi da decidere
- `B` (bin per dimensione) per leaf.
- `N_batch` (quanti candidati per batch).
- `beta` se usi `mu+beta*sigma`.
- strategia split-transfer: S1 / S2 / S3.
- se e come applicare “schiarimento” (kernel radius, strength).

---

## Checkpoints di validazione
1. **Correctness**
   - con `B` piccolo e `N_batch` piccolo il sampler deve funzionare e non uscire dai bounds.

2. **No RAM blow-up**
   - verificare che non si materializzano arrays giganteschi.

3. **Split transfer**
   - test manuale: crea un leaf con heatmap non vuota, splitta, verifica che:
     - le statistiche finiscono nel figlio giusto
     - se `B_child > B_parent` il rescaling produce densità maggiore e i bordi risultano più “diluiti/sfumati” se S3.

4. **Confronto benchmark**
   - lancia battery ParamNet e confronta contro baseline.

---

## Note sul “perché è più pulito”
- Elimina la candidate generation a miscuglio.
- Rende la geometria esplicita (griglia per leaf).
- Heatmap è una memoria interpretabile.
- LGS resta un “motore veloce” per scegliere 1 punto senza consumare budget.

---

## Prossimo passo operativo
Decidere:
1) `B` iniziale (es. 6? 8?)
2) strategia di transfer (S1 per MVP o S2/S3 se vuoi subito zoom+schiarimento)
3) scoring: `mu` o `mu+beta*sigma`

Poi si implementa `grid.py` e si integra in `optimizer.py`.
