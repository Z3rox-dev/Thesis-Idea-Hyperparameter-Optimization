# Coherence + Potential Field — Value-Added Ablation

Lower is better for `final_best` and `auc_best`.

## Sphere (2D, budget=200)

Baseline for win-rate: `coherence`.

| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |

|---|---:|---:|---:|---:|

| both | 0.01061±0.0116 | 0.006022 | 0.5693±0.344 | 0.142±0.008 |
| coherence | 0.01447±0.0149 | 0.01228 | 0.5701±0.34 | 0.14±0.009 |
| none | 0.01148±0.0144 | 0.005163 | 0.5784±0.405 | 0.0969±0.0056 |
| potential | 0.007595±0.0105 | 0.004125 | 0.5954±0.409 | 0.146±0.0096 |

| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |

|---|---:|---:|---:|---:|

| both | 22 | 0.749 | 0.231 | 0.828 |
| coherence | 22 | 0.723 | 0.228 | 0.742 |
| none | 23 | nan | nan | nan |
| potential | 23 | 0.733 | 0.232 | 0.776 |

| vs baseline | win-rate | median(Δ final_best) |

|---|---:|---:|

| both | 55% | -0.0005779 |
| none | 65% | -0.002333 |
| potential | 70% | -0.005539 |

## Rastrigin (2D, budget=200)

Baseline for win-rate: `coherence`.

| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |

|---|---:|---:|---:|---:|

| both | 2.173±1.44 | 2.09 | 4.779±1.77 | 0.143±0.016 |
| coherence | 2.148±1.48 | 2.091 | 4.799±1.78 | 0.138±0.0074 |
| none | 1.81±1.1 | 1.974 | 4.628±1.67 | 0.0958±0.0032 |
| potential | 2.265±1.59 | 2.141 | 4.815±1.84 | 0.15±0.015 |

| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |

|---|---:|---:|---:|---:|

| both | 22 | 0.601 | 0.226 | 0.335 |
| coherence | 22 | 0.600 | 0.229 | 0.332 |
| none | 22 | nan | nan | nan |
| potential | 23 | 0.569 | 0.231 | 0.228 |

| vs baseline | win-rate | median(Δ final_best) |

|---|---:|---:|

| both | 40% | +0 |
| none | 55% | -0.02335 |
| potential | 35% | +0.05975 |

## Rosenbrock (2D, budget=200)

Baseline for win-rate: `coherence`.

| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |

|---|---:|---:|---:|---:|

| both | 1.568±3.53 | 0.2652 | 1356±1.83e+03 | 0.138±0.011 |
| coherence | 1.053±2.43 | 0.218 | 1356±1.83e+03 | 0.138±0.011 |
| none | 1.483±3.27 | 0.5073 | 1357±1.83e+03 | 0.0976±0.005 |
| potential | 1.258±3.62 | 0.3142 | 1357±1.83e+03 | 0.145±0.013 |

| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |

|---|---:|---:|---:|---:|

| both | 21 | 0.621 | 0.229 | 0.401 |
| coherence | 21 | 0.617 | 0.235 | 0.388 |
| none | 22 | nan | nan | nan |
| potential | 23 | 0.601 | 0.224 | 0.335 |

| vs baseline | win-rate | median(Δ final_best) |

|---|---:|---:|

| both | 30% | +0 |
| none | 25% | +0.1818 |
| potential | 45% | +0.0006281 |

## Notes / Caveats

- This is a synthetic suite: conclusions should be confirmed on your target benchmarks.
- `auc_best` captures sample-efficiency (how quickly good solutions appear), not just the final result.
- If you see mixed results (wins on some functions, losses on others), that is expected: these signals
  are most useful when local gradients are informative but not globally consistent.
