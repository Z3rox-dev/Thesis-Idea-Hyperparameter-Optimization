# Coherence + Potential Field — Value-Added Ablation

Lower is better for `final_best` and `auc_best`.

## Sphere (20D, budget=500)

Baseline for win-rate: `coherence`.

| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |

|---|---:|---:|---:|---:|

| both | 11.21±3.05 | 10.73 | 27.07±6.5 | 2.44±0.46 |
| coherence | 11.21±3.05 | 10.73 | 27.07±6.5 | 2.5±0.45 |
| none | 11.96±3.46 | 12.02 | 27.05±6.42 | 2.48±0.48 |
| potential | 11.96±3.46 | 12.02 | 27.05±6.42 | 2.49±0.38 |

| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |

|---|---:|---:|---:|---:|

| both | 6 | 0.475 | 0.319 | 0.000 |
| coherence | 6 | 0.475 | 0.319 | 0.000 |
| none | 6 | nan | nan | nan |
| potential | 6 | 0.481 | 0.306 | 0.000 |

| vs baseline | win-rate | median(Δ final_best) |

|---|---:|---:|

| both | 0% | +0 |
| none | 12% | +0.3598 |
| potential | 12% | +0.3598 |

## Rastrigin (20D, budget=500)

Baseline for win-rate: `coherence`.

| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |

|---|---:|---:|---:|---:|

| both | 157.8±17.6 | 163.8 | 180.7±7.21 | 1.74±0.5 |
| coherence | 157.8±17.6 | 163.8 | 180.7±7.21 | 1.88±0.54 |
| none | 156.8±17.4 | 160.5 | 180.7±7.22 | 1.98±0.78 |
| potential | 156.1±16.6 | 160.5 | 180.7±7.22 | 1.83±0.56 |

| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |

|---|---:|---:|---:|---:|

| both | 6 | 0.460 | 0.332 | 0.000 |
| coherence | 6 | 0.460 | 0.332 | 0.000 |
| none | 6 | nan | nan | nan |
| potential | 6 | 0.454 | 0.322 | 0.000 |

| vs baseline | win-rate | median(Δ final_best) |

|---|---:|---:|

| both | 0% | +0 |
| none | 25% | +0 |
| potential | 38% | +0 |

## Rosenbrock (20D, budget=500)

Baseline for win-rate: `coherence`.

| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |

|---|---:|---:|---:|---:|

| both | 1.686e+04±1.5e+04 | 1.13e+04 | 1.203e+05±4.52e+04 | 1.39±0.081 |
| coherence | 1.686e+04±1.5e+04 | 1.13e+04 | 1.203e+05±4.52e+04 | 1.41±0.066 |
| none | 1.729e+04±1.45e+04 | 1.024e+04 | 1.204e+05±4.51e+04 | 1.43±0.073 |
| potential | 1.66e+04±1.51e+04 | 1.024e+04 | 1.203e+05±4.52e+04 | 1.59±0.43 |

| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |

|---|---:|---:|---:|---:|

| both | 6 | 0.529 | 0.359 | 0.096 |
| coherence | 6 | 0.529 | 0.359 | 0.096 |
| none | 6 | nan | nan | nan |
| potential | 6 | 0.526 | 0.359 | 0.085 |

| vs baseline | win-rate | median(Δ final_best) |

|---|---:|---:|

| both | 0% | +0 |
| none | 12% | +0 |
| potential | 12% | +0 |

## Notes / Caveats

- This is a synthetic suite: conclusions should be confirmed on your target benchmarks.
- `auc_best` captures sample-efficiency (how quickly good solutions appear), not just the final result.
- If you see mixed results (wins on some functions, losses on others), that is expected: these signals
  are most useful when local gradients are informative but not globally consistent.
