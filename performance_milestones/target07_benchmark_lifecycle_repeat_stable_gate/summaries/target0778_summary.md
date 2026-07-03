# TARGET 07.78 Summary

Decision: `keep opt-in`.

4096/1024 output tok/s:

| Variant | Mean | Median | Best | Worst | Std | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | 131.9759 | 132.0834 | 132.2002 | 131.6442 | 0.2932 | 0.002221 |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 131.9375 | 132.0993 | 132.2654 | 131.4479 | 0.4321 | 0.003275 |

4096/128 output tok/s:

| Variant | Mean | Median | Best | Worst | Std | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dsv4_sm80_a100_victory` | 62.3617 | 62.3462 | 62.4297 | 62.3093 | 0.0617 | 0.000989 |
| `dsv4_sm80_a100_victory_densefp8marlinproj` | 61.5131 | 62.4034 | 62.4479 | 59.6880 | 1.5807 | 0.025697 |

Promotion thresholds:

- `long_median_output_delta_pct`: `0.012069`
- `long_mean_output_delta_pct`: `-0.029078`
- `short_median_output_delta_pct`: `0.091723`
- `baseline_long_output_cv`: `0.002221`
- `candidate_long_output_cv`: `0.003275`
- `candidate_cv_threshold`: `0.020000`
- `candidate_cv_ok`: `True`
- `catastrophic_repeat_regressions`: `[]`
- `smoke_ok`: `True`
- `long_graph_ok`: `True`
- `short_graph_ok`: `True`
