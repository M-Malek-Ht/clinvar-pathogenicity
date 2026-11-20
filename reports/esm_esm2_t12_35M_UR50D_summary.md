# Level 2: ESM-2 (esm2_t12_35M_UR50D), mapped validation variants, 5 seeds

| Model | Split | AUROC | AUPRC | Brier | ECE | Per-gene AUROC |
|---|---|---|---|---|---|---|
| Level 1 trees (swap + position) | random | 0.834 ± 0.010 | 0.718 ± 0.018 | | | |
| Level 1 trees (swap + position) | gene | 0.781 ± 0.013 | 0.628 ± 0.019 | | | |
| ESM-2 zero-shot (no training) | random | 0.764 ± 0.005 | 0.692 ± 0.008 |  |  | 0.736 ± 0.012 |
| ESM-2 zero-shot (no training) | gene | 0.768 ± 0.011 | 0.696 ± 0.013 |  |  | 0.738 ± 0.021 |
| ESM-2 embedding + LR | random | 0.923 ± 0.003 | 0.863 ± 0.006 | 0.097 ± 0.002 | 0.005 ± 0.001 | 0.864 ± 0.011 |
| ESM-2 embedding + LR | gene | 0.918 ± 0.006 | 0.854 ± 0.009 | 0.102 ± 0.004 | 0.009 ± 0.002 | 0.851 ± 0.009 |
| Trees + ESM-2 score | random | 0.888 ± 0.008 | 0.822 ± 0.012 | 0.114 ± 0.004 | 0.017 ± 0.002 | 0.824 ± 0.013 |
| Trees + ESM-2 score | gene | 0.848 ± 0.016 | 0.775 ± 0.018 | 0.131 ± 0.006 | 0.017 ± 0.007 | 0.821 ± 0.013 |

Cost (mean over runs; ESM-2 inference itself is in reports/esm_esm2_t12_35M_UR50D_run.json):

- ESM-2 embedding + LR: training incl. tuning 7.3 s, inference 45.56 ms per 1,000, 14 KB
- Trees + ESM-2 score: training incl. tuning 13.2 s, inference 49.17 ms per 1,000, 444 KB
