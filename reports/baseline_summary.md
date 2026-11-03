# Initial baseline results (validation part, 5 seeds)

Validation prevalence (= AUPRC of a random guess): 0.296

| Model | Split | AUROC | AUPRC | Brier (lower = better) | Per-gene AUROC |
|---|---|---|---|---|---|
| gene_prior | gene | 0.500 ± 0.000 | 0.296 ± 0.000 | 0.208 ± 0.000 | 0.500 ± 0.000 |
| gene_prior | random | 0.931 ± 0.002 | 0.862 ± 0.004 | 0.091 ± 0.001 | 0.500 ± 0.000 |
| logreg | gene | 0.732 ± 0.005 | 0.530 ± 0.008 | 0.178 ± 0.001 | 0.724 ± 0.004 |
| logreg | random | 0.731 ± 0.005 | 0.529 ± 0.005 | 0.178 ± 0.001 | 0.714 ± 0.005 |

## Logistic regression: settings and cost

- Chosen C per run: [0.001, 0.01, 0.1, 10.0, 100.0]
- Training time incl. tuning (CPU): 2.3 s on average
- Inference: 0.288 ms per 1,000 variants
- Model size: 1.6 KB
- Per-gene AUROC uses val genes that contain both classes (on average 616 genes).
