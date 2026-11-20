# Gradient-boosted trees on the swap features (validation part, 5 seeds)

| Model | Split | AUROC | AUPRC | BRIER | ECE | PER GENE AUROC |
|---|---|---|---|---|---|---|
| logreg | random | 0.731 ± 0.005 | 0.529 ± 0.005 | 0.178 ± 0.001 | - | 0.714 ± 0.005 |
| logreg | gene | 0.732 ± 0.005 | 0.530 ± 0.008 | 0.178 ± 0.001 | - | 0.724 ± 0.004 |
| gbm | random | 0.764 ± 0.005 | 0.583 ± 0.004 | 0.168 ± 0.001 | 0.008 ± 0.002 | 0.736 ± 0.008 |
| gbm | gene | 0.764 ± 0.003 | 0.574 ± 0.007 | 0.169 ± 0.001 | 0.011 ± 0.003 | 0.742 ± 0.009 |
| swap_lookup | random | 0.764 ± 0.005 | 0.583 ± 0.004 | 0.168 ± 0.001 | 0.008 ± 0.002 | 0.736 ± 0.008 |
| swap_lookup | gene | 0.764 ± 0.003 | 0.574 ± 0.007 | 0.169 ± 0.001 | 0.011 ± 0.003 | 0.740 ± 0.010 |

ECE for logreg was computed only for seed 0 in the notebook (0.018).

## Gradient-boosted trees: settings and cost

- Grid: {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [7, 31], 'min_samples_leaf': [20, 200]}, max_iter=200, tuned by 5-fold GroupKFold on train, scored by AUPRC
- Chosen (learning_rate, max_leaf_nodes, min_samples_leaf) across the 10 runs: 0.1, 31.0, 200.0 x3; 0.1, 7.0, 200.0 x2; 0.1, 7.0, 20.0 x2; 0.05, 31.0, 20.0 x1; 0.1, 31.0, 20.0 x1; 0.05, 31.0, 200.0 x1
- Training time incl. tuning: 18.9 s on average (wall-clock, parallel; depends on the machine)
- Inference: 3.764 ms per 1,000 variants
- Model size: 490 KB
- Predicted probabilities range 0.02 to 0.88

swap_lookup is a diagnostic: the best any swap-only model can do on this data.
