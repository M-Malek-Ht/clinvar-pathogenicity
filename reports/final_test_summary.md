# Final test results (ESM-2 esm2_t12_35M_UR50D)

Each model refit on train + val, scored once on the TEST part; mean ± std over 5 seeds, mapped test variants. 'AUROC, all' includes unmapped variants (Level 1 models only).

| Model | Split | AUROC | AUPRC | Brier | ECE | Per-gene AUROC | AUROC, all |
|---|---|---|---|---|---|---|---|
| Gene-only reference (diagnostic) | random | 0.932 ± 0.002 | 0.869 ± 0.005 | 0.091 ± 0.002 | 0.019 ± 0.002 | 0.500 ± 0.000 |  |
| Gene-only reference (diagnostic) | gene | 0.500 ± 0.000 | 0.301 ± 0.002 | 0.210 ± 0.001 | 0.005 ± 0.002 | 0.500 ± 0.000 |  |
| L1 logistic regression (swap) | random | 0.731 ± 0.002 | 0.541 ± 0.004 | 0.181 ± 0.001 | 0.019 ± 0.002 | 0.716 ± 0.008 | 0.731 ± 0.002 |
| L1 logistic regression (swap) | gene | 0.720 ± 0.005 | 0.516 ± 0.008 | 0.183 ± 0.002 | 0.022 ± 0.001 | 0.715 ± 0.010 | 0.720 ± 0.006 |
| L1 trees (swap) | random | 0.763 ± 0.002 | 0.592 ± 0.003 | 0.171 ± 0.001 | 0.008 ± 0.001 | 0.735 ± 0.009 | 0.762 ± 0.002 |
| L1 trees (swap) | gene | 0.754 ± 0.006 | 0.567 ± 0.006 | 0.173 ± 0.002 | 0.011 ± 0.001 | 0.734 ± 0.010 | 0.754 ± 0.005 |
| L1 trees (swap + length + position) | random | 0.830 ± 0.009 | 0.711 ± 0.018 | 0.147 ± 0.005 | 0.022 ± 0.004 | 0.755 ± 0.009 | 0.828 ± 0.009 |
| L1 trees (swap + length + position) | gene | 0.775 ± 0.012 | 0.612 ± 0.015 | 0.166 ± 0.004 | 0.016 ± 0.003 | 0.748 ± 0.009 | 0.773 ± 0.011 |
| L2 ESM-2 zero-shot | random | 0.763 ± 0.002 | 0.692 ± 0.003 |  |  | 0.736 ± 0.007 |  |
| L2 ESM-2 zero-shot | gene | 0.752 ± 0.011 | 0.673 ± 0.017 |  |  | 0.716 ± 0.008 |  |
| L2 ESM-2 embedding + LR | random | 0.923 ± 0.001 | 0.864 ± 0.002 | 0.097 ± 0.000 | 0.004 ± 0.001 | 0.856 ± 0.006 |  |
| L2 ESM-2 embedding + LR | gene | 0.924 ± 0.003 | 0.861 ± 0.006 | 0.097 ± 0.003 | 0.006 ± 0.002 | 0.848 ± 0.011 |  |
| L2 trees + ESM-2 score | random | 0.892 ± 0.001 | 0.827 ± 0.003 | 0.113 ± 0.001 | 0.017 ± 0.002 | 0.826 ± 0.004 |  |
| L2 trees + ESM-2 score | gene | 0.841 ± 0.011 | 0.758 ± 0.016 | 0.134 ± 0.005 | 0.017 ± 0.006 | 0.812 ± 0.010 |  |

## Gene-level bootstrap 95% CIs (seed 0, 1,000 resamples of test genes)

| Model | Split | AUROC [95% CI] | AUPRC [95% CI] |
|---|---|---|---|
| Gene-only reference (diagnostic) | random | 0.932 [0.925, 0.938] | 0.868 [0.848, 0.884] |
| L1 logistic regression (swap) | random | 0.729 [0.717, 0.744] | 0.541 [0.502, 0.584] |
| L1 trees (swap) | random | 0.761 [0.748, 0.775] | 0.594 [0.553, 0.637] |
| L1 trees (swap + length + position) | random | 0.829 [0.814, 0.842] | 0.711 [0.668, 0.752] |
| L2 ESM-2 zero-shot | random | 0.764 [0.747, 0.782] | 0.693 [0.655, 0.730] |
| L2 ESM-2 embedding + LR | random | 0.923 [0.916, 0.929] | 0.864 [0.844, 0.882] |
| L2 trees + ESM-2 score | random | 0.892 [0.882, 0.902] | 0.829 [0.803, 0.854] |
| AUROC difference: ESM-2 LR minus L1 trees (swap + position) | random | +0.094 [+0.083, +0.106] |  |
| Gene-only reference (diagnostic) | gene | 0.500 [0.500, 0.500] | 0.302 [0.249, 0.352] |
| L1 logistic regression (swap) | gene | 0.719 [0.703, 0.736] | 0.515 [0.445, 0.576] |
| L1 trees (swap) | gene | 0.754 [0.738, 0.770] | 0.568 [0.494, 0.628] |
| L1 trees (swap + length + position) | gene | 0.790 [0.758, 0.821] | 0.634 [0.524, 0.720] |
| L2 ESM-2 zero-shot | gene | 0.744 [0.716, 0.774] | 0.657 [0.585, 0.721] |
| L2 ESM-2 embedding + LR | gene | 0.923 [0.909, 0.935] | 0.858 [0.811, 0.895] |
| L2 trees + ESM-2 score | gene | 0.853 [0.827, 0.875] | 0.771 [0.702, 0.823] |
| AUROC difference: ESM-2 LR minus L1 trees (swap + position) | gene | +0.132 [+0.106, +0.160] |  |

## Drop from random to gene split (sub-question 1)

| Model | AUROC random - gene |
|---|---:|
| Gene-only reference (diagnostic) | +0.432 |
| L1 logistic regression (swap) | +0.012 |
| L1 trees (swap) | +0.009 |
| L1 trees (swap + length + position) | +0.055 |
| L2 ESM-2 zero-shot | +0.011 |
| L2 ESM-2 embedding + LR | -0.000 |
| L2 trees + ESM-2 score | +0.050 |

## Complexity (sub-question 2: what the improvement costs)

Mean over the 10 refits on train + val. Training includes the hyperparameter search. Inference is the classifier only; the ESM-2 feature cost is listed separately (one GPU pass over every protein).

| Model | Features | Learned numbers | Size (KB) | Training incl. tuning (s, CPU) | Inference (ms / 1,000, CPU) | Feature cost (ms / 1,000) | Test AUROC, gene split |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 logistic regression (swap) | 16 swap | 17 | 2 | 1.8 | 0.15 | ≈ 0 | 0.720 |
| L1 trees (swap) | 16 swap | 10,280 | 595 | 17.1 | 4.92 | ≈ 0 | 0.754 |
| L1 trees (swap + length + position) | 18 | 5,480 | 337 | 16.4 | 3.76 | ≈ 0 | 0.775 |
| L2 ESM-2 zero-shot | 1 (ESM-2 LLR) | 0 (pretrained only) | | 0 | 0 | 1,137 on NVIDIA GeForce RTX 5070 Ti Laptop GPU | 0.752 |
| L2 ESM-2 embedding + LR | 481 (ESM-2) | 482 | 14 | 8.0 | 2.07 | 1,137 on NVIDIA GeForce RTX 5070 Ti Laptop GPU | 0.924 |
| L2 trees + ESM-2 score | 19 (incl. ESM-2 LLR) | 12,200 | 707 | 17.2 | 5.13 | 1,137 on NVIDIA GeForce RTX 5070 Ti Laptop GPU | 0.841 |

ESM-2 pass over all 14,743 proteins: 210 s on NVIDIA GeForce RTX 5070 Ti Laptop GPU (18,282 windows of up to 1,022 residues). ESM-2 esm2_t12_35M_UR50D itself is a pretrained network of about 35 M parameters (not trained here).

## Abstention on the test part (ESM-2 embedding + LR)

| Split | Deferred | Error rate on kept | Errors removed | AUROC on kept |
|---|---:|---:|---:|---:|
| gene | 0% | 13.5% | 0.0% | 0.924 |
| gene | 10% | 10.2% | 32.5% | 0.936 |
| gene | 20% | 7.3% | 56.8% | 0.946 |
| gene | 30% | 5.3% | 72.8% | 0.955 |
| random | 0% | 13.6% | 0.0% | 0.923 |
| random | 10% | 10.2% | 32.5% | 0.935 |
| random | 20% | 7.4% | 56.3% | 0.945 |
| random | 30% | 5.3% | 72.8% | 0.955 |
