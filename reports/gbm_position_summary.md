# Adding protein length and relative position (validation part, 5 seeds)

| Features | Split | AUROC | AUPRC | Brier | ECE | Per-gene AUROC |
|---|---|---|---|---|---|---|
| Swap only (16) | random | 0.764 ± 0.005 | 0.583 ± 0.004 | 0.168 ± 0.001 | 0.008 ± 0.002 | 0.736 ± 0.008 |
| Swap only (16) | gene | 0.764 ± 0.003 | 0.574 ± 0.007 | 0.169 ± 0.001 | 0.011 ± 0.003 | 0.742 ± 0.009 |
| Swap + length + position (18) | random | 0.831 ± 0.010 | 0.707 ± 0.018 | 0.144 ± 0.004 | 0.021 ± 0.004 | 0.758 ± 0.016 |
| Swap + length + position (18) | gene | 0.782 ± 0.013 | 0.618 ± 0.018 | 0.162 ± 0.004 | 0.021 ± 0.005 | 0.762 ± 0.006 |

- AUROC gain from length + position: random +0.067, gene +0.018. A larger gain on the random split than on the gene split is gene recognition, not biology.
- Mapped variants only (swap + position): AUROC {'gene': 0.781, 'random': 0.834}, AUPRC {'gene': 0.628, 'random': 0.718}
- Training incl. tuning 14.1 s, inference 2.43 ms per 1,000, size 232 KB, predictions 0.00 to 0.97
