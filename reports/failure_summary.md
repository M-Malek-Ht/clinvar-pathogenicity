# Failure analysis: gene split, mapped validation variants, 5 seeds (ESM-2 esm2_t12_35M_UR50D)

Numbers are means over the 5 seeds; n is variants per seed. Error rate uses a 0.5 threshold. AUROC is blank for one-class slices.

## Genes with few known variants

| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC | ESM LR: mean p | ESM LR: error | ESM LR: AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <= 5 variants | 2,012 | 4.5% | 0.261 | 11.9% | 0.744 | 0.176 | 9.4% | 0.898 |
| 6-20 | 4,753 | 12.8% | 0.276 | 17.3% | 0.776 | 0.206 | 10.8% | 0.901 |
| 21-100 | 5,614 | 33.2% | 0.286 | 25.4% | 0.798 | 0.309 | 15.3% | 0.916 |
| > 100 | 5,758 | 52.7% | 0.315 | 32.6% | 0.835 | 0.440 | 18.3% | 0.924 |

## Benign variants in disease-heavy genes (false-positive rate)

| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC | ESM LR: mean p | ESM LR: error | ESM LR: AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benign, disease-heavy genes | 835 | 0.0% | 0.254 | 11.2% |  | 0.219 | 13.7% |  |
| benign, other genes | 11,702 | 0.0% | 0.220 | 8.6% |  | 0.141 | 6.9% |  |

## Pathogenic variants with mild-looking swaps (false-negative rate)

| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC | ESM LR: mean p | ESM LR: error | ESM LR: AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pathogenic, mild swap (BLOSUM62 >= 1) | 971 | 100.0% | 0.225 | 97.6% |  | 0.580 | 41.4% |  |
| pathogenic, harsh swap (BLOSUM62 <= 0) | 4,629 | 100.0% | 0.485 | 50.1% |  | 0.693 | 27.9% |  |

## Recently classified variants

| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC | ESM LR: mean p | ESM LR: error | ESM LR: AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| evaluated 2025-2026 | 8,539 | 28.3% | 0.283 | 23.2% | 0.770 | 0.300 | 13.7% | 0.920 |
| evaluated before 2025 | 9,420 | 32.3% | 0.293 | 24.3% | 0.788 | 0.311 | 14.8% | 0.918 |
| date missing | 177 | 83.9% | 0.420 | 50.4% | 0.680 | 0.609 | 26.4% | 0.851 |

## Label confidence (ClinVar review stars)

| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC | ESM LR: mean p | ESM LR: error | ESM LR: AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 star | 12,025 | 26.6% | 0.283 | 20.9% | 0.796 | 0.282 | 13.1% | 0.921 |
| 2+ stars | 6,112 | 39.3% | 0.302 | 30.3% | 0.756 | 0.362 | 17.1% | 0.911 |

## Low-confidence predictions (bins are per model)

| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC | ESM LR: mean p | ESM LR: error | ESM LR: AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| |p - 0.5| in [0.0, 0.1) | 2,690 / 1,547 | 52.1% | 0.495 | 45.5% | 0.568 | 0.494 | 45.7% | 0.558 |
| |p - 0.5| in [0.1, 0.2) | 3,106 / 1,705 | 46.9% | 0.451 | 34.8% | 0.667 | 0.464 | 35.4% | 0.652 |
| |p - 0.5| in [0.2, 0.3) | 4,022 / 2,003 | 36.2% | 0.348 | 25.0% | 0.708 | 0.402 | 26.6% | 0.729 |
| |p - 0.5| in [0.3, 0.4) | 4,853 / 2,981 | 20.9% | 0.189 | 16.2% | 0.646 | 0.321 | 14.3% | 0.835 |
| |p - 0.5| in [0.4, 0.5] | 3,465 / 9,900 | 7.8% | 0.058 | 7.7% | 0.695 | 0.231 | 3.5% | 0.967 |

## Mitigation 1: recalibration on separate validation genes

Isotonic / Platt fitted on half of the validation genes, applied to the other half, then swapped.

| Model | Method | Brier | ECE | AUROC |
|---|---|---|---|---|
| Level 1 trees | none | 0.166 ± 0.002 | 0.022 ± 0.007 | 0.780 |
| Level 1 trees | isotonic | 0.171 ± 0.006 | 0.031 ± 0.016 | 0.764 |
| Level 1 trees | platt | 0.170 ± 0.005 | 0.018 ± 0.008 | 0.768 |
| ESM-2 embedding + LR | none | 0.102 ± 0.004 | 0.009 ± 0.002 | 0.918 |
| ESM-2 embedding + LR | isotonic | 0.105 ± 0.006 | 0.013 ± 0.009 | 0.914 |
| ESM-2 embedding + LR | platt | 0.104 ± 0.006 | 0.009 ± 0.003 | 0.915 |

## Mitigation 2: abstain on the least confident variants

| Model | Deferred to expert review | Error rate on kept | Errors removed | AUROC on kept |
|---|---|---:|---:|---:|
| Level 1 trees | 0% | 24.1% | 0.0% | 0.780 |
| Level 1 trees | 10% | 21.4% | 19.8% | 0.780 |
| Level 1 trees | 20% | 19.1% | 36.4% | 0.772 |
| Level 1 trees | 30% | 17.0% | 50.5% | 0.760 |
| ESM-2 embedding + LR | 0% | 14.4% | 0.0% | 0.918 |
| ESM-2 embedding + LR | 10% | 11.0% | 31.1% | 0.931 |
| ESM-2 embedding + LR | 20% | 8.2% | 54.7% | 0.943 |
| ESM-2 embedding + LR | 30% | 5.8% | 72.0% | 0.953 |

- Unmapped variants (7.2%) have no ESM-2 score; in deployment they would fall back to the Level 1 trees or be deferred.
