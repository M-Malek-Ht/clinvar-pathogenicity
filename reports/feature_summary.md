# Swap features (initial baseline)

199,374 variants, 16 features, no missing values: True

| Feature | Meaning | Mean, benign | Mean, pathogenic |
|---|---|---:|---:|
| `blosum62` | BLOSUM62 score of the swap (low = rarely tolerated in evolution) | -0.01 | -1.15 |
| `ref_volume` | Side-chain volume of the original amino acid | 131.08 | 128.60 |
| `alt_volume` | Side-chain volume of the new amino acid | 136.14 | 138.14 |
| `delta_volume` | Change in volume (new - original) | 5.05 | 9.54 |
| `abs_delta_volume` | Size of the volume change, ignoring direction | 35.92 | 50.78 |
| `ref_hydro` | Hydrophobicity of the original amino acid | -0.63 | -0.63 |
| `alt_hydro` | Hydrophobicity of the new amino acid | -0.13 | -0.77 |
| `delta_hydro` | Change in hydrophobicity (new - original) | 0.50 | -0.14 |
| `abs_delta_hydro` | Size of the hydrophobicity change, ignoring direction | 2.18 | 3.15 |
| `ref_charge` | Charge of the original amino acid | 0.10 | 0.08 |
| `alt_charge` | Charge of the new amino acid | 0.05 | 0.07 |
| `delta_charge` | Change in charge (new - original) | -0.05 | -0.01 |
| `charge_changed` | 1 if the charge changes at all | 0.33 | 0.46 |
| `involves_cys` | 1 if cysteine is lost or gained (disulfide bonds) | 0.04 | 0.14 |
| `involves_pro` | 1 if proline is lost or gained (breaks helices) | 0.10 | 0.13 |
| `involves_gly` | 1 if glycine is lost or gained (backbone flexibility) | 0.10 | 0.22 |
