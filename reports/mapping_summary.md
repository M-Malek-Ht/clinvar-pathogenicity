# Mapping variants to UniProt sequences

| Status | Variants | Share | % pathogenic | Genes |
|---|---:|---:|---:|---:|
| mapped | 185,104 | 92.8% | 30.5% | 14,731 |
| residue mismatch | 12,774 | 6.4% | 18.2% | 1,216 |
| position beyond protein length | 848 | 0.4% | 9.1% | 250 |
| gene not in UniProt | 648 | 0.3% | 18.7% | 119 |

- Mapped proteins longer than 1022 residues: 2,162 proteins holding 83,448 variants (45.1% of mapped). ESM-2 reads these in overlapping 1,022-residue windows (step 511), each variant from the window whose centre is closest to it.
- Genes with both classes among mapped variants: 3,120
- Runtime: 0.8 s
