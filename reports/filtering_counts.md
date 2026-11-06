# Dataset filtering counts

Source: reports/data_snapshot.json

| Step | Variants remaining |
|---|---:|
| All rows in the snapshot | 9,345,867 |
| 1. One genome version (GRCh38), one row per variant | 4,636,853 |
| 2. Single-nucleotide variants | 4,285,982 |
| 3. Missense (one amino acid replaced by another) | 2,571,841 |
|    Start-codon changes (p.Met1...) removed | 2,567,977 |
| 4. Label is Pathogenic/Likely pathogenic or Benign/Likely benign | 216,739 |
| 5. At least one review star | 200,778 |
| 6. One row per protein change (duplicates and conflicts removed) | 199,374 |

Step 6 in detail:

- Rows removed because their protein change had contradictory labels: 2 (from 1 protein change(s))
- Duplicate rows of a protein change whose copies agree (one row kept): 1,402

## Final dataset

- Pathogenic (label 1): 59,031 (29.6%)
- Benign (label 0): 140,343 (70.4%)
- Genes: 15,272 (with both classes: 3,281)
