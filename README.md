# Predicting Missense Variant Pathogenicity from ClinVar

CMPS 261 — Machine Learning, Fall 2026 · Section L2 · Instructor: Prof. Rida Assaf
Mohamad Ali Al Kammouni · Mohamad Malek Hotait

## Question

Can a model predict whether a **missense variant** (one amino acid replaced by another) is
**pathogenic or benign**, and how much of that accuracy survives on **genes it has never seen**?

Every result is reported under two splits:

| Split | What it tests |
|---|---|
| **Random** | New variants in genes the model has already seen |
| **Gene-grouped** | Variants in genes never seen during training |

The gap between them measures how much a model relies on recognising the gene instead of
learning general rules of protein damage.

## Status (progress report, 6 Nov 2026)

| Phase | Status |
|---|---|
| Environment and hardware record | Done |
| ClinVar snapshot (1 Nov 2026), filtering, leakage audit | Done |
| Swap features (BLOSUM62, volume, hydrophobicity, charge) | Done |
| Random and gene-grouped splits, 5 seeds | Done |
| Initial baseline: logistic regression | Done |
| Extra features (protein length, position, conservation), gradient boosting | Next |
| Level 2: ESM-2 protein language model | Next |
| Calibration of the baseline | Done |
| Failure analysis, recalibration, complexity analysis | Next |

### Initial baseline (validation part, mean ± std over 5 seeds)

| Model | Split | AUROC | AUPRC | Per-gene AUROC |
|---|---|---|---|---|
| Logistic regression (swap features) | Random | 0.731 ± 0.005 | 0.529 ± 0.005 | 0.714 ± 0.005 |
| Logistic regression (swap features) | Gene | 0.732 ± 0.005 | 0.530 ± 0.008 | 0.724 ± 0.004 |
| Gene-only reference (diagnostic) | Random | 0.931 ± 0.002 | 0.862 ± 0.004 | 0.500 |
| Gene-only reference (diagnostic) | Gene | 0.500 | 0.296 | 0.500 |

AUPRC of a random guess = 0.296. The logistic regression is well calibrated on both splits
(expected calibration error 0.018, seed 0), but rarely confident: its predictions range from
0.07 to 0.80. The test part is untouched and is used once, for the final results.
Full details: [`reports/baseline_summary.md`](reports/baseline_summary.md).

## Repository layout

```
scripts/        the pipeline, run in the order below; inspect_clinvar.py (optional) lists the
                raw file's columns -> reports/clinvar_columns.txt
src/            shared code (from Phase 2 on; empty for now)
reports/        every generated table, figure and record (committed)
  figures/      plots
data/raw/       ClinVar download (not committed, ~460 MB)
data/processed/ variants.parquet = the filtered dataset (committed, ~5 MB);
                features are regenerated from it by build_features.py
data/splits/    train/val/test assignment per variant (committed, so splits are identical)
models/         trained models (committed, downloadable)
notebooks/      walkthrough notebook
```

## Reproducing the results

**Requirements:** Windows, macOS or Linux; [Miniconda](https://www.anaconda.com/download).
PyTorch and an NVIDIA GPU are needed only for the ESM-2 stage (later), not for Phase 1. The
machine we used is recorded in [`reports/environment.md`](reports/environment.md).

### 1. Environment

```bash
git clone https://github.com/M-Malek-Ht/clinvar-pathogenicity.git
cd clinvar-pathogenicity
conda env create -f environment.yml
conda activate clinvar
pip install torch --index-url https://download.pytorch.org/whl/cu128   # later, for ESM-2 (NVIDIA GPU)
```

`environment.yml` pins the minor versions of the main packages we used (e.g. `pandas=3.1`), on any platform. Without an
NVIDIA GPU (e.g. on macOS), install PyTorch with `pip install torch` instead.
On **Windows**, `conda env create -f environment-lock.yml` reproduces our environment exactly,
every package and build included (it was exported on Windows 11 and only works there).

### 2a. Quick reproduction — exact results, no download (about 2 minutes)

The filtered dataset is committed, so the download and filtering steps can be skipped:

```bash
python scripts/build_features.py    # amino-acid swap features          -> reports/feature_summary.md
python scripts/run_baseline.py      # train and evaluate the baseline   -> reports/baseline_summary.md
```

The train/val/test splits are committed in `data/splits/`, so `make_splits.py` does not need to be
run (it would regenerate the same splits). These two scripts overwrite the committed baseline
outputs in `reports/` and `models/`; the metrics come out the same, while timings depend on the
machine.

### 2b. Full reproduction — from the raw ClinVar file

Needs the original snapshot (see [Data snapshot](#data-snapshot)). The download is about 460 MB
(about 25 minutes on our connection); every other step takes seconds to a few minutes.

```bash
python scripts/env_report.py        # record hardware and software      -> reports/environment.md
python scripts/download_clinvar.py  # get or verify the ClinVar file    -> reports/data_snapshot.json
python scripts/build_dataset.py     # filter missense P/LP vs B/LB      -> reports/filtering_counts.md
python scripts/build_features.py    # amino-acid swap features          -> reports/feature_summary.md
python scripts/make_splits.py       # random + gene splits, 5 seeds     -> reports/splits_summary.md
python scripts/run_baseline.py      # train and evaluate the baseline   -> reports/baseline_summary.md
```

### 3. Check that you reproduced the results

| Check | Expected |
|---|---|
| Variants in `data/processed/variants.parquet` | 199,374 (59,031 pathogenic, 140,343 benign) |
| `build_features.py` | 16 features, 0 missing values |
| Logistic regression, `reports/baseline_summary.md` | AUROC 0.731 (random) and 0.732 (gene) |
| Gene-only reference | AUROC 0.931 (random) and 0.500 (gene) |

Tiny differences in the last decimal can appear with other package versions, and on another
machine the tuning may pick a different `C` in a run without changing the results (the tuning
curve is flat).

### Walkthrough notebook

[`notebooks/01_phase1_baseline.ipynb`](notebooks/01_phase1_baseline.ipynb) explains every step,
with exploration plots, the tuning curve, results and calibration. It reads the outputs of the
scripts above, so run the quick reproduction (2a) first on a fresh clone. Setting
`RUN_PIPELINE = True` in its first cells reruns the full pipeline (2b), which needs the original
ClinVar file in `data/raw/`.

> **Windows + VS Code:** open VS Code from an activated environment
> (`conda activate clinvar`, then `code .` in the project folder). If VS Code starts the kernel
> without activating the environment, Windows can load mismatched DLLs and the kernel crashes
> on the first plot.

### Data snapshot

NCBI regularly replaces this file with newer releases, so the project is pinned to one snapshot:

| | |
|---|---|
| File | `variant_summary.txt.gz` from <https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/> |
| Release | 1 November 2026 |
| Size | 457,862,838 bytes |
| MD5 | `d7e0d7e581e220100fc84ca64fe953ba` |

If you have the snapshot, place it in `data/raw/` and `download_clinvar.py` checks it against the
MD5 above, stopping with an error if it differs. Without it, the script downloads NCBI's current
file:

- if it is still the same release, it is used;
- if the download does not match NCBI's own published checksum, it is deleted as broken and the
  script stops (run it again);
- if NCBI has published a newer release, it is saved separately as
  `variant_summary_NEWER.txt.gz` and **not** used, the snapshot record is left unchanged, and the
  script stops with an error. Use the quick reproduction (2a) to get our exact results.

`build_dataset.py` also verifies the file against the recorded MD5 before filtering and refuses
to run on any other snapshot. To rebuild deliberately with newer data, rename the file to
`variant_summary.txt.gz` and run `python scripts/build_dataset.py --allow-different-snapshot`
(all counts and results will then differ slightly from ours).

ClinVar data are public domain (NCBI), which allows us to share the filtered dataset.

## Dataset

From 9,345,867 rows to **199,374 missense variants** in **15,272 genes**
(59,031 pathogenic, 140,343 benign). Steps:

1. One genome version (GRCh38), one row per variant
2. Single-nucleotide variants
3. Missense only; start-codon changes removed
4. Label Pathogenic / Likely pathogenic (1) or Benign / Likely benign (0); VUS and conflicting dropped
5. At least one ClinVar review star
6. One row per protein change: 1,402 duplicates removed, and the one protein change with
   contradictory labels (2 records) dropped

Counts after every step: [`reports/filtering_counts.md`](reports/filtering_counts.md).

**Leakage rules.** Review status (stars) is used only for filtering, and the last-evaluated date
only to define a failure-analysis slice. Neither is ever a model feature. Submitter counts,
condition names and existing predictor scores (several were trained on ClinVar) are excluded.

## Trained models

`models/baseline_lr_random_s0.joblib` and `models/baseline_lr_gene_s0.joblib` are scikit-learn
pipelines (scaler + logistic regression). The feature order is in `models/baseline_lr_features.json`.

After the quick reproduction (which creates `features_swap.parquet`):

```python
import json
from pathlib import Path
import joblib, pandas as pd
model = joblib.load("models/baseline_lr_gene_s0.joblib")
features = json.loads(Path("models/baseline_lr_features.json").read_text())
X = pd.read_parquet("data/processed/features_swap.parquet")[features].to_numpy()
p = model.predict_proba(X)[:, 1]   # probability of pathogenic
```

The models were saved with scikit-learn 1.9.1; load them with the same version.

## References

- Landrum et al. (2018). ClinVar: improving access to variant interpretations and supporting evidence. *Nucleic Acids Research*.
- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2). *Science*.
- Grimm et al. (2015). The evaluation of tools used to predict the impact of missense variants is hindered by two types of circularity. *Human Mutation*.
- Karczewski et al. (2020). The mutational constraint spectrum quantified from variation in 141,456 humans (gnomAD). *Nature*.
- Henikoff & Henikoff (1992). Amino acid substitution matrices from protein blocks (BLOSUM62). *PNAS*.
- Kyte & Doolittle (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*.
- Zamyatnin (1972). Protein volume in solution. *Progress in Biophysics and Molecular Biology*.

## License

Code: MIT License (see [`LICENSE`](LICENSE)). ClinVar data: public domain, from NCBI.
