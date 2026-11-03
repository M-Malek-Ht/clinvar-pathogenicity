"""Create the two evaluation splits, for several seeds (proposal section 6).

Run from the repo root:  python scripts/make_splits.py
Input:   data/processed/variants.parquet
Outputs: data/splits/splits.parquet   one row per variant, one column per (split type, seed),
                                      each value is "train", "val" or "test"
         reports/splits_summary.md    sizes, class balance and gene overlap of every split

Two split types:
  random : variants are shuffled at random. A test variant's gene is usually also in training
           ("new variants in familiar genes").
  gene   : whole genes are assigned to one side. No test gene is ever seen in training
           ("unseen genes"). The gap between the two is a main result of the project.

Proportions: 70% train, 10% val, 20% test, with the pathogenic/benign ratio kept similar
in every part. Done by cutting the data into 10 equal folds: folds 0-1 = test, fold 2 = val,
folds 3-9 = train.

How each part is used:
  train : model fitting, and hyperparameter tuning by cross-validation inside train only
  val   : preliminary results (progress report) and, later, recalibration
  test  : touched ONCE, for the final results

The splits file is committed to git so both teammates use identical splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
IN_DATA = ROOT / "data" / "processed" / "variants.parquet"
OUT_SPLITS = ROOT / "data" / "splits" / "splits.parquet"
OUT_SUMMARY = ROOT / "reports" / "splits_summary.md"

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 10
FOLD_TO_PART = {0: "test", 1: "test", 2: "val"}  # every other fold -> "train"


def assign(fold_iter, n):
    """Turn a 10-fold splitter into a train/val/test label per variant."""
    part = np.empty(n, dtype=object)
    for fold, (_, idx) in enumerate(fold_iter):
        part[idx] = FOLD_TO_PART.get(fold, "train")
    return part


def main():
    v = pd.read_parquet(IN_DATA, columns=["variation_id", "gene", "label"])
    y, genes = v["label"].to_numpy(), v["gene"].to_numpy()
    out = pd.DataFrame({"variation_id": v["variation_id"]})

    rows = []
    for seed in SEEDS:
        rand = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(v, y)
        grp = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(v, y, genes)
        for kind, folds in [("random", rand), ("gene", grp)]:
            col = f"{kind}_s{seed}"
            out[col] = pd.Categorical(assign(folds, len(v)), categories=["train", "val", "test"])

            train_genes = set(genes[out[col] == "train"])
            for part in ["train", "val", "test"]:
                m = (out[col] == part).to_numpy()
                part_genes = set(genes[m])
                rows.append({
                    "split": col, "part": part,
                    "variants": int(m.sum()),
                    "share": m.mean(),
                    "pathogenic": y[m].mean(),
                    "genes": len(part_genes),
                    "genes_also_in_train": (len(part_genes & train_genes)
                                            if part != "train" else None),
                })
            print(f"done: {col}")

    # Safety check: the gene split must never put the same gene on two sides.
    for seed in SEEDS:
        sides = pd.DataFrame({"gene": genes, "part": out[f"gene_s{seed}"]})
        assert (sides.groupby("gene")["part"].nunique() == 1).all(), f"gene leak in seed {seed}"

    OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_SPLITS, index=False)

    s = pd.DataFrame(rows)
    target = s["part"].map({"train": 0.7, "val": 0.1, "test": 0.2})
    uneven = s[((s["share"] - target).abs() > 0.03)
               | ((s["pathogenic"] - y.mean()).abs() > 0.03)]
    if len(uneven):
        print("\nWARNING - these parts are uneven (share or % pathogenic off by >3 points):")
        print(uneven[["split", "part", "share", "pathogenic"]].to_string(index=False))

    md =["# Evaluation splits", "",
          f"Seeds: {SEEDS}. Target proportions 70 / 10 / 20 (train / val / test).", "",
          "| Split | Part | Variants | Share | % pathogenic | Genes | Genes also in train |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in s.itertuples():
        overlap = "-" if r.genes_also_in_train is None else f"{r.genes_also_in_train:,}"
        md.append(f"| {r.split} | {r.part} | {r.variants:,} | {r.share:.1%} | "
                  f"{r.pathogenic:.1%} | {r.genes:,} | {overlap} |")
    OUT_SUMMARY.write_text("\n".join(md) + "\n", encoding="utf-8")

    show = s[s["split"].str.endswith("_s0")].copy()
    show["share"] = show["share"].map("{:.1%}".format)
    show["pathogenic"] = show["pathogenic"].map("{:.1%}".format)
    print("\nSeed 0 (all seeds are in reports/splits_summary.md):")
    print(show.drop(columns="split").assign(split=show["split"]).set_index(["split", "part"])
          .to_string())
    print("\nGene-leak check passed: no gene appears on two sides of any gene split.")
    print(f"\nSaved {OUT_SPLITS}\nSaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
