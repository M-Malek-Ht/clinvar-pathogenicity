"""Feature analysis: add protein length and relative position to the gradient-boosted trees.

Run from the repo root:  python scripts/run_gbm_position.py
Inputs:  data/processed/features_swap.parquet, data/processed/variants_mapped.parquet,
         data/splits/splits.parquet, reports/gbm_results.csv (from run_gbm.py, for comparison)
Outputs: reports/gbm_position_results.csv   one row per (split type, seed)
         reports/gbm_position_summary.md    swap-only vs swap + position, both splits
         models/gbm_position_<split>_s0.joblib

Same model, grid and tuning protocol as run_gbm.py; only two features are added:
  protein_length  length of the UniProt protein
  rel_position    position of the variant / protein length (0 = start, 1 = end)
Both are missing (NaN) for the 7.2% of variants that did not map to UniProt; the trees handle
missing values natively, so every variant is kept and the numbers stay comparable with
run_gbm.py. Results are also reported on mapped variants only.

Why both splits matter here: protein length is almost a gene fingerprint (most genes have a
unique length), so on the random split the model can partly recognise genes again. Any gain
that appears on the random split but not on the gene split is gene recognition, not biology.
"""
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features_swap.parquet"
MAPPED = ROOT / "data" / "processed" / "variants_mapped.parquet"
SPLITS = ROOT / "data" / "splits" / "splits.parquet"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

SEEDS = [0, 1, 2, 3, 4]
SPLIT_TYPES = ["random", "gene"]
GRID = {"learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [7, 31],
        "min_samples_leaf": [20, 200]}


def per_gene_auroc(genes, y, p):
    df = pd.DataFrame({"g": genes, "y": y, "p": p})
    scores = [roc_auc_score(d["y"], d["p"]) for _, d in df.groupby("g") if d["y"].nunique() == 2]
    return (float(np.mean(scores)) if scores else np.nan), len(scores)


def ece(y, p, n_bins=10):
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def evaluate(genes, y, p):
    pga, n_genes = per_gene_auroc(genes, y, p)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece(y, p), "per_gene_auroc": pga,
            "genes_in_per_gene_auroc": n_genes}


def main():
    feats = pd.read_parquet(FEATURES)
    mapped = pd.read_parquet(MAPPED)
    splits = pd.read_parquet(SPLITS)
    for name, d in [("variants_mapped", mapped), ("splits", splits)]:
        if not (feats["variation_id"].to_numpy() == d["variation_id"].to_numpy()).all():
            raise RuntimeError(f"features_swap and {name} are not in the same row order.")

    swap_cols = [c for c in feats.columns if c not in ("variation_id", "gene", "label")]
    X = pd.concat([feats[swap_cols], mapped[["protein_length", "rel_position"]]], axis=1)
    feature_cols = list(X.columns)
    X = X.to_numpy(dtype=float)
    y, genes = feats["label"].to_numpy(), feats["gene"].to_numpy()
    is_mapped = mapped["mapped"].to_numpy()
    print(f"{len(feature_cols)} features; {is_mapped.mean():.1%} of variants mapped")

    MODELS.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            part = splits[f"{split_type}_s{seed}"].to_numpy()
            tr, va = part == "train", part == "val"
            search = GridSearchCV(
                HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                               random_state=seed),
                GRID, scoring="average_precision", cv=GroupKFold(n_splits=5), n_jobs=-1)
            t0 = time.perf_counter()
            search.fit(X[tr], y[tr], groups=genes[tr])
            train_s = time.perf_counter() - t0
            model = search.best_estimator_
            t0 = time.perf_counter()
            p = model.predict_proba(X[va])[:, 1]
            infer_ms = (time.perf_counter() - t0) / va.sum() * 1e6
            vm = is_mapped[va]
            rows.append({"model": "gbm_position", "split": split_type, "seed": seed,
                         **evaluate(genes[va], y[va], p),
                         "auroc_mapped_only": roc_auc_score(y[va][vm], p[vm]),
                         "auprc_mapped_only": average_precision_score(y[va][vm], p[vm]),
                         **search.best_params_, "train_seconds": train_s,
                         "infer_ms_per_1000": infer_ms,
                         "size_kb": len(pickle.dumps(model)) / 1024,
                         "p_min": float(p.min()), "p_max": float(p.max())})
            if seed == 0:
                joblib.dump(model, MODELS / f"gbm_position_{split_type}_s0.joblib")
            r = rows[-1]
            print(f"{split_type:<6} seed {seed}: AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  "
                  f"per-gene {r['per_gene_auroc']:.3f}  ({search.best_params_}, {train_s:.1f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "gbm_position_results.csv", index=False)

    metrics = ["auroc", "auprc", "brier", "ece", "per_gene_auroc"]
    base = pd.read_csv(REPORTS / "gbm_results.csv")
    base = base[base["model"] == "gbm"]
    both = pd.concat([base, res], ignore_index=True)
    agg = both.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    lines = ["# Adding protein length and relative position (validation part, 5 seeds)", "",
             "| Features | Split | AUROC | AUPRC | Brier | ECE | Per-gene AUROC |",
             "|---|---|---|---|---|---|---|"]
    names = {"gbm": "Swap only (16)", "gbm_position": "Swap + length + position (18)"}
    for model_name in ["gbm", "gbm_position"]:
        for split_type in SPLIT_TYPES:
            r = agg.loc[(model_name, split_type)]
            cells = [f"{r[(m, 'mean')]:.3f} ± {r[(m, 'std')]:.3f}" for m in metrics]
            lines.append(f"| {names[model_name]} | {split_type} | " + " | ".join(cells) + " |")
    gap = {s: agg.loc[("gbm_position", s), ("auroc", "mean")]
           - agg.loc[("gbm", s), ("auroc", "mean")] for s in SPLIT_TYPES}
    lines += ["",
              f"- AUROC gain from length + position: random {gap['random']:+.3f}, "
              f"gene {gap['gene']:+.3f}. A larger gain on the random split than on the gene "
              "split is gene recognition, not biology.",
              f"- Mapped variants only (swap + position): AUROC "
              f"{res.groupby('split')['auroc_mapped_only'].mean().round(3).to_dict()}, AUPRC "
              f"{res.groupby('split')['auprc_mapped_only'].mean().round(3).to_dict()}",
              f"- Training incl. tuning {res['train_seconds'].mean():.1f} s, inference "
              f"{res['infer_ms_per_1000'].mean():.2f} ms per 1,000, size "
              f"{res['size_kb'].mean():.0f} KB, predictions {res['p_min'].min():.2f} to "
              f"{res['p_max'].max():.2f}"]
    (REPORTS / "gbm_position_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
