"""Level 2, part 2: evaluate ESM-2 zero-shot, an ESM-2 embedding classifier, and trees + ESM score.

Run from the repo root (after esm_scores.py):
    python scripts/run_esm_models.py --model esm2_t12_35M_UR50D
Inputs:  data/processed/esm_<model>.parquet, features_swap.parquet, variants_mapped.parquet,
         data/splits/splits.parquet, reports/gbm_position_results.csv (for comparison)
Outputs: reports/esm_<model>_results.csv, reports/esm_<model>_summary.md,
         models/esm_<model>_{lr,gbm}_<split>_s0.joblib

All three are evaluated on the MAPPED variants of the validation part (ESM-2 needs a sequence),
with the same splits, seeds and gene-grouped tuning as the Level-1 models:
  zero_shot     -esm_llr as the score; no training at all
  emb_lr        L2 logistic regression on the ESM-2 embedding + esm_llr (the proposal's
                "small classifier on top"), C tuned by 5-fold GroupKFold on train, AUPRC
  gbm_esm       the Level-1 trees (swap + length + position) with esm_llr added as one feature,
                same grid as run_gbm.py
For comparison the summary also shows the Level-1 trees (swap + length + position) on the same
mapped validation variants (from run_gbm_position.py).
"""
import argparse
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS, MODELS = ROOT / "reports", ROOT / "models"
SEEDS = [0, 1, 2, 3, 4]
SPLIT_TYPES = ["random", "gene"]
GBM_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 31],
            "min_samples_leaf": [20, 200]}
LR_GRID = {"logisticregression__C": [0.01, 0.1, 1]}


def per_gene_auroc(genes, y, p):
    df = pd.DataFrame({"g": genes, "y": y, "p": p})
    s = [roc_auc_score(d["y"], d["p"]) for _, d in df.groupby("g") if d["y"].nunique() == 2]
    return float(np.mean(s)) if s else np.nan


def ece(y, p, n_bins=10):
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def evaluate(genes, y, p, probabilities=True):
    r = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
         "per_gene_auroc": per_gene_auroc(genes, y, p)}
    if probabilities:  # the zero-shot score is not a probability: no Brier or ECE
        r.update(brier=brier_score_loss(y, p), ece=ece(y, p))
    return r


def tune(estimator, grid, X, y, groups):
    search = GridSearchCV(estimator, grid, scoring="average_precision",
                          cv=GroupKFold(n_splits=5), n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    args = ap.parse_args()
    tag = f"esm_{args.model}"

    feats = pd.read_parquet(PROC / "features_swap.parquet")
    mapped = pd.read_parquet(PROC / "variants_mapped.parquet")
    splits = pd.read_parquet(ROOT / "data" / "splits" / "splits.parquet")
    for name, d in [("variants_mapped", mapped), ("splits", splits)]:
        if not (feats["variation_id"].to_numpy() == d["variation_id"].to_numpy()).all():
            raise RuntimeError(f"features_swap and {name} are not in the same row order.")
    esm = pd.read_parquet(PROC / f"{tag}.parquet")

    keep = mapped["mapped"].to_numpy()
    base = pd.concat([feats, mapped[["protein_length", "rel_position"]]], axis=1)[keep]
    base = base.merge(esm, on="variation_id", how="left", validate="one_to_one")
    if base["esm_llr"].isna().any():
        raise RuntimeError("Some mapped variants have no ESM-2 score: rerun esm_scores.py.")
    splits = splits[keep].reset_index(drop=True)
    y, genes = base["label"].to_numpy(), base["gene"].to_numpy()
    swap_cols = [c for c in feats.columns if c not in ("variation_id", "gene", "label")]
    emb_cols = [c for c in base.columns if c.startswith("emb_")]
    X_emb = base[emb_cols + ["esm_llr"]].to_numpy(dtype=np.float32)
    X_gbm = base[swap_cols + ["protein_length", "rel_position", "esm_llr"]].to_numpy(dtype=float)
    llr = base["esm_llr"].to_numpy()
    print(f"{len(base):,} mapped variants, embedding size {len(emb_cols)}")

    MODELS.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            part = splits[f"{split_type}_s{seed}"].to_numpy()
            tr, va = part == "train", part == "val"
            g_va, y_va = genes[va], y[va]

            rows.append({"model": "zero_shot", "split": split_type, "seed": seed,
                         **evaluate(g_va, y_va, -llr[va], probabilities=False)})

            lr, lr_params, lr_s = tune(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                LR_GRID, X_emb[tr], y[tr], genes[tr])
            t0 = time.perf_counter()
            p = lr.predict_proba(X_emb[va])[:, 1]
            rows.append({"model": "emb_lr", "split": split_type, "seed": seed,
                         **evaluate(g_va, y_va, p), **lr_params, "train_seconds": lr_s,
                         "infer_ms_per_1000": (time.perf_counter() - t0) / va.sum() * 1e6,
                         "size_kb": len(pickle.dumps(lr)) / 1024})

            gbm, gbm_params, gbm_s = tune(
                HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                               random_state=seed),
                GBM_GRID, X_gbm[tr], y[tr], genes[tr])
            t0 = time.perf_counter()
            p = gbm.predict_proba(X_gbm[va])[:, 1]
            rows.append({"model": "gbm_esm", "split": split_type, "seed": seed,
                         **evaluate(g_va, y_va, p), **gbm_params, "train_seconds": gbm_s,
                         "infer_ms_per_1000": (time.perf_counter() - t0) / va.sum() * 1e6,
                         "size_kb": len(pickle.dumps(gbm)) / 1024})
            if seed == 0:
                joblib.dump(lr, MODELS / f"{tag}_lr_{split_type}_s0.joblib")
                joblib.dump(gbm, MODELS / f"{tag}_gbm_{split_type}_s0.joblib")
            z, e, gb = rows[-3], rows[-2], rows[-1]
            print(f"{split_type:<6} seed {seed}: zero-shot AUROC {z['auroc']:.3f} | "
                  f"emb_lr {e['auroc']:.3f} ({lr_s:.0f}s) | gbm_esm {gb['auroc']:.3f} "
                  f"({gbm_s:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / f"{tag}_results.csv", index=False)

    metrics = ["auroc", "auprc", "brier", "ece", "per_gene_auroc"]
    agg = res.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    lv1 = pd.read_csv(REPORTS / "gbm_position_results.csv").groupby("split")[
        ["auroc_mapped_only", "auprc_mapped_only"]].agg(["mean", "std"])
    lines = [f"# Level 2: ESM-2 ({args.model}), mapped validation variants, 5 seeds", "",
             "| Model | Split | AUROC | AUPRC | Brier | ECE | Per-gene AUROC |",
             "|---|---|---|---|---|---|---|"]
    for split_type in SPLIT_TYPES:
        a, b = lv1.loc[split_type, "auroc_mapped_only"], lv1.loc[split_type, "auprc_mapped_only"]
        lines.append(f"| Level 1 trees (swap + position) | {split_type} | {a['mean']:.3f} ± "
                     f"{a['std']:.3f} | {b['mean']:.3f} ± {b['std']:.3f} | | | |")
    names = {"zero_shot": "ESM-2 zero-shot (no training)", "emb_lr": "ESM-2 embedding + LR",
             "gbm_esm": "Trees + ESM-2 score"}
    for model_name in ["zero_shot", "emb_lr", "gbm_esm"]:
        for split_type in SPLIT_TYPES:
            r = agg.loc[(model_name, split_type)]
            cells = ["" if pd.isna(r[(m, "mean")]) else
                     f"{r[(m, 'mean')]:.3f} ± {r[(m, 'std')]:.3f}" for m in metrics]
            lines.append(f"| {names[model_name]} | {split_type} | " + " | ".join(cells) + " |")
    cost = res[res["model"] != "zero_shot"].groupby("model")[
        ["train_seconds", "infer_ms_per_1000", "size_kb"]].mean()
    lines += ["", "Cost (mean over runs; ESM-2 inference itself is in reports/"
              f"{tag}_run.json):", ""]
    for model_name, r in cost.iterrows():
        lines.append(f"- {names[model_name]}: training incl. tuning {r['train_seconds']:.1f} s, "
                     f"inference {r['infer_ms_per_1000']:.2f} ms per 1,000, "
                     f"{r['size_kb']:.0f} KB")
    (REPORTS / f"{tag}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
