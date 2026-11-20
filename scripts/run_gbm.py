"""Strong Level-1 baseline: gradient-boosted trees on the swap features (proposal section 5).

Run from the repo root:  python scripts/run_gbm.py
Inputs:  data/processed/features_swap.parquet, data/splits/splits.parquet
Outputs: reports/gbm_results.csv          one row per (model, split type, seed)
         reports/gbm_summary.md           mean +/- std over seeds, chosen settings, cost
         models/gbm_<split>_s0.joblib     trained models (seed 0)

Models
  gbm          : scikit-learn HistGradientBoostingClassifier on the same 16 swap features as the
                 logistic regression. Hyperparameters tuned by 5-fold gene-grouped CV inside the
                 training part only, scored by AUPRC (same protocol as run_baseline.py).
  swap_lookup  : diagnostic, not a model. Predicts each variant by the share of pathogenic
                 variants with the SAME amino-acid swap (e.g. Arg>Cys) in training. The 16 swap
                 features take only 150 distinct values (one per swap type), so every model that
                 sees only these features is a function of the swap type; this lookup is the best
                 such function the training data supports, i.e. the ceiling for swap-only models.

Evaluation is on the VAL part; the TEST part is not touched. No class weighting (see
run_baseline.py).
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
SPLITS = ROOT / "data" / "splits" / "splits.parquet"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

SEEDS = [0, 1, 2, 3, 4]
SPLIT_TYPES = ["random", "gene"]
GRID = {"learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [7, 31],
        "min_samples_leaf": [20, 200]}
METRICS = ["auroc", "auprc", "brier", "ece", "per_gene_auroc"]


def per_gene_auroc(genes, y, p):
    """Mean AUROC computed separately inside each gene that has both classes."""
    df = pd.DataFrame({"g": genes, "y": y, "p": p})
    scores = [roc_auc_score(d["y"], d["p"]) for _, d in df.groupby("g") if d["y"].nunique() == 2]
    return (float(np.mean(scores)) if scores else np.nan), len(scores)


def ece(y, p, n_bins=10):
    """Expected calibration error, 10 bins with equal numbers of variants (as in the notebook)."""
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def evaluate(genes, y, p):
    pga, n_genes = per_gene_auroc(genes, y, p)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece(y, p), "per_gene_auroc": pga,
            "genes_in_per_gene_auroc": n_genes, "p_min": float(p.min()), "p_max": float(p.max())}


def fit_gbm(X, y, groups, seed):
    """Tune by gene-grouped CV inside the training data; return best model, params, seconds."""
    model = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed)
    search = GridSearchCV(model, GRID, scoring="average_precision",
                          cv=GroupKFold(n_splits=5), n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, time.perf_counter() - t0


def swap_lookup(train_swaps, train_y, query_swaps):
    """Share of pathogenic variants per swap type in training; unseen swaps get the overall share."""
    rate = pd.Series(train_y).groupby(train_swaps).mean()
    return pd.Series(query_swaps).map(rate).fillna(train_y.mean()).to_numpy()


def main():
    feats = pd.read_parquet(FEATURES)
    splits = pd.read_parquet(SPLITS)
    if not (feats["variation_id"].to_numpy() == splits["variation_id"].to_numpy()).all():
        raise RuntimeError("Features and splits are not in the same order: rerun "
                           "build_features.py, then make_splits.py.")
    feature_cols = [c for c in feats.columns if c not in ("variation_id", "gene", "label")]
    X, y, genes = feats[feature_cols].to_numpy(), feats["label"].to_numpy(), feats["gene"].to_numpy()
    # One id per swap type: identical feature vectors <=> identical swap.
    swaps = pd.Series([tuple(r) for r in X]).astype("category").cat.codes.to_numpy()
    print(f"{len(np.unique(swaps))} distinct swap types among {len(X):,} variants")

    MODELS.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            part = splits[f"{split_type}_s{seed}"].to_numpy()
            tr, va = part == "train", part == "val"

            model, params, train_s = fit_gbm(X[tr], y[tr], genes[tr], seed)
            t0 = time.perf_counter()
            p = model.predict_proba(X[va])[:, 1]
            infer_ms = (time.perf_counter() - t0) / va.sum() * 1000 * 1000  # per 1,000 variants
            rows.append({"model": "gbm", "split": split_type, "seed": seed,
                         **evaluate(genes[va], y[va], p), **params,
                         "train_seconds": train_s, "infer_ms_per_1000": infer_ms,
                         "size_kb": len(pickle.dumps(model)) / 1024})
            if seed == 0:
                joblib.dump(model, MODELS / f"gbm_{split_type}_s0.joblib")

            p_lookup = swap_lookup(swaps[tr], y[tr], swaps[va])
            rows.append({"model": "swap_lookup", "split": split_type, "seed": seed,
                         **evaluate(genes[va], y[va], p_lookup)})
            print(f"{split_type:<6} seed {seed}: gbm AUROC {rows[-2]['auroc']:.3f} "
                  f"AUPRC {rows[-2]['auprc']:.3f} | swap_lookup AUROC {rows[-1]['auroc']:.3f} "
                  f"AUPRC {rows[-1]['auprc']:.3f}  ({params}, {train_s:.1f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "gbm_results.csv", index=False)

    # Summary, with the logistic regression from run_baseline.py alongside for comparison.
    both = res
    lr_path = REPORTS / "baseline_results.csv"
    if lr_path.exists():
        lr = pd.read_csv(lr_path)
        lr = lr[lr["model"] == "logreg"]
        both = pd.concat([lr, res], ignore_index=True)
    shown = [m for m in METRICS if m in both.columns]
    agg = both.groupby(["model", "split"])[shown].agg(["mean", "std"])
    lines = ["# Gradient-boosted trees on the swap features (validation part, 5 seeds)", "",
             "| Model | Split | " + " | ".join(m.upper().replace("_", " ") for m in shown) + " |",
             "|---|---|" + "---|" * len(shown)]
    for model_name in ["logreg", "gbm", "swap_lookup"]:
        for split_type in SPLIT_TYPES:
            if (model_name, split_type) not in agg.index:
                continue
            r = agg.loc[(model_name, split_type)]
            cells = ["-" if pd.isna(r[(m, "mean")])
                     else f"{r[(m, 'mean')]:.3f} ± {r[(m, 'std')]:.3f}" for m in shown]
            lines.append(f"| {model_name} | {split_type} | " + " | ".join(cells) + " |")
    g = res[res["model"] == "gbm"]
    chosen = g[list(GRID)].astype(str).agg(", ".join, axis=1).value_counts()
    lines += ["", "ECE for logreg was computed only for seed 0 in the notebook (0.018).", "",
              "## Gradient-boosted trees: settings and cost", "",
              f"- Grid: {GRID}, max_iter=200, tuned by 5-fold GroupKFold on train, scored by AUPRC",
              "- Chosen (learning_rate, max_leaf_nodes, min_samples_leaf) across the 10 runs: "
              + "; ".join(f"{k} x{v}" for k, v in chosen.items()),
              f"- Training time incl. tuning: {g['train_seconds'].mean():.1f} s on average "
              "(wall-clock, parallel; depends on the machine)",
              f"- Inference: {g['infer_ms_per_1000'].mean():.3f} ms per 1,000 variants",
              f"- Model size: {g['size_kb'].mean():.0f} KB",
              f"- Predicted probabilities range {g['p_min'].min():.2f} to {g['p_max'].max():.2f}",
              "", "swap_lookup is a diagnostic: the best any swap-only model can do on this data."]
    (REPORTS / "gbm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
