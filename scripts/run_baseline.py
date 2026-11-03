"""Train and evaluate the initial baseline (proposal section 9) under both splits and all seeds.

Run from the repo root:  python scripts/run_baseline.py
Inputs:  data/processed/features_swap.parquet, data/splits/splits.parquet
Outputs: reports/baseline_results.csv        one row per (model, split type, seed)
         reports/baseline_summary.md         mean +/- std over seeds, plus chosen settings
         reports/baseline_coefficients.md    what the logistic regression learned
         reports/figures/baseline_*.png      metric comparison and calibration curves
         models/baseline_lr_<split>_s0.joblib  trained models (seed 0) for download

Models
  logreg       : Level-1 initial baseline. Standardised swap features + L2 logistic regression.
                 The regularisation strength C is tuned by 5-fold gene-grouped cross-validation
                 INSIDE the training part only (proposal section 5), scored by AUPRC.
  gene_prior   : diagnostic only, not a real model. Predicts each variant by the share of
                 pathogenic variants its gene had in training. It knows nothing about the
                 swap, so whatever it scores shows how much "knowing the gene" alone is worth.

Evaluation is on the VAL part. The TEST part is not touched here (it is used once, at the end).
Class weighting is not used: AUROC/AUPRC only depend on the ranking, and weighting would
distort the predicted probabilities that the calibration analysis needs.
"""
import json
import pickle
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features_swap.parquet"
SPLITS = ROOT / "data" / "splits" / "splits.parquet"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

SEEDS = [0, 1, 2, 3, 4]
SPLIT_TYPES = ["random", "gene"]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
METRICS = ["auroc", "auprc", "brier", "per_gene_auroc"]


def per_gene_auroc(genes, y, p):
    """Mean AUROC computed separately inside each gene that has both classes."""
    df = pd.DataFrame({"g": genes, "y": y, "p": p})
    scores = [roc_auc_score(d["y"], d["p"]) for _, d in df.groupby("g") if d["y"].nunique() == 2]
    return (float(np.mean(scores)) if scores else np.nan), len(scores)


def evaluate(genes, y, p):
    pga, n_genes = per_gene_auroc(genes, y, p)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "per_gene_auroc": pga,
            "genes_in_per_gene_auroc": n_genes}


def fit_logreg(X, y, groups, seed):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    search = GridSearchCV(pipe, {"logisticregression__C": C_GRID},
                          scoring="average_precision", cv=GroupKFold(n_splits=5), n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_["logisticregression__C"], \
        time.perf_counter() - t0


def gene_prior(train_genes, train_y, query_genes, strength=1.0):
    """Smoothed share of pathogenic variants per gene; unseen genes get the overall share."""
    overall = train_y.mean()
    stats = pd.DataFrame({"g": train_genes, "y": train_y}).groupby("g")["y"].agg(["sum", "count"])
    rate = (stats["sum"] + strength * overall) / (stats["count"] + strength)
    return pd.Series(query_genes).map(rate).fillna(overall).to_numpy()


def main():
    feats = pd.read_parquet(FEATURES)
    splits = pd.read_parquet(SPLITS)
    assert (feats["variation_id"].to_numpy() == splits["variation_id"].to_numpy()).all(), \
        "features and splits are not in the same order - rerun make_splits.py"
    feature_cols = [c for c in feats.columns if c not in ("variation_id", "gene", "label")]
    X, y, genes = feats[feature_cols].to_numpy(), feats["label"].to_numpy(), feats["gene"].to_numpy()

    for d in (FIGURES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}

    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            part = splits[f"{split_type}_s{seed}"].to_numpy()
            tr, va = part == "train", part == "val"

            model, best_c, train_s = fit_logreg(X[tr], y[tr], genes[tr], seed)
            t0 = time.perf_counter()
            p = model.predict_proba(X[va])[:, 1]
            infer_ms = (time.perf_counter() - t0) / va.sum() * 1000 * 1000  # per 1,000 variants
            rows.append({"model": "logreg", "split": split_type, "seed": seed,
                         **evaluate(genes[va], y[va], p), "best_C": best_c,
                         "train_seconds": train_s, "infer_ms_per_1000": infer_ms,
                         "size_kb": len(pickle.dumps(model)) / 1024,
                         "val_prevalence": y[va].mean()})
            if seed == 0:
                curves[split_type] = (y[va], p)
                joblib.dump(model, MODELS / f"baseline_lr_{split_type}_s0.joblib")
                if split_type == "gene":
                    coef_model = model

            p_prior = gene_prior(genes[tr], y[tr], genes[va])
            rows.append({"model": "gene_prior", "split": split_type, "seed": seed,
                         **evaluate(genes[va], y[va], p_prior),
                         "val_prevalence": y[va].mean()})
            r = rows[-2]
            print(f"{split_type:<6} seed {seed}: logreg AUROC {r['auroc']:.3f}  "
                  f"AUPRC {r['auprc']:.3f}  | gene_prior AUROC {rows[-1]['auroc']:.3f}  "
                  f"(C={best_c}, {train_s:.1f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "baseline_results.csv", index=False)

    # ---- Summary table: mean +/- std over seeds ----
    agg = res.groupby(["model", "split"])[METRICS].agg(["mean", "std"])
    lines = ["# Initial baseline results (validation part, 5 seeds)", "",
             f"Validation prevalence (= AUPRC of a random guess): "
             f"{res['val_prevalence'].mean():.3f}", "",
             "| Model | Split | AUROC | AUPRC | Brier (lower = better) | Per-gene AUROC |",
             "|---|---|---|---|---|---|"]
    for (model_name, split_type), r in agg.iterrows():
        cells = [f"{r[(m, 'mean')]:.3f} ± {r[(m, 'std')]:.3f}" for m in METRICS]
        lines.append(f"| {model_name} | {split_type} | " + " | ".join(cells) + " |")
    lr = res[res["model"] == "logreg"]
    lines += ["", "## Logistic regression: settings and cost", "",
              f"- Chosen C per run: {sorted(lr['best_C'].unique().tolist())}",
              f"- Training time incl. tuning (CPU): {lr['train_seconds'].mean():.1f} s on average",
              f"- Inference: {lr['infer_ms_per_1000'].mean():.3f} ms per 1,000 variants",
              f"- Model size: {lr['size_kb'].mean():.1f} KB",
              f"- Per-gene AUROC uses val genes that contain both classes "
              f"(on average {lr['genes_in_per_gene_auroc'].mean():.0f} genes)."]
    (REPORTS / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Coefficients (seed 0, gene split): what the model learned ----
    coefs = pd.Series(coef_model[-1].coef_[0], index=feature_cols).sort_values()
    cl = ["# Logistic regression coefficients (gene split, seed 0)", "",
          "Features are standardised, so sizes are comparable. Positive = pushes towards "
          "pathogenic.", "", "| Feature | Coefficient |", "|---|---:|"]
    cl += [f"| `{k}` | {v:+.3f} |" for k, v in coefs.items()]
    (REPORTS / "baseline_coefficients.md").write_text("\n".join(cl) + "\n", encoding="utf-8")

    # ---- Figure 1: metric comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, ["auroc", "auprc", "per_gene_auroc"],
                                 ["AUROC", "AUPRC", "Per-gene AUROC"]):
        for i, model_name in enumerate(["logreg", "gene_prior"]):
            m = [agg.loc[(model_name, s), (metric, "mean")] for s in SPLIT_TYPES]
            e = [agg.loc[(model_name, s), (metric, "std")] for s in SPLIT_TYPES]
            ax.bar(np.arange(2) + (i - 0.5) * 0.38, m, 0.38, yerr=e, capsize=4,
                   label={"logreg": "Logistic regression", "gene_prior": "Gene-only reference"}
                   [model_name])
        ax.axhline(res["val_prevalence"].mean() if metric == "auprc" else 0.5,
                   color="grey", ls="--", lw=1)
        ax.set_xticks(range(2), ["Random split", "Gene split"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Initial baseline on the validation part (mean ± std over 5 seeds; "
                 "dashed = random guess)")
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_metrics.png", dpi=150)

    # ---- Figure 2: calibration (reliability) curves, seed 0 ----
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for split_type in SPLIT_TYPES:
        yt, pt = curves[split_type]
        frac, mean_p = calibration_curve(yt, pt, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", label=f"{split_type.capitalize()} split")
    ax.set_xlabel("Predicted probability of pathogenic")
    ax.set_ylabel("Observed share pathogenic")
    ax.set_title("Calibration, logistic regression (seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_calibration.png", dpi=150)

    (MODELS / "baseline_lr_features.json").write_text(json.dumps(feature_cols, indent=2))

    print("\n" + "\n".join(lines))
    print(f"\nSaved results, summary, coefficients, 2 figures and 2 models.")


if __name__ == "__main__":
    main()
