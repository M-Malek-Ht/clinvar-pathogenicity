"""Final evaluation: the one and only use of the TEST part, plus the complexity table.

Run from the repo root (after run_esm_models.py and failure_analysis.py):
    python scripts/final_test.py --model esm2_t12_35M_UR50D
Inputs:  data/processed/{variants_mapped,features_swap,esm_<model>}.parquet,
         data/splits/splits.parquet, reports/esm_<model>_run.json
Outputs: reports/final_test_results.csv     one row per (model, split, seed) on the test part
         reports/final_test_bootstrap.csv   gene-level bootstrap 95% CIs (seed 0)
         reports/final_test_summary.md      test table, CIs, complexity table, abstention
         reports/figures/final_test_{metrics,calibration}.png
         models/final_esm_lr.joblib + models/final_esm_lr.json   the released model
         data/processed/test_predictions.parquet   per-variant test predictions (seed 0)

Protocol (fixed before looking at the test part):
  - For every split type and seed, each model is re-tuned and refit on train + val together
    (5-fold GroupKFold by gene, AUPRC, same grids as before), then scored once on test.
  - Main comparison on MAPPED test variants (every model can score them). The Level-1 models
    are also scored on ALL test variants (they handle unmapped variants), column auroc_all.
  - 95% CIs: resample test GENES with replacement (1,000 times), seed 0, both splits; also the
    paired AUROC difference ESM-2 LR minus Level-1 trees.
  - Abstention at the deferral rates from failure_analysis.py, on the test part.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS, MODELS, FIGURES = ROOT / "reports", ROOT / "models", ROOT / "reports" / "figures"
SEEDS = [0, 1, 2, 3, 4]
SPLIT_TYPES = ["random", "gene"]
N_BOOT = 1000
DEFER = [0.0, 0.1, 0.2, 0.3]
GBM_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 31], "min_samples_leaf": [20, 200]}
LR_C_SWAP = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]}
LR_C_EMB = {"logisticregression__C": [0.01, 0.1, 1]}
ORDER = ["gene_prior", "logreg", "gbm", "gbm_position", "zero_shot", "emb_lr", "gbm_esm"]
NAMES = {"gene_prior": "Gene-only reference (diagnostic)", "logreg": "L1 logistic regression (swap)",
         "gbm": "L1 trees (swap)", "gbm_position": "L1 trees (swap + length + position)",
         "zero_shot": "L2 ESM-2 zero-shot", "emb_lr": "L2 ESM-2 embedding + LR",
         "gbm_esm": "L2 trees + ESM-2 score"}
FEATURES = {"logreg": "16 swap", "gbm": "16 swap", "gbm_position": "18", "zero_shot": "1 (ESM-2 LLR)",
            "emb_lr": "481 (ESM-2)", "gbm_esm": "19 (incl. ESM-2 LLR)", "gene_prior": "gene name"}


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
    if probabilities:
        r.update(brier=brier_score_loss(y, p), ece=ece(y, p))
    return r


def tune(estimator, grid, X, y, groups):
    search = GridSearchCV(estimator, grid, scoring="average_precision",
                          cv=GroupKFold(n_splits=5), n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, time.perf_counter() - t0


def gene_prior(train_genes, train_y, query_genes):
    overall = train_y.mean()
    st = pd.DataFrame({"g": train_genes, "y": train_y}).groupby("g")["y"].agg(["sum", "count"])
    rate = (st["sum"] + overall) / (st["count"] + 1)
    return pd.Series(query_genes).map(rate).fillna(overall).to_numpy()


def n_params(model):
    """Learned numbers in the model: LR coefficients, or tree nodes for the GBM."""
    if hasattr(model, "_predictors"):
        return int(sum(len(t.nodes) for it in model._predictors for t in it))
    lr = model[-1]
    return int(lr.coef_.size + lr.intercept_.size)


def gene_bootstrap(genes, y, preds, rng):
    """preds: {name: p}. Returns {name: (auroc samples, auprc samples)} resampling genes."""
    uniq, inv = np.unique(genes, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = {k: ([], []) for k in preds}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        for k, p in preds.items():
            out[k][0].append(roc_auc_score(y[idx], p[idx]))
            out[k][1].append(average_precision_score(y[idx], p[idx]))
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


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
    esm_run = json.loads((REPORTS / f"{tag}_run.json").read_text())

    base = pd.concat([feats, mapped[["protein_length", "rel_position", "mapped"]]], axis=1)
    base = base.merge(esm, on="variation_id", how="left", validate="one_to_one")
    is_mapped = base["mapped"].to_numpy()
    if base.loc[is_mapped, "esm_llr"].isna().any():
        raise RuntimeError("Some mapped variants have no ESM-2 score: rerun esm_scores.py.")
    y, genes = base["label"].to_numpy(), base["gene"].to_numpy()
    swap_cols = [c for c in feats.columns if c not in ("variation_id", "gene", "label")]
    emb_cols = [c for c in base.columns if c.startswith("emb_")]
    X = {"logreg": base[swap_cols].to_numpy(dtype=float),
         "gbm": base[swap_cols].to_numpy(dtype=float),
         "gbm_position": base[swap_cols + ["protein_length", "rel_position"]].to_numpy(dtype=float),
         "emb_lr": base[emb_cols + ["esm_llr"]].to_numpy(dtype=np.float32),
         "gbm_esm": base[swap_cols + ["protein_length", "rel_position", "esm_llr"]]
         .to_numpy(dtype=float)}
    llr = base["esm_llr"].to_numpy()
    print(f"{len(base):,} variants ({is_mapped.sum():,} mapped); refitting on train + val, "
          "scoring the TEST part once")

    def gbm(seed):
        return HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed)

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    specs = {"logreg": (lr, LR_C_SWAP, False), "gbm": (gbm, GBM_GRID, False),
             "gbm_position": (gbm, GBM_GRID, False), "emb_lr": (lr, LR_C_EMB, True),
             "gbm_esm": (gbm, GBM_GRID, True)}

    for d in (MODELS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    rows, boot_rows, abst_rows, cost, seed0 = [], [], [], {}, {}
    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            part = splits[f"{split_type}_s{seed}"].to_numpy()
            fit_all = np.isin(part, ["train", "val"])
            te_all = part == "test"
            te = te_all & is_mapped
            preds = {}
            for name, (make, grid, needs_map) in specs.items():
                fit = fit_all & is_mapped if needs_map else fit_all
                est = make(seed) if make is gbm else make()
                model, params, train_s = tune(est, grid, X[name][fit], y[fit], genes[fit])
                t0 = time.perf_counter()
                p_full = model.predict_proba(X[name][te_all if not needs_map else te])[:, 1]
                infer = (time.perf_counter() - t0) / len(p_full) * 1e6
                if needs_map:
                    p, extra = p_full, {}
                else:
                    p = p_full[is_mapped[te_all]]
                    extra = {"auroc_all": roc_auc_score(y[te_all], p_full),
                             "auprc_all": average_precision_score(y[te_all], p_full)}
                preds[name] = p
                rows.append({"model": name, "split": split_type, "seed": seed,
                             **evaluate(genes[te], y[te], p), **extra,
                             "params": json.dumps(params), "train_seconds": train_s,
                             "infer_ms_per_1000": infer})
                cost.setdefault(name, []).append(
                    {"train_seconds": train_s, "infer_ms_per_1000": infer,
                     "size_kb": len(pickle.dumps(model)) / 1024, "n_params": n_params(model)})
                if name == "emb_lr" and split_type == "gene" and seed == 0:
                    final = model
            preds["zero_shot"] = -llr[te]
            rows.append({"model": "zero_shot", "split": split_type, "seed": seed,
                         **evaluate(genes[te], y[te], preds["zero_shot"], probabilities=False)})
            prior_fit = fit_all
            p_prior = gene_prior(genes[prior_fit], y[prior_fit], genes[te])
            preds["gene_prior"] = p_prior
            rows.append({"model": "gene_prior", "split": split_type, "seed": seed,
                         **evaluate(genes[te], y[te], p_prior)})

            p = preds["emb_lr"]
            wrong, order = (p >= 0.5) != y[te], np.argsort(np.abs(p - 0.5))
            for f in DEFER:
                kept = np.ones(len(p), dtype=bool)
                kept[order[:int(round(f * len(p)))]] = False
                abst_rows.append({"split": split_type, "seed": seed, "deferred": f,
                                  "error_rate_kept": wrong[kept].mean(),
                                  "errors_removed": 1 - wrong[kept].sum() / wrong.sum(),
                                  "auroc_kept": roc_auc_score(y[te][kept], p[kept])})
            if seed == 0:
                seed0[split_type] = (genes[te], y[te], preds, base["variation_id"].to_numpy()[te])
            r = {x["model"]: x["auroc"] for x in rows[-7:]}
            print(f"{split_type:<6} seed {seed}: test AUROC (mapped) " +
                  "  ".join(f"{k} {r[k]:.3f}" for k in ORDER))

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "final_test_results.csv", index=False)

    # ---- Gene-level bootstrap, seed 0 ----
    rng = np.random.default_rng(0)
    for split_type, (g, yt, preds, _) in seed0.items():
        bs = gene_bootstrap(g, yt, preds, rng)
        for name in ORDER:
            a, b = bs[name]
            boot_rows.append({"split": split_type, "model": name,
                              "auroc": roc_auc_score(yt, preds[name]),
                              "auroc_lo": np.percentile(a, 2.5), "auroc_hi": np.percentile(a, 97.5),
                              "auprc": average_precision_score(yt, preds[name]),
                              "auprc_lo": np.percentile(b, 2.5), "auprc_hi": np.percentile(b, 97.5)})
        diff = bs["emb_lr"][0] - bs["gbm_position"][0]
        boot_rows.append({"split": split_type, "model": "diff emb_lr - gbm_position",
                          "auroc": roc_auc_score(yt, preds["emb_lr"]) -
                          roc_auc_score(yt, preds["gbm_position"]),
                          "auroc_lo": np.percentile(diff, 2.5), "auroc_hi": np.percentile(diff, 97.5)})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(REPORTS / "final_test_bootstrap.csv", index=False)

    # ---- Per-variant predictions and the released model ----
    g, yt, preds, vids = seed0["gene"]
    pd.DataFrame({"variation_id": vids, "gene": g, "label": yt,
                  **{f"p_{k}": v for k, v in preds.items()}}).to_parquet(
        PROC / "test_predictions.parquet", index=False)
    joblib.dump(final, MODELS / "final_esm_lr.joblib")
    (MODELS / "final_esm_lr.json").write_text(json.dumps({
        "what": "ESM-2 embedding + L2 logistic regression (StandardScaler + LogisticRegression)",
        "esm_model": esm_run["model"],
        "inputs": [f"emb_0..emb_{len(emb_cols) - 1}: last-layer ESM-2 embedding of the wild-type "
                   "protein at the variant position (see scripts/esm_scores.py)",
                   "esm_llr: log P(mutant) - log P(wild type) at that position"],
        "feature_order": emb_cols + ["esm_llr"],
        "output": "probability that the missense variant is pathogenic (P/LP vs B/LB)",
        "trained_on": "train + val parts of gene split seed 0 (mapped variants)",
        "params": json.loads(res[(res.model == "emb_lr") & (res.split == "gene") &
                                 (res.seed == 0)]["params"].iloc[0]),
        "test_auroc_gene_split_seed0": float(roc_auc_score(yt, preds["emb_lr"]))}, indent=2))

    # ---- Summary ----
    metrics = ["auroc", "auprc", "brier", "ece", "per_gene_auroc"]
    agg = res.groupby(["model", "split"])[metrics + ["auroc_all"]].agg(["mean", "std"])

    def cell(r, m):
        return "" if pd.isna(r[(m, "mean")]) else f"{r[(m, 'mean')]:.3f} ± {r[(m, 'std')]:.3f}"

    lines = [f"# Final test results (ESM-2 {args.model})", "",
             "Each model refit on train + val, scored once on the TEST part; mean ± std over 5 "
             "seeds, mapped test variants. 'AUROC, all' includes unmapped variants (Level 1 "
             "models only).", "",
             "| Model | Split | AUROC | AUPRC | Brier | ECE | Per-gene AUROC | AUROC, all |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        for split_type in SPLIT_TYPES:
            r = agg.loc[(name, split_type)]
            lines.append(f"| {NAMES[name]} | {split_type} | " +
                         " | ".join(cell(r, m) for m in metrics + ["auroc_all"]) + " |")
    lines += ["", "## Gene-level bootstrap 95% CIs (seed 0, 1,000 resamples of test genes)", "",
              "| Model | Split | AUROC [95% CI] | AUPRC [95% CI] |", "|---|---|---|---|"]
    for _, r in boot.iterrows():
        name = NAMES.get(r["model"], "AUROC difference: ESM-2 LR minus L1 trees (swap + position)")
        pr = "" if pd.isna(r.get("auprc")) else f"{r['auprc']:.3f} [{r['auprc_lo']:.3f}, {r['auprc_hi']:.3f}]"
        lines.append(f"| {name} | {r['split']} | {r['auroc']:+.3f} [{r['auroc_lo']:+.3f}, "
                     f"{r['auroc_hi']:+.3f}] | {pr} |" if r["model"].startswith("diff") else
                     f"| {name} | {r['split']} | {r['auroc']:.3f} [{r['auroc_lo']:.3f}, "
                     f"{r['auroc_hi']:.3f}] | {pr} |")
    gap = {n: agg.loc[(n, "random"), ("auroc", "mean")] - agg.loc[(n, "gene"), ("auroc", "mean")]
           for n in ORDER}
    lines += ["", "## Drop from random to gene split (sub-question 1)", "",
              "| Model | AUROC random - gene |", "|---|---:|"]
    lines += [f"| {NAMES[n]} | {gap[n]:+.3f} |" for n in ORDER]

    esm_ms = esm_run["ms_per_1000_variants"]
    lines += ["", "## Complexity (sub-question 2: what the improvement costs)", "",
              "Mean over the 10 refits on train + val. Training includes the hyperparameter "
              "search. Inference is the classifier only; the ESM-2 feature cost is listed "
              "separately (one GPU pass over every protein).", "",
              "| Model | Features | Learned numbers | Size (KB) | Training incl. tuning (s, CPU) "
              "| Inference (ms / 1,000, CPU) | Feature cost (ms / 1,000) | Test AUROC, gene split |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    gpu = f"{esm_ms:,.0f} on {esm_run.get('gpu') or esm_run['device']}"
    for name in ["logreg", "gbm", "gbm_position", "zero_shot", "emb_lr", "gbm_esm"]:
        auc = agg.loc[(name, "gene"), ("auroc", "mean")]
        if name == "zero_shot":
            lines.append(f"| {NAMES[name]} | {FEATURES[name]} | 0 (pretrained only) | | 0 | 0 "
                         f"| {gpu} | {auc:.3f} |")
            continue
        c = pd.DataFrame(cost[name]).mean()
        fc = gpu if name in ("emb_lr", "gbm_esm") else "≈ 0"
        lines.append(f"| {NAMES[name]} | {FEATURES[name]} | {c['n_params']:,.0f} | "
                     f"{c['size_kb']:,.0f} | {c['train_seconds']:.1f} | "
                     f"{c['infer_ms_per_1000']:.2f} | {fc} | {auc:.3f} |")
    lines += ["", f"ESM-2 pass over all {esm_run['proteins']:,} proteins: "
              f"{esm_run['seconds']:.0f} s on {esm_run.get('gpu') or esm_run['device']} "
              f"({esm_run['windows']:,} windows of up to 1,022 residues). ESM-2 {args.model} itself "
              "is a pretrained network of about 35 M parameters (not trained here)."]
    ab = pd.DataFrame(abst_rows).groupby(["split", "deferred"])[
        ["error_rate_kept", "errors_removed", "auroc_kept"]].mean()
    lines += ["", "## Abstention on the test part (ESM-2 embedding + LR)", "",
              "| Split | Deferred | Error rate on kept | Errors removed | AUROC on kept |",
              "|---|---:|---:|---:|---:|"]
    for (split_type, f), r in ab.iterrows():
        lines.append(f"| {split_type} | {f:.0%} | {r['error_rate_kept']:.1%} | "
                     f"{r['errors_removed']:.1%} | {r['auroc_kept']:.3f} |")
    (REPORTS / "final_test_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Figures ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    show = ["gene_prior", "logreg", "gbm", "gbm_position", "zero_shot", "emb_lr", "gbm_esm"]
    for ax, metric, title in zip(axes, ["auroc", "auprc"], ["AUROC", "AUPRC"]):
        for i, split_type in enumerate(SPLIT_TYPES):
            m = [agg.loc[(n, split_type), (metric, "mean")] for n in show]
            e = [agg.loc[(n, split_type), (metric, "std")] for n in show]
            ax.bar(np.arange(len(show)) + (i - 0.5) * 0.38, m, 0.38, yerr=e, capsize=3,
                   label=f"{split_type.capitalize()} split")
        ax.set_xticks(range(len(show)), [NAMES[n].replace(" (", "\n(") for n in show],
                      rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"Test {title} (mapped variants, 5 seeds)")
    axes[0].axhline(0.5, color="grey", ls="--", lw=1)
    axes[1].axhline(y[splits["gene_s0"].to_numpy() == "test"].mean(), color="grey", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_test_metrics.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for name in ["logreg", "gbm_position", "emb_lr", "gbm_esm"]:
        frac, mp = calibration_curve(yt, preds[name], n_bins=10, strategy="quantile")
        ax.plot(mp, frac, "o-", ms=4, label=NAMES[name])
    ax.set_xlabel("Predicted probability of pathogenic")
    ax.set_ylabel("Observed share pathogenic")
    ax.set_title("Calibration on test, gene split (seed 0)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_test_calibration.png", dpi=150)

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
