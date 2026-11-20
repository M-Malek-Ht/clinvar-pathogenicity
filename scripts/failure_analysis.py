"""Failure analysis and mitigation (proposal section 7), on the gene split's validation part.

Run from the repo root (after run_esm_models.py):
    python scripts/failure_analysis.py --model esm2_t12_35M_UR50D
Inputs:  data/processed/{variants,features_swap,variants_mapped,esm_<model>}.parquet,
         data/splits/splits.parquet
Outputs: reports/failure_slices.csv          one row per (model, seed, slice)
         reports/failure_mitigation.csv      one row per (model, seed, method)
         reports/failure_summary.md          slices, recalibration and abstention tables
         data/processed/val_predictions_gene.parquet   per-variant validation predictions
                                                       (for the notebook figures)

Two models are compared on the MAPPED variants of the gene split's validation part, so every
validation gene is absent from training (5 seeds; the test part is not touched):
  trees    Level-1 gradient-boosted trees, swap + length + position (as run_gbm_position.py,
           but trained on mapped variants only, so both models see the same rows)
  emb_lr   ESM-2 embedding + esm_llr, logistic regression (as run_esm_models.py)
Each is re-tuned exactly as before (5-fold GroupKFold on train, AUPRC).

Slices (proposal section 7):
  few_variants     genes with <= 5 / 6-20 / 21-100 / > 100 variants in the whole dataset
  benign_disease   benign variants in disease-heavy genes (>= 50% pathogenic, >= 10 variants)
                   vs benign variants in other genes: false-positive rate at 0.5
  mild_pathogenic  pathogenic variants with a mild-looking swap (BLOSUM62 >= 1) vs harsh
                   swaps (BLOSUM62 <= 0): false-negative rate at 0.5
  recent           last evaluated in 2025-2026 vs before 2025 vs date missing
  review_stars     1 star vs 2+ stars (label confidence)
  confidence       predictions binned by distance from 0.5: error rate per bin
Mitigations:
  recalibration    validation genes are split in two halves; isotonic / Platt is fitted on
                   one half and applied to the other, then the halves swap (no gene is used to
                   fit its own calibration). Brier and ECE before and after.
  abstention       defer the 10 / 20 / 30% least confident variants to expert review; error
                   rate and AUROC on the variants kept, and the share of errors removed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
GBM_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 31],
            "min_samples_leaf": [20, 200]}
LR_GRID = {"logisticregression__C": [0.01, 0.1, 1]}
DEFER = [0.0, 0.1, 0.2, 0.3]
NAMES = {"trees": "Level 1 trees", "emb_lr": "ESM-2 embedding + LR"}


def ece(y, p, n_bins=10):
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def tune(estimator, grid, X, y, groups):
    search = GridSearchCV(estimator, grid, scoring="average_precision",
                          cv=GroupKFold(n_splits=5), n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def slice_metrics(y, p):
    """Metrics that make sense for any slice, including one-class slices."""
    r = {"n": int(len(y)), "pct_pathogenic": float(y.mean()) if len(y) else np.nan,
         "mean_p": float(p.mean()) if len(y) else np.nan,
         "error_rate": float(((p >= 0.5) != y).mean()) if len(y) else np.nan,
         "auroc": np.nan, "auprc": np.nan}
    if len(np.unique(y)) == 2:
        r.update(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))
    return r


def slices(d):
    """d has columns y, p, gene_n, gene_path_share, blosum62, year, stars. Yields (group, name, mask)."""
    n = d["gene_n"]
    for name, m in [("<= 5 variants", n <= 5), ("6-20", (n > 5) & (n <= 20)),
                    ("21-100", (n > 20) & (n <= 100)), ("> 100", n > 100)]:
        yield "few_variants", name, m
    heavy = (d["gene_path_share"] >= 0.5) & (n >= 10)
    yield "benign_disease", "benign, disease-heavy genes", (d["y"] == 0) & heavy
    yield "benign_disease", "benign, other genes", (d["y"] == 0) & ~heavy
    yield "mild_pathogenic", "pathogenic, mild swap (BLOSUM62 >= 1)", (d["y"] == 1) & (d["blosum62"] >= 1)
    yield "mild_pathogenic", "pathogenic, harsh swap (BLOSUM62 <= 0)", (d["y"] == 1) & (d["blosum62"] <= 0)
    yield "recent", "evaluated 2025-2026", d["year"] >= 2025
    yield "recent", "evaluated before 2025", d["year"] < 2025
    yield "recent", "date missing", d["year"].isna()
    yield "review_stars", "1 star", d["stars"] == 1
    yield "review_stars", "2+ stars", d["stars"] >= 2
    conf = (d["p"] - 0.5).abs()
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.51)]:
        yield "confidence", f"|p - 0.5| in [{lo:.1f}, {min(hi, 0.5):.1f}{']' if hi > 0.5 else ')'}", \
            (conf >= lo) & (conf < hi)


def recalibrate(y, p, genes, seed):
    """Cross-fitted recalibration over two disjoint halves of the validation genes."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(genes)
    half = np.isin(genes, rng.choice(uniq, size=len(uniq) // 2, replace=False))
    out = {"isotonic": np.empty_like(p), "platt": np.empty_like(p)}
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None]
    for fit, app in [(half, ~half), (~half, half)]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[fit], y[fit])
        out["isotonic"][app] = iso.predict(p[app])
        pl = LogisticRegression(C=1e6).fit(logit[fit], y[fit])
        out["platt"][app] = pl.predict_proba(logit[app])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    args = ap.parse_args()

    v = pd.read_parquet(PROC / "variants.parquet", columns=["variation_id", "stars", "last_evaluated"])
    feats = pd.read_parquet(PROC / "features_swap.parquet")
    mapped = pd.read_parquet(PROC / "variants_mapped.parquet")
    splits = pd.read_parquet(ROOT / "data" / "splits" / "splits.parquet")
    for name, d in [("variants", v), ("variants_mapped", mapped), ("splits", splits)]:
        if not (feats["variation_id"].to_numpy() == d["variation_id"].to_numpy()).all():
            raise RuntimeError(f"features_swap and {name} are not in the same row order.")
    esm = pd.read_parquet(PROC / f"esm_{args.model}.parquet")

    # Gene-level facts from the whole dataset (used only to define slices, never as features).
    gstats = feats.groupby("gene")["label"].agg(gene_n="size", gene_path_share="mean")

    keep = mapped["mapped"].to_numpy()
    base = pd.concat([feats, mapped[["protein_length", "rel_position"]],
                      v[["stars", "last_evaluated"]]], axis=1)[keep]
    base = base.merge(esm, on="variation_id", how="left", validate="one_to_one")
    if base["esm_llr"].isna().any():
        raise RuntimeError("Some mapped variants have no ESM-2 score: rerun esm_scores.py.")
    base = base.join(gstats, on="gene")
    base["year"] = pd.to_datetime(base["last_evaluated"], format="%b %d, %Y",
                                  errors="coerce").dt.year
    splits = splits[keep].reset_index(drop=True)
    y, genes = base["label"].to_numpy(), base["gene"].to_numpy()
    swap_cols = [c for c in feats.columns if c not in ("variation_id", "gene", "label")]
    emb_cols = [c for c in base.columns if c.startswith("emb_")]
    X = {"trees": base[swap_cols + ["protein_length", "rel_position"]].to_numpy(dtype=float),
         "emb_lr": base[emb_cols + ["esm_llr"]].to_numpy(dtype=np.float32)}
    print(f"{len(base):,} mapped variants; gene split, validation part, 5 seeds")

    slice_rows, mit_rows, preds = [], [], []
    for seed in SEEDS:
        part = splits[f"gene_s{seed}"].to_numpy()
        tr, va = part == "train", part == "val"
        models = {
            "trees": tune(HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                                         random_state=seed),
                          GBM_GRID, X["trees"][tr], y[tr], genes[tr]),
            "emb_lr": tune(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                           LR_GRID, X["emb_lr"][tr], y[tr], genes[tr])}
        d = base.loc[va, ["variation_id", "gene", "gene_n", "gene_path_share", "blosum62",
                          "year", "stars"]].reset_index(drop=True)
        d["y"] = y[va]
        for mname, model in models.items():
            p = model.predict_proba(X[mname][va])[:, 1]
            dm = d.assign(p=p)
            preds.append(dm.assign(model=mname, seed=seed)[
                ["variation_id", "gene", "model", "seed", "y", "p"]])

            for group, sname, m in slices(dm):
                m = m.to_numpy()
                slice_rows.append({"model": mname, "seed": seed, "group": group, "slice": sname,
                                   **slice_metrics(dm["y"].to_numpy()[m], p[m])})

            yv = dm["y"].to_numpy()
            cal = recalibrate(yv, p, dm["gene"].to_numpy(), seed)
            for method, q in [("none", p), ("isotonic", cal["isotonic"]),
                              ("platt", cal["platt"])]:
                mit_rows.append({"model": mname, "seed": seed, "kind": "recalibration",
                                 "method": method, "brier": brier_score_loss(yv, q),
                                 "ece": ece(yv, q), "auroc": roc_auc_score(yv, q)})

            wrong = (p >= 0.5) != yv
            order = np.argsort(np.abs(p - 0.5))  # least confident first
            for f in DEFER:
                kept = np.ones(len(p), dtype=bool)
                kept[order[:int(round(f * len(p)))]] = False
                mit_rows.append({"model": mname, "seed": seed, "kind": "abstention",
                                 "method": f"defer {f:.0%}", "deferred": f,
                                 "error_rate_kept": wrong[kept].mean(),
                                 "errors_removed": 1 - wrong[kept].sum() / wrong.sum(),
                                 "auroc": roc_auc_score(yv[kept], p[kept])})
        print(f"seed {seed}: done ({len(d):,} validation variants, "
              f"{d['gene'].nunique():,} genes)")

    sl, mit = pd.DataFrame(slice_rows), pd.DataFrame(mit_rows)
    sl.to_csv(REPORTS / "failure_slices.csv", index=False)
    mit.to_csv(REPORTS / "failure_mitigation.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(PROC / "val_predictions_gene.parquet",
                                                   index=False)

    # ---- Summary ----
    def fmt(x, pct=False):
        return "" if pd.isna(x) else (f"{x:.1%}" if pct else f"{x:.3f}")

    agg = sl.groupby(["group", "slice", "model"], sort=False)[
        ["n", "pct_pathogenic", "mean_p", "error_rate", "auroc"]].mean()
    lines = [f"# Failure analysis: gene split, mapped validation variants, 5 seeds (ESM-2 {args.model})",
             "", "Numbers are means over the 5 seeds; n is variants per seed. Error rate uses a "
             "0.5 threshold. AUROC is blank for one-class slices.", ""]
    titles = {"few_variants": "Genes with few known variants",
              "benign_disease": "Benign variants in disease-heavy genes (false-positive rate)",
              "mild_pathogenic": "Pathogenic variants with mild-looking swaps (false-negative rate)",
              "recent": "Recently classified variants",
              "review_stars": "Label confidence (ClinVar review stars)",
              "confidence": "Low-confidence predictions (bins are per model)"}
    for group, title in titles.items():
        lines += [f"## {title}", "",
                  "| Slice | n | % pathogenic | Trees: mean p | Trees: error | Trees: AUROC "
                  "| ESM LR: mean p | ESM LR: error | ESM LR: AUROC |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for sname in sl.loc[sl["group"] == group, "slice"].unique():
            t, e = agg.loc[(group, sname, "trees")], agg.loc[(group, sname, "emb_lr")]
            n_txt = f"{t['n']:,.0f}" if group != "confidence" else f"{t['n']:,.0f} / {e['n']:,.0f}"
            lines.append(f"| {sname} | {n_txt} | {fmt(t['pct_pathogenic'], True)} | "
                         f"{fmt(t['mean_p'])} | {fmt(t['error_rate'], True)} | {fmt(t['auroc'])} | "
                         f"{fmt(e['mean_p'])} | {fmt(e['error_rate'], True)} | {fmt(e['auroc'])} |")
        lines.append("")

    rc = mit[mit["kind"] == "recalibration"].groupby(["model", "method"], sort=False)[
        ["brier", "ece", "auroc"]].agg(["mean", "std"])
    lines += ["## Mitigation 1: recalibration on separate validation genes", "",
              "Isotonic / Platt fitted on half of the validation genes, applied to the other "
              "half, then swapped.", "",
              "| Model | Method | Brier | ECE | AUROC |", "|---|---|---|---|---|"]
    for (mname, method), r in rc.iterrows():
        lines.append(f"| {NAMES[mname]} | {method} | {r[('brier', 'mean')]:.3f} ± "
                     f"{r[('brier', 'std')]:.3f} | {r[('ece', 'mean')]:.3f} ± "
                     f"{r[('ece', 'std')]:.3f} | {r[('auroc', 'mean')]:.3f} |")
    ab = mit[mit["kind"] == "abstention"].groupby(["model", "method"], sort=False)[
        ["error_rate_kept", "errors_removed", "auroc"]].mean()
    lines += ["", "## Mitigation 2: abstain on the least confident variants", "",
              "| Model | Deferred to expert review | Error rate on kept | Errors removed "
              "| AUROC on kept |", "|---|---|---:|---:|---:|"]
    for (mname, method), r in ab.iterrows():
        lines.append(f"| {NAMES[mname]} | {method.replace('defer ', '')} | "
                     f"{r['error_rate_kept']:.1%} | {r['errors_removed']:.1%} | {r['auroc']:.3f} |")
    lines += ["", "- Unmapped variants (7.2%) have no ESM-2 score; in deployment they would "
              "fall back to the Level 1 trees or be deferred."]
    (REPORTS / "failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
