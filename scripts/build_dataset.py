"""Filter the ClinVar snapshot into the project dataset (proposal section 3).

Run from the repo root:  python scripts/build_dataset.py
Outputs:
  data/processed/variants.parquet   one row per missense variant, with its label
  reports/filtering_counts.md       how many variants survive each filtering step

Before filtering, the ClinVar file is checked against the MD5 recorded in
reports/data_snapshot.json; the script stops if the record is missing or the file differs
(override deliberately with --allow-different-snapshot).

Leakage note: 'stars' and 'last_evaluated' are kept ONLY for filtering and for the
failure-analysis slices. They must never be used as model features (proposal section 4).
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "variant_summary.txt.gz"
OUT_DATA = ROOT / "data" / "processed" / "variants.parquet"
OUT_COUNTS = ROOT / "reports" / "filtering_counts.md"
MANIFEST = ROOT / "reports" / "data_snapshot.json"

COLUMNS = ["VariationID", "Type", "Name", "ClinicalSignificance", "ReviewStatus",
           "LastEvaluated", "Assembly", "Chromosome", "PositionVCF",
           "ReferenceAlleleVCF", "AlternateAlleleVCF"]

# The two classes (proposal section 3). Anything else (VUS, conflicting, risk factor...) is dropped.
PATHOGENIC = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}

# ClinVar's review status -> number of stars.
STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no classification for the individual variant": 0,
    "no assertion provided": 0,
}

# The 20 standard amino acids (three-letter codes used in ClinVar names).
AA3 = {"Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
       "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"}

# Example name: NM_000059.4(BRCA2):c.122A>G (p.Arg41Gly)
NAME_RE = (r"^(?P<transcript>[^(]+)\((?P<gene>[^)]+)\):c\.\S+ "
           r"\(p\.(?P<ref_aa>[A-Z][a-z]{2})(?P<protein_pos>\d+)(?P<alt_aa>[A-Z][a-z]{2})\)$")


def check_snapshot():
    """Refuse to run on a ClinVar file other than the recorded snapshot.

    Override deliberately (e.g. to rebuild with a newer release) with:
        python scripts/build_dataset.py --allow-different-snapshot
    """
    if not RAW.exists():
        sys.exit(f"ERROR: {RAW} not found. Run scripts/download_clinvar.py first, or use the "
                 "committed data/processed/variants.parquet (README, quick reproduction).")
    if not MANIFEST.exists():
        sys.exit(f"ERROR: snapshot record missing: {MANIFEST}\n"
                 "It is created by scripts/download_clinvar.py and committed to the repository; "
                 "without it the ClinVar file cannot be verified. Stopping.")
    print("Verifying the ClinVar file against the recorded snapshot...")
    h = hashlib.md5()
    with open(RAW, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    expected = json.loads(MANIFEST.read_text())["md5"]
    if h.hexdigest() == expected:
        print("OK: file matches the recorded snapshot.")
    elif "--allow-different-snapshot" in sys.argv:
        print("WARNING: file differs from the recorded snapshot; continuing as requested. "
              "Counts and results will differ from the reported ones.")
    else:
        sys.exit(f"ERROR: {RAW.name} is not the recorded snapshot "
                 f"(expected MD5 {expected}, found {h.hexdigest()}). Stopping.\n"
                 "Use --allow-different-snapshot to process it anyway.")


def main():
    check_snapshot()
    counts = []

    def log(step, d):
        counts.append((step, len(d)))
        print(f"{step:<68}{len(d):>11,}")

    print("Reading the ClinVar file (takes 1-2 minutes)...")
    df = pd.read_csv(RAW, sep="\t", usecols=COLUMNS, dtype=str, low_memory=False)
    log("All rows in the snapshot", df)

    df = df[df["Assembly"] == "GRCh38"].drop_duplicates("VariationID")
    log("1. One genome version (GRCh38), one row per variant", df)

    df = df[df["Type"] == "single nucleotide variant"]
    log("2. Single-nucleotide variants", df)

    df = df.join(df["Name"].str.extract(NAME_RE))
    missense = (df["ref_aa"].isin(AA3) & df["alt_aa"].isin(AA3)
                & (df["ref_aa"] != df["alt_aa"]))
    df = df[missense]
    log("3. Missense (one amino acid replaced by another)", df)

    start_codon = (df["ref_aa"] == "Met") & (df["protein_pos"] == "1")
    df = df[~start_codon]
    log("   Start-codon changes (p.Met1...) removed", df)

    print("\nMost common labels among these missense variants:")
    print(df["ClinicalSignificance"].value_counts().head(15).to_string(), "\n")

    df = df[df["ClinicalSignificance"].isin(PATHOGENIC | BENIGN)]
    log("4. Label is Pathogenic/Likely pathogenic or Benign/Likely benign", df)

    df["stars"] = df["ReviewStatus"].map(STARS)
    unknown = df.loc[df["stars"].isna(), "ReviewStatus"].unique()
    if len(unknown):
        print("WARNING - review statuses not recognised (treated as 0 stars):", list(unknown))
    df = df[df["stars"].fillna(0) >= 1]
    log("5. At least one review star", df)

    # The same protein change can come from different DNA changes. If its copies disagree
    # on the label, drop all of them; if they agree, keep one row. Both are counted.
    df["label"] = df["ClinicalSignificance"].isin(PATHOGENIC).astype(int)
    key = ["gene", "protein_pos", "ref_aa", "alt_aa"]
    agree = df.groupby(key)["label"].transform("nunique") == 1
    n_conflict_rows = int((~agree).sum())
    n_conflict_changes = int(df.loc[~agree, key].drop_duplicates().shape[0])
    n_before_dedup = int(agree.sum())
    df = df[agree].drop_duplicates(key)
    n_duplicate_rows = n_before_dedup - len(df)
    log("6. One row per protein change (duplicates and conflicts removed)", df)
    step6 = [
        f"Rows removed because their protein change had contradictory labels: "
        f"{n_conflict_rows:,} (from {n_conflict_changes:,} protein change(s))",
        f"Duplicate rows of a protein change whose copies agree (one row kept): "
        f"{n_duplicate_rows:,}",
    ]
    print("   " + "\n   ".join(step6))

    out = pd.DataFrame({
        "variation_id": df["VariationID"],
        "gene": df["gene"],
        "transcript": df["transcript"],
        "ref_aa": df["ref_aa"],
        "protein_pos": df["protein_pos"].astype(int),
        "alt_aa": df["alt_aa"],
        "label": df["label"],
        "clinical_significance": df["ClinicalSignificance"],
        "review_status": df["ReviewStatus"],
        "stars": df["stars"].astype(int),
        "last_evaluated": df["LastEvaluated"],
        "chromosome": df["Chromosome"],
        "position_vcf": df["PositionVCF"],
        "ref_allele": df["ReferenceAlleleVCF"],
        "alt_allele": df["AlternateAlleleVCF"],
    }).reset_index(drop=True)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DATA, index=False)

    n_pos, n_neg = int(out["label"].sum()), int((out["label"] == 0).sum())
    genes = out.groupby("gene")["label"].agg(["min", "max"])
    both = int(((genes["min"] == 0) & (genes["max"] == 1)).sum())
    summary = [
        f"Pathogenic (label 1): {n_pos:,} ({n_pos / len(out):.1%})",
        f"Benign (label 0): {n_neg:,} ({n_neg / len(out):.1%})",
        f"Genes: {len(genes):,} (with both classes: {both:,})",
    ]

    md = ["# Dataset filtering counts", "",
          "Source: reports/data_snapshot.json", "",
          "| Step | Variants remaining |", "|---|---:|"]
    md += [f"| {s} | {n:,} |" for s, n in counts]
    md += ["", "Step 6 in detail:", ""] + [f"- {s}" for s in step6]
    md += ["", "## Final dataset", ""] + [f"- {s}" for s in summary]
    OUT_COUNTS.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n" + "\n".join(summary))
    print(f"\nSaved {OUT_DATA}\nSaved {OUT_COUNTS}")


if __name__ == "__main__":
    main()
