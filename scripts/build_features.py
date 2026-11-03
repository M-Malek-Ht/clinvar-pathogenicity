"""Build the Level-1 baseline features: properties of the amino-acid swap (proposal section 4).

Run from the repo root:  python scripts/build_features.py
Input:   data/processed/variants.parquet   (from build_dataset.py)
Outputs: data/processed/features_swap.parquet   variation_id + gene + label + features
         reports/feature_summary.md             what each feature is, and its average per class

Only the swap itself is described here: which amino acid was lost and which one replaced it.
Nothing about the gene, the position, the review status or the date is used, so there is no
leakage from the curation process (proposal section 4). Protein length, relative position and
conservation are added after the initial baseline.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DATA = ROOT / "data" / "processed" / "variants.parquet"
OUT_DATA = ROOT / "data" / "processed" / "features_swap.parquet"
OUT_SUMMARY = ROOT / "reports" / "feature_summary.md"

THREE_TO_ONE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
                "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
                "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
                "Tyr": "Y", "Val": "V"}

# BLOSUM62 (Henikoff & Henikoff, 1992), copied from Biopython's official table.
# How often one amino acid replaces another in related proteins: high = tolerated swap.
BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62_ROWS = {
    "A": [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    "R": [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    "N": [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    "D": [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    "C": [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "Q": [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    "E": [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    "G": [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    "H": [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    "K": [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    "M": [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    "S": [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    "T": [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    "V": [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
}
BLOSUM62 = {(a, b): BLOSUM62_ROWS[a][j] for a in BLOSUM_ORDER for j, b in enumerate(BLOSUM_ORDER)}

# Size: side-chain volume in cubic angstroms (Zamyatnin, 1972).
VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
          "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
          "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}

# Hydrophobicity: Kyte-Doolittle scale (Kyte & Doolittle, 1982). Positive = water-avoiding.
HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
         "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
         "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

# Charge at physiological pH (histidine is treated as neutral).
CHARGE = {aa: 0 for aa in BLOSUM_ORDER}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1})

FEATURE_DESCRIPTIONS = {
    "blosum62": "BLOSUM62 score of the swap (low = rarely tolerated in evolution)",
    "ref_volume": "Side-chain volume of the original amino acid",
    "alt_volume": "Side-chain volume of the new amino acid",
    "delta_volume": "Change in volume (new - original)",
    "abs_delta_volume": "Size of the volume change, ignoring direction",
    "ref_hydro": "Hydrophobicity of the original amino acid",
    "alt_hydro": "Hydrophobicity of the new amino acid",
    "delta_hydro": "Change in hydrophobicity (new - original)",
    "abs_delta_hydro": "Size of the hydrophobicity change, ignoring direction",
    "ref_charge": "Charge of the original amino acid",
    "alt_charge": "Charge of the new amino acid",
    "delta_charge": "Change in charge (new - original)",
    "charge_changed": "1 if the charge changes at all",
    "involves_cys": "1 if cysteine is lost or gained (disulfide bonds)",
    "involves_pro": "1 if proline is lost or gained (breaks helices)",
    "involves_gly": "1 if glycine is lost or gained (backbone flexibility)",
}


def build(variants: pd.DataFrame) -> pd.DataFrame:
    ref = variants["ref_aa"].map(THREE_TO_ONE)
    alt = variants["alt_aa"].map(THREE_TO_ONE)
    if ref.isna().any() or alt.isna().any():
        raise ValueError("Unknown amino-acid code found; check build_dataset.py output.")

    f = pd.DataFrame(index=variants.index)
    f["blosum62"] = [BLOSUM62[(r, a)] for r, a in zip(ref, alt)]
    for name, table in [("volume", VOLUME), ("hydro", HYDRO), ("charge", CHARGE)]:
        f[f"ref_{name}"] = ref.map(table)
        f[f"alt_{name}"] = alt.map(table)
        f[f"delta_{name}"] = f[f"alt_{name}"] - f[f"ref_{name}"]
    f["abs_delta_volume"] = f["delta_volume"].abs()
    f["abs_delta_hydro"] = f["delta_hydro"].abs()
    f["charge_changed"] = (f["delta_charge"] != 0).astype(int)
    for aa, name in [("C", "cys"), ("P", "pro"), ("G", "gly")]:
        f[f"involves_{name}"] = ((ref == aa) | (alt == aa)).astype(int)
    return f[list(FEATURE_DESCRIPTIONS)]


def main():
    variants = pd.read_parquet(IN_DATA)
    feats = build(variants)
    out = pd.concat([variants[["variation_id", "gene", "label"]], feats], axis=1)
    out.to_parquet(OUT_DATA, index=False)

    # Sanity check: average of each feature in each class.
    means = out.groupby("label")[list(FEATURE_DESCRIPTIONS)].mean().T
    means.columns = ["Benign (0)", "Pathogenic (1)"]

    md = ["# Swap features (initial baseline)", "",
          f"{len(out):,} variants, {feats.shape[1]} features, no missing values: "
          f"{not feats.isna().any().any()}", "",
          "| Feature | Meaning | Mean, benign | Mean, pathogenic |", "|---|---|---:|---:|"]
    for col, desc in FEATURE_DESCRIPTIONS.items():
        md.append(f"| `{col}` | {desc} | {means.loc[col, 'Benign (0)']:.2f} | "
                  f"{means.loc[col, 'Pathogenic (1)']:.2f} |")
    OUT_SUMMARY.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"{len(out):,} variants, {feats.shape[1]} features")
    print(f"Missing values: {int(feats.isna().sum().sum())}\n")
    print("Average of each feature per class:")
    print(means.round(2).to_string())
    print(f"\nSaved {OUT_DATA}\nSaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
