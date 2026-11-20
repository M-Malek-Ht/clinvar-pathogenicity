"""Map every variant to its UniProt protein and add protein length and relative position.

Run from the repo root:  python scripts/map_sequences.py
Inputs:  data/processed/variants.parquet, data/processed/uniprot_human.tsv (fetch_uniprot.py)
Outputs: data/processed/variants_mapped.parquet   same rows and order as variants.parquet
         reports/mapping_summary.md               how many variants map, and why the rest don't

A variant is "mapped" when its gene is in UniProt AND the UniProt sequence has ClinVar's
original amino acid at ClinVar's position. ClinVar positions come from a RefSeq transcript,
which is not always the same isoform as UniProt's canonical sequence; the residue check
catches those cases instead of silently attaching the wrong position.
Protein length and relative position are only filled in for mapped variants.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "data" / "processed" / "variants.parquet"
UNIPROT = ROOT / "data" / "processed" / "uniprot_human.tsv"
OUT = ROOT / "data" / "processed" / "variants_mapped.parquet"
SUMMARY = ROOT / "reports" / "mapping_summary.md"

THREE_TO_ONE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
                "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
                "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
                "Tyr": "Y", "Val": "V"}
ESM_MAX_LEN = 1022  # longest protein ESM-2 reads in one piece


def main():
    t0 = time.perf_counter()
    v = pd.read_parquet(VARIANTS, columns=["variation_id", "gene", "label",
                                           "ref_aa", "protein_pos", "alt_aa"])
    u = pd.read_csv(UNIPROT, sep="\t")
    u.columns = ["accession", "gene", "length", "sequence"]
    u = u.dropna(subset=["gene"])

    # A gene can have more than one UniProt entry: try them all, keep a matching one.
    m = v.reset_index().merge(u, on="gene", how="left")
    ref1 = m["ref_aa"].map(THREE_TO_ONE)
    in_range = m["sequence"].notna() & (m["protein_pos"] <= m["length"])
    residue = [s[p - 1] if ok else "" for s, p, ok in
               zip(m["sequence"].fillna(""), m["protein_pos"], in_range)]
    m["match"] = in_range & (pd.Series(residue, index=m.index) == ref1)
    m = (m.sort_values(["index", "match"], ascending=[True, False])
          .drop_duplicates("index").sort_values("index").reset_index(drop=True))
    if len(m) != len(v) or not (m["variation_id"].to_numpy() == v["variation_id"].to_numpy()).all():
        raise RuntimeError("Row order changed while mapping; nothing was saved.")

    status = np.select(
        [m["sequence"].isna(), m["protein_pos"] > m["length"], ~m["match"]],
        ["gene not in UniProt", "position beyond protein length", "residue mismatch"],
        "mapped")
    mapped = status == "mapped"
    out = pd.DataFrame({
        "variation_id": m["variation_id"],
        "uniprot_accession": m["accession"].where(mapped),
        "protein_length": m["length"].where(mapped),
        "rel_position": (m["protein_pos"] / m["length"]).where(mapped),
        "mapping_status": status,
        "mapped": mapped,
    })
    out.to_parquet(OUT, index=False)

    # ---- Summary ----
    lines = ["# Mapping variants to UniProt sequences", "",
             "| Status | Variants | Share | % pathogenic | Genes |", "|---|---:|---:|---:|---:|"]
    for s in ["mapped", "residue mismatch", "position beyond protein length",
              "gene not in UniProt"]:
        k = status == s
        if k.sum() == 0:
            continue
        lines.append(f"| {s} | {k.sum():,} | {k.mean():.1%} | {v['label'][k].mean():.1%} | "
                     f"{v['gene'][k].nunique():,} |")
    long_prot = mapped & (m["length"] > ESM_MAX_LEN).to_numpy()
    lines += ["",
              f"- Mapped proteins longer than {ESM_MAX_LEN} residues: "
              f"{m.loc[long_prot, 'accession'].nunique():,} proteins holding "
              f"{long_prot.sum():,} variants ({long_prot.sum() / mapped.sum():.1%} of mapped). "
              "ESM-2 will read these in a window centred on the variant.",
              f"- Genes with both classes among mapped variants: "
              f"{(v[mapped].groupby('gene')['label'].nunique() == 2).sum():,}",
              f"- Runtime: {time.perf_counter() - t0:.1f} s"]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {OUT}\nSaved {SUMMARY}")


if __name__ == "__main__":
    main()
