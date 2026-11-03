"""Show the columns of the ClinVar snapshot with one example value each.

Run from the repo root:  python scripts/inspect_clinvar.py
Also saves the listing to reports/clinvar_columns.txt
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "variant_summary.txt.gz"
OUT = ROOT / "reports" / "clinvar_columns.txt"

# Read only the first 2,000 rows: enough to see the structure, fast to load.
df = pd.read_csv(RAW, sep="\t", nrows=2000, low_memory=False)

# Pick a row that looks like a missense variant (protein change "p." in its name), if any.
mask = df.iloc[:, 2].astype(str).str.contains(r"\(p\.", regex=True)
example = df[mask].iloc[0] if mask.any() else df.iloc[0]

lines = [f"{len(df.columns)} columns\n", f"{'#':>3}  {'column':<35} example value"]
for i, col in enumerate(df.columns):
    lines.append(f"{i:>3}  {col:<35} {str(example[col])[:70]}")

text = "\n".join(lines)
print(text)
OUT.write_text(text + "\n", encoding="utf-8")
print(f"\nSaved to {OUT}")
