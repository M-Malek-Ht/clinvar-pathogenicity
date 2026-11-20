"""Download the reviewed (Swiss-Prot) human proteome from UniProt, once, and record it.

Run from the repo root:  python scripts/fetch_uniprot.py
Outputs: data/processed/uniprot_human.tsv   one row per protein: accession, gene, length, sequence
         reports/uniprot_snapshot.json      release, download date, row count, MD5

UniProt is updated every 8 weeks, so the release number is recorded, like the ClinVar snapshot.
If the TSV already exists, the script only checks it and does not download again.
"""
import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "uniprot_human.tsv"
RECORD = ROOT / "reports" / "uniprot_snapshot.json"
URL = ("https://rest.uniprot.org/uniprotkb/stream"
       "?query=(reviewed:true)%20AND%20(organism_id:9606)"
       "&format=tsv&fields=accession,gene_primary,length,sequence")


def main():
    if not OUT.exists():
        print("Downloading reviewed human proteins from UniProt (about 15 MB)...")
        r = requests.get(URL, timeout=300)
        r.raise_for_status()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(r.content)
        record = {"source_url": URL,
                  "uniprot_release": r.headers.get("X-UniProt-Release", "unknown"),
                  "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
                  "md5": hashlib.md5(r.content).hexdigest()}
    else:
        print(f"Found {OUT}; not downloading again.")
        record = json.loads(RECORD.read_text()) if RECORD.exists() else {}

    df = pd.read_csv(OUT, sep="\t")
    df.columns = ["accession", "gene", "length", "sequence"]
    record["rows"] = len(df)
    record["rows_with_gene_name"] = int(df["gene"].notna().sum())
    record["genes_with_more_than_one_entry"] = int(df["gene"].dropna().duplicated().sum())
    RECORD.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"\nSaved {OUT}\nSaved {RECORD}")


if __name__ == "__main__":
    main()
