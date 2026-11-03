"""Download one fixed snapshot of ClinVar's variant_summary file and record it.

Run from the repo root:  python scripts/download_clinvar.py

- First run: downloads the file to data/raw/, checks it against NCBI's published
  MD5, and writes reports/data_snapshot.json (release date, size, MD5).
  Commit that JSON: it defines the exact dataset the project uses.
- Later runs / teammates: if data/raw/variant_summary.txt.gz already exists, nothing is
  downloaded; the file is checked against the MD5 in data_snapshot.json instead.
  Teammates should copy the SAME file from the team (not re-download), because
  NCBI replaces it every month.
"""
import datetime
import hashlib
import json
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "variant_summary.txt.gz"
MANIFEST = ROOT / "reports" / "data_snapshot.json"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_existing():
    print(f"Found existing file: {RAW}")
    md5 = md5_of(RAW)
    if not MANIFEST.exists():
        print(f"No {MANIFEST.name} to compare against. MD5 = {md5}")
        return
    expected = json.loads(MANIFEST.read_text())["md5"]
    if md5 == expected:
        print("OK: file matches the recorded snapshot (MD5 identical).")
    else:
        print("MISMATCH: this file is NOT the recorded snapshot.\n"
              f"  expected {expected}\n  found    {md5}")


def download():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW.with_suffix(".part")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        last_modified = r.headers.get("Last-Modified", "unknown")
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc="Downloading") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(RAW)

    md5 = md5_of(RAW)
    try:  # NCBI publishes a checksum file next to the data file
        ncbi_md5 = requests.get(URL + ".md5", timeout=30).text.split()[0]
        md5_check = "match" if ncbi_md5 == md5 else f"MISMATCH (NCBI says {ncbi_md5})"
    except Exception:
        md5_check = "NCBI checksum file not available"

    snapshot = {
        "source_url": URL,
        "release_date_server": last_modified,
        "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "size_bytes": RAW.stat().st_size,
        "md5": md5,
        "md5_vs_ncbi": md5_check,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(snapshot, indent=2))
    print(json.dumps(snapshot, indent=2))
    print(f"\nSnapshot recorded in {MANIFEST}")


if __name__ == "__main__":
    verify_existing() if RAW.exists() else download()
