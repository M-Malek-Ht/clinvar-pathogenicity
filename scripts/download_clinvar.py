"""Get the ClinVar variant_summary snapshot used by this project, and verify it.

Run from the repo root:  python scripts/download_clinvar.py

The project is pinned to ONE snapshot, recorded in reports/data_snapshot.json (release date,
size, MD5). That record is committed and is never overwritten once it exists.

What the script does:
  1. data/raw/variant_summary.txt.gz exists  -> check it against the recorded MD5.
  2. It does not exist                        -> download the current file from NCBI, then:
       - MD5 matches the record   -> saved as data/raw/variant_summary.txt.gz, ready to use.
       - MD5 differs (NCBI has published a newer release since our snapshot)
                                  -> saved as data/raw/variant_summary_NEWER.txt.gz, which the
                                     pipeline does NOT use, and a warning explains the options.
  3. No record exists yet (first ever run)    -> download and create the record.

To reproduce our exact results without the original file, use the committed filtered dataset
data/processed/variants.parquet and start from build_features.py (see README).
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
NEWER = ROOT / "data" / "raw" / "variant_summary_NEWER.txt.gz"
MANIFEST = ROOT / "reports" / "data_snapshot.json"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_md5():
    return json.loads(MANIFEST.read_text())["md5"] if MANIFEST.exists() else None


def verify_existing():
    print(f"Found existing file: {RAW}")
    md5, expected = md5_of(RAW), recorded_md5()
    if expected is None:
        print(f"No {MANIFEST.name} to compare against. MD5 = {md5}")
    elif md5 == expected:
        print("OK: file matches the recorded snapshot (MD5 identical).")
    else:
        print("MISMATCH: this file is NOT the recorded snapshot.\n"
              f"  expected {expected}\n  found    {md5}\n"
              "Results built from it will differ from the project's results.")


def download(target):
    """Download the current NCBI file to `target`; return (md5, Last-Modified header)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        last_modified = r.headers.get("Last-Modified", "unknown")
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc="Downloading") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(target)
    return md5_of(target), last_modified


def ncbi_check(md5):
    """Compare with the checksum NCBI publishes next to the file (detects a broken download)."""
    try:
        ncbi_md5 = requests.get(URL + ".md5", timeout=30).text.split()[0]
        return "match" if ncbi_md5 == md5 else f"MISMATCH (NCBI says {ncbi_md5})"
    except Exception:
        return "NCBI checksum file not available"


def main():
    if RAW.exists():
        verify_existing()
        return

    expected = recorded_md5()
    if expected is None:  # first ever run: create the record
        md5, last_modified = download(RAW)
        snapshot = {"source_url": URL, "release_date_server": last_modified,
                    "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "size_bytes": RAW.stat().st_size, "md5": md5,
                    "md5_vs_ncbi": ncbi_check(md5)}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(snapshot, indent=2))
        print(json.dumps(snapshot, indent=2))
        print(f"\nSnapshot recorded in {MANIFEST}")
        return

    md5, last_modified = download(NEWER)
    print(f"Download integrity vs NCBI checksum: {ncbi_check(md5)}")
    if md5 == expected:
        NEWER.replace(RAW)
        print(f"OK: NCBI's current file is the project snapshot. Saved to {RAW}")
        return

    snapshot = json.loads(MANIFEST.read_text())
    print(f"""
WARNING: NCBI has published a newer release since the project snapshot.
  Project snapshot : {snapshot.get('release_date_server')}  MD5 {expected}
  Downloaded file  : {last_modified}  MD5 {md5}

The newer file was saved as {NEWER.name} and is NOT used by the pipeline,
and {MANIFEST.name} was left unchanged. Options:
  - Reproduce the project's results exactly: skip this step and build_dataset.py, and start
    from build_features.py using the committed data/processed/variants.parquet.
  - Use the original snapshot: get the file with the MD5 above and put it at {RAW}.
  - Deliberately use the newer data: rename {NEWER.name} to {RAW.name}. Every count and
    result will then differ slightly from the reported ones.""")


if __name__ == "__main__":
    main()
