"""Level 2, part 1: run ESM-2 on every mapped protein and extract per-variant outputs.

Run from the repo root:
    python scripts/esm_scores.py --model esm2_t12_35M_UR50D      (smallest, start here)
    python scripts/esm_scores.py --model esm2_t33_650M_UR50D     (larger, if time allows)
Inputs:  data/processed/variants.parquet, data/processed/variants_mapped.parquet,
         data/processed/uniprot_human.tsv
Outputs: data/processed/esm_<model>.parquet   one row per MAPPED variant:
             variation_id, esm_llr, emb_0 ... emb_<d-1>
         reports/esm_<model>_run.json         runtime, hardware, windows, checks

What is computed (one forward pass per protein, on the wild-type sequence only):
  esm_llr  zero-shot score = log P(mutant aa) - log P(wild-type aa) at the variant's position
           (wild-type marginals, Meier et al. 2021). Lower = less tolerated = more likely
           pathogenic. Needs no training and never saw ClinVar labels.
  emb_*    ESM-2's last-layer embedding of the wild-type protein at the variant's position:
           what the model knows about that position in its protein context. A classifier
           is trained on these in the next script.
Proteins longer than 1,022 residues are cut into overlapping windows of 1,022 (step 511);
each variant is read from the window whose centre is closest to it.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "data" / "processed" / "variants.parquet"
MAPPED = ROOT / "data" / "processed" / "variants_mapped.parquet"
UNIPROT = ROOT / "data" / "processed" / "uniprot_human.tsv"

THREE_TO_ONE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
                "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
                "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
                "Tyr": "Y", "Val": "V"}
WINDOW, STEP = 1022, 511


def window_starts(length):
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length - WINDOW + 1, STEP))
    if starts[-1] != length - WINDOW:
        starts.append(length - WINDOW)
    return starts


def best_window(pos0, starts):
    """Start of the window whose centre is closest to 0-based position pos0."""
    ok = [s for s in starts if s <= pos0 < s + WINDOW]
    return min(ok, key=lambda s: abs(pos0 - (s + WINDOW / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--limit-proteins", type=int, default=0,
                    help="only the first N proteins (quick test); 0 = all")
    args = ap.parse_args()
    out_data = ROOT / "data" / "processed" / f"esm_{args.model}.parquet"
    out_run = ROOT / "reports" / f"esm_{args.model}_run.json"

    v = pd.read_parquet(VARIANTS, columns=["variation_id", "ref_aa", "protein_pos", "alt_aa"])
    m = pd.read_parquet(MAPPED, columns=["variation_id", "uniprot_accession", "mapped"])
    if not (v["variation_id"].to_numpy() == m["variation_id"].to_numpy()).all():
        raise RuntimeError("variants and variants_mapped are not in the same row order.")
    v = v.assign(acc=m["uniprot_accession"])[m["mapped"].to_numpy()]
    v["ref1"] = v["ref_aa"].map(THREE_TO_ONE)
    v["alt1"] = v["alt_aa"].map(THREE_TO_ONE)
    v["pos0"] = v["protein_pos"] - 1

    u = pd.read_csv(UNIPROT, sep="\t")
    u.columns = ["accession", "gene", "length", "sequence"]
    seqs = dict(zip(u["accession"], u["sequence"]))
    proteins = list(v["acc"].unique())
    if args.limit_proteins:
        proteins = proteins[:args.limit_proteins]
        v = v[v["acc"].isin(proteins)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"facebook/{args.model}"
    print(f"Loading {name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = EsmForMaskedLM.from_pretrained(name).to(device).eval()
    if device == "cuda":
        model = model.half()
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in THREE_TO_ONE.values()}
    dim = model.config.hidden_size

    llr = np.full(len(v), np.nan, dtype=np.float32)
    emb = np.full((len(v), dim), np.nan, dtype=np.float16)
    ref_ok = np.zeros(len(v), dtype=bool)
    row_of = {vid: i for i, vid in enumerate(v["variation_id"])}
    n_windows = 0
    t0 = time.perf_counter()
    groups = v.groupby("acc", sort=False)
    for k, acc in enumerate(proteins):
        seq = seqs[acc]
        d = groups.get_group(acc)
        starts = window_starts(len(seq))
        d = d.assign(start=[best_window(p, starts) for p in d["pos0"]])
        for start, dw in d.groupby("start"):
            piece = seq[start:start + WINDOW]
            enc = tok(piece, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            logp = torch.log_softmax(out.logits[0].float(), dim=-1).cpu().numpy()
            hid = out.hidden_states[-1][0].float().cpu().numpy()
            n_windows += 1
            tpos = (dw["pos0"] - start + 1).to_numpy()  # +1 for the <cls> token
            rows = [row_of[x] for x in dw["variation_id"]]
            refs = [aa_ids[a] for a in dw["ref1"]]
            alts = [aa_ids[a] for a in dw["alt1"]]
            llr[rows] = logp[tpos, alts] - logp[tpos, refs]
            emb[rows] = hid[tpos].astype(np.float16)
            ref_ok[rows] = [piece[p - 1] == r for p, r in zip(tpos, dw["ref1"])]
        if (k + 1) % 500 == 0 or k + 1 == len(proteins):
            rate = (k + 1) / (time.perf_counter() - t0)
            print(f"  {k + 1:,}/{len(proteins):,} proteins, {n_windows:,} windows, "
                  f"{rate:.1f} proteins/s, about {(len(proteins) - k - 1) / rate / 60:.1f} min left")
    seconds = time.perf_counter() - t0

    res = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(dim)])
    res.insert(0, "esm_llr", llr)
    res.insert(0, "variation_id", v["variation_id"].to_numpy())
    res.to_parquet(out_data, index=False)

    run = {"model": name, "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "torch": torch.__version__, "platform": platform.platform(),
           "proteins": len(proteins), "windows": n_windows, "variants": int(len(v)),
           "embedding_dim": dim, "seconds": round(seconds, 1),
           "ms_per_1000_variants": round(seconds / len(v) * 1e6, 1),
           "wild_type_residue_matches": int(ref_ok.sum()),
           "missing_scores": int(np.isnan(llr).sum())}
    out_run.write_text(json.dumps(run, indent=2))
    print(json.dumps(run, indent=2))
    print(f"\nSaved {out_data}\nSaved {out_run}")


if __name__ == "__main__":
    main()
