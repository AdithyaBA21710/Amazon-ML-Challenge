"""
Build a REALISTIC training set.

The old dev sample kept only 7% of the businesses, so most lookalikes were
missing and the model learned that the right owner always stands out. Here
blocking runs against the FULL pool of training businesses, exactly as on the
test set, so the model learns from real crowding.

Training entities are split into two fixed halves by a hash of their id:
  half A (0): the model learns from a sample of these records
  half B (1): never trained on; tune.py uses it for an honest score
Each S2/S3 record belongs to its owner's half (records with no owner are
hashed on their own id).

Usage:
    python src/build_dense.py                    # ~20% of half-A records
    python src/build_dense.py --query-frac 0.3
Output (in work/train/):
    blk_<country>.parquet     blocking candidates (reused by predict.py)
    dense_<country>.parquet   features + is_match for the sampled records
"""
import argparse
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR
from predict import (PROJECT_ROOT, _W, country_frames, features_for, get_candidates,
                     load_split, query_batches, run_batches, safe_name, setup_worker_state,
                     split_idf)


def half_of(ids):
    """Stable 0/1 split of entity ids."""
    return np.fromiter((zlib.crc32(str(i).encode()) & 1 for i in ids),
                       dtype=np.int8, count=len(ids))


def owner_series(data_dir):
    gt = pd.read_csv(Path(data_dir) / "train" / "train_ground_truth.tsv",
                     sep="\t", dtype=str, keep_default_na=False)
    ex = gt.assign(q=gt["matched_entity_ids"].str.split(",")).explode("q")
    ex = ex[ex["q"].notna() & (ex["q"] != "")]
    return pd.Series(ex["source1_entity_id"].to_numpy(), index=ex["q"].to_numpy())


def dense_batch(bounds):
    df = features_for(bounds)
    df["is_match"] = (_W["owner_idx"][df["q"].to_numpy()] == df["s"].to_numpy()).astype(np.int8)
    return df.drop(columns=["query_id", "s1_id"])


def main():
    ap = argparse.ArgumentParser(description="Build full-density training features")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--query-frac", type=float, default=0.2,
                    help="share of half-A S2/S3 records to build features for")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk-rows", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reblock", action="store_true", help="redo blocking instead of reusing it")
    ap.add_argument("--overwrite", action="store_true", help="rebuild finished countries")
    args = ap.parse_args()

    d = Path(args.work_dir) / "train"
    s1, queries = load_split(d)
    idf, default_idf = split_idf(s1)
    owner = owner_series(args.data_dir)
    print(f"train: {len(s1):,} S1 records, {len(queries):,} S2/S3 records")

    t0 = time.time()
    for country in sorted(s1["country"].unique()):
        out_path = d / f"dense_{safe_name(country)}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"\n[{country}] already built, skipping")
            continue
        s1c, qc = country_frames(s1, queries, country)
        print(f"\n[{country}] {len(s1c):,} S1 records, {len(qc):,} S2/S3 records", flush=True)
        m = get_candidates(d, country, s1c, qc, args)
        if m is None:
            continue

        # which records belong to half A, and who owns them
        s1_half = half_of(s1c["entity_id"])
        s1_pos = pd.Series(np.arange(len(s1c)), index=s1c["entity_id"].to_numpy())
        owner_idx = (qc["entity_id"].map(owner).map(s1_pos)
                     .fillna(-1).astype(np.int64).to_numpy())
        q_half = np.where(owner_idx >= 0, s1_half[np.clip(owner_idx, 0, None)],
                          half_of(qc["entity_id"]))
        rng = np.random.default_rng(args.seed)
        chosen = (q_half == 0) & (rng.random(len(qc)) < args.query_frac)
        rows = m[chosen[m["q"].to_numpy()]].reset_index(drop=True)

        owned = owner_idx[chosen] >= 0
        found = rows["s"].to_numpy() == owner_idx[rows["q"].to_numpy()]
        print(f"   {chosen.sum():,} half-A records sampled ({owned.mean():.1%} have an owner), "
              f"{len(rows):,} pairs; owner among candidates for "
              f"{found.sum() / max(owned.sum(), 1):.1%} of owned records", flush=True)

        setup_worker_state(m, rows, s1c, qc, idf, default_idf,
                           extra={"owner_idx": owner_idx})
        batches = query_batches(rows["q"].to_numpy(), args.batch)
        print(f"   features on {len(batches)} batches...", flush=True)
        parts = run_batches(dense_batch, batches, args.workers, None, t0)
        _W.clear()
        df = pd.concat(parts, ignore_index=True)
        df["country"] = country
        df.to_parquet(out_path, index=False)
        print(f"   saved {len(df):,} pairs ({df['is_match'].mean():.2%} positive) to {out_path.name}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        del m, rows, df, parts

    print("\nNext: python src/train_dense.py")


if __name__ == "__main__":
    main()