"""
Preprocessing: normalize every source file, save as Parquet, and build a
smaller development sample from the training data.

Quick smoke test first (a few seconds, writes to work_smoke/):
    python src/preprocess.py --data-dir "C:/path/to/dataset" --limit 20000

Full run (writes to work/):
    python src/preprocess.py --data-dir "C:/path/to/dataset"

Output:
    work/train/source1.parquet, source2.parquet, source3.parquet
    work/test/source1.parquet,  source2.parquet, source3.parquet
    work/dev/source1.parquet, source2.parquet, source3.parquet, ground_truth.parquet
"""
import argparse
import os
import time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from normalize import OUTPUT_COLUMNS, SCRIPT_DICT, normalize_record

from config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_ROWS = 250_000
SOURCES = ["source1", "source2", "source3"]


def process_file(in_path, out_path, pool, limit=None):
    """Normalize one TSV in chunks (keeps memory low) and write Parquet."""
    tmp_path = out_path.with_suffix(".tmp")
    writer, schema, total, t0 = None, None, 0, time.time()
    reader = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False,
                         chunksize=CHUNK_ROWS, nrows=limit)
    for chunk in reader:
        chunk["country"] = chunk["country"].str.strip()
        rows = list(zip(chunk["business_name"], chunk["business_address"]))
        if pool is not None:
            results = pool.map(normalize_record, rows, chunksize=2000)
        else:
            results = [normalize_record(r) for r in rows]
        out = pd.concat(
            [chunk.reset_index(drop=True), pd.DataFrame(results, columns=OUTPUT_COLUMNS)],
            axis=1,
        )
        out["name_is_web"] = out["name_is_web"].astype("int8")
        out["addr_missing"] = out["addr_missing"].astype("int8")

        if writer is None:
            table = pa.Table.from_pandas(out, preserve_index=False)
            schema = table.schema
            writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
        else:
            table = pa.Table.from_pandas(out, schema=schema, preserve_index=False)
        writer.write_table(table)
        total += len(out)
        print(f"      {total:,} rows  ({time.time() - t0:.0f}s)", flush=True)

    if writer is not None:
        writer.close()
        os.replace(tmp_path, out_path)  # only a finished file gets the real name


def read_filtered(path, keep_ids):
    """Read only the rows whose entity_id is in keep_ids, batch by batch."""
    pf = pq.ParquetFile(path)
    parts = []
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        parts.append(df[df["entity_id"].isin(keep_ids)])
    return pd.concat(parts, ignore_index=True)


def explode_ids(series):
    ids = series.str.split(",").explode().str.strip()
    return ids[ids.notna() & (ids != "")]


def make_dev_sample(data_dir, work_dir, n, seed):
    """
    A miniature copy of the full problem:
      - n random Source 1 entities
      - all their true S2/S3 matches
      - the same fraction of unowned S2/S3 'distractor' records
    Caveat: a smaller pool has fewer look-alike businesses, so dev scores
    will be somewhat optimistic. Confirm big decisions on a larger sample.
    """
    dev_dir = work_dir / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(Path(data_dir) / "train" / "train_ground_truth.tsv",
                     sep="\t", dtype=str, keep_default_na=False)
    n = min(n, len(gt))
    frac = n / len(gt)
    dev_gt = gt.sample(n=n, random_state=seed).reset_index(drop=True)

    owned_all = explode_ids(gt["matched_entity_ids"])
    owned_dev = explode_ids(dev_gt["matched_entity_ids"])

    s1 = read_filtered(work_dir / "train" / "source1.parquet",
                       set(dev_gt["source1_entity_id"]))
    s1.to_parquet(dev_dir / "source1.parquet", index=False)
    print(f"   dev source1: {len(s1):,} rows")

    for src, prefix in [("source2", "S2-"), ("source3", "S3-")]:
        path = work_dir / "train" / f"{src}.parquet"
        ids = pd.read_parquet(path, columns=["entity_id"])["entity_id"]
        unowned = ids[~ids.isin(owned_all)]
        distractors = unowned.sample(frac=frac, random_state=seed)
        keep = set(owned_dev[owned_dev.str.startswith(prefix)]) | set(distractors)
        df = read_filtered(path, keep)
        df.to_parquet(dev_dir / f"{src}.parquet", index=False)
        print(f"   dev {src}: {len(df):,} rows "
              f"({len(distractors):,} distractors with no owner)")

    dev_gt.to_parquet(dev_dir / "ground_truth.parquet", index=False)
    singles = (dev_gt["matched_entity_ids"] == "").mean()
    print(f"   dev ground truth: {len(dev_gt):,} entities, singleton rate {singles:.1%}")


def show_examples(path, n=8):
    df = next(pq.ParquetFile(path).iter_batches(batch_size=n)).to_pandas()
    print(f"\n   Examples from {path.name}:")
    for _, r in df.iterrows():
        print(f"   {r['business_name']!r}")
        print(f"      main={r['name_main']!r} alt={r['name_alt']!r} "
              f"nospace={r['name_nospace']!r} web={r['name_is_web']}")
        print(f"   {r['business_address']!r}")
        print(f"      addr={r['addr_norm']!r} nums={r['addr_nums']!r}")


def main():
    ap = argparse.ArgumentParser(description="Normalize data and build dev sample")
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="folder containing train/ and test/")
    ap.add_argument("--work-dir", default=None,
                    help="where Parquet files go (default: work/, or work_smoke/ with --limit)")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--dev-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None,
                    help="only read this many rows per file (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo files that already exist")
    args = ap.parse_args()

    default_work = "work_smoke" if args.limit else "work"
    work_dir = Path(args.work_dir) if args.work_dir else PROJECT_ROOT / default_work
    print(f"Data: {args.data_dir}\nOutput: {work_dir}\nWorkers: {args.workers}")
    print(f"Script dictionary: {len(SCRIPT_DICT):,} entries"
          + ("" if SCRIPT_DICT else " (none found; run src/script_dict.py to create it)"))

    pool_ctx = Pool(args.workers) if args.workers > 1 else nullcontext(None)
    with pool_ctx as pool:
        for split in args.splits:
            (work_dir / split).mkdir(parents=True, exist_ok=True)
            for src in SOURCES:
                in_path = Path(args.data_dir) / split / f"{split}_{src}.tsv"
                out_path = work_dir / split / f"{src}.parquet"
                if out_path.exists() and not args.overwrite:
                    print(f"\n[{split}/{src}] already done, skipping")
                    continue
                print(f"\n[{split}/{src}] normalizing {in_path}")
                process_file(in_path, out_path, pool, args.limit)

    if args.limit:
        print("\nSmoke test: skipping dev sample (needs the full training files).")
        show_examples(work_dir / args.splits[0] / "source3.parquet")
    elif "train" in args.splits:
        print("\n[dev] building development sample")
        make_dev_sample(args.data_dir, work_dir, args.dev_size, args.seed)
        show_examples(work_dir / "dev" / "source3.parquet")
    print("\nDone.")


if __name__ == "__main__":
    main()