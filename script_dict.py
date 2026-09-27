"""
Learn a dictionary from Indian-script words to English words, using ONLY the
provided training data.

Many true matches pair an Indian-script name with its English original:
    'சன் டெக் பிரைவேட் லிமிடெட்'  <->  'Sun Tech Private Limited'
When both names have the same number of words, we line them up word by word
and count. Across thousands of pairs, each script word's most frequent
partner is its English equivalent ('லிமிடெட்' -> 'limited').

Dev-sample entities are excluded so dev evaluation stays honest.

Usage (run after preprocess.py has built work/train and work/dev):
    python src/script_dict.py
    python src/script_dict.py --data-dir "C:/path/to/dataset"

Output: work/script_dict.json
Then rerun preprocess.py with --overwrite so the dictionary is applied.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from normalize import INDIC_RE, split_script_tokens, to_ascii_lower

from config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_names(path, column="business_name"):
    """Rows whose name contains Indian-script characters."""
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=500_000, columns=["entity_id", column]):
        df = batch.to_pandas()
        has_script = df[column].map(lambda x: bool(INDIC_RE.search(x or "")))
        parts.append(df[has_script])
    return pd.concat(parts, ignore_index=True)


def read_names(path, keep_ids):
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=500_000, columns=["entity_id", "business_name"]):
        df = batch.to_pandas()
        parts.append(df[df["entity_id"].isin(keep_ids)])
    return pd.concat(parts, ignore_index=True)


def coverage(names, mapping):
    """Share of script words (and whole names) the dictionary can translate."""
    tok_total = tok_known = full = 0
    for name in names:
        toks = [t for t in split_script_tokens(name) if INDIC_RE.search(t)]
        known = sum(t in mapping for t in toks)
        tok_total += len(toks)
        tok_known += known
        full += int(known == len(toks))
    return tok_known / max(tok_total, 1), full / max(len(names), 1)


def main():
    ap = argparse.ArgumentParser(description="Learn Indian-script -> English word dictionary")
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="folder containing train/ and test/ (for the ground truth file)")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--min-count", type=int, default=3,
                    help="a word pair must be seen this many times to be trusted")
    args = ap.parse_args()
    work = Path(args.work_dir)

    # 1. S2/S3 training records whose names use an Indian script
    q = pd.concat([load_script_names(work / "train" / f"source{i}.parquet")
                   for i in (2, 3)], ignore_index=True)
    print(f"S2/S3 training names containing Indian script: {len(q):,}")

    # 2. Their owners, from the ground truth (dev entities excluded)
    gt = pd.read_csv(Path(args.data_dir) / "train" / "train_ground_truth.tsv",
                     sep="\t", dtype=str, keep_default_na=False)
    owner = (gt.assign(q=gt["matched_entity_ids"].str.split(","))
               .explode("q").query("q != ''").set_index("q")["source1_entity_id"])
    q["s1_id"] = q["entity_id"].map(owner)
    q = q.dropna(subset=["s1_id"])
    dev_gt = work / "dev" / "ground_truth.parquet"
    if dev_gt.exists():
        dev_ids = set(pd.read_parquet(dev_gt)["source1_entity_id"])
        q = q[~q["s1_id"].isin(dev_ids)]
    print(f"   ... with a known owner, outside the dev sample: {len(q):,}")

    s1 = read_names(work / "train" / "source1.parquet", set(q["s1_id"]))
    pairs = q.merge(s1.rename(columns={"entity_id": "s1_id", "business_name": "s1_name"}),
                    on="s1_id")

    # 3. Word-by-word alignment and counting
    counts = defaultdict(Counter)
    aligned = 0
    for native, english in zip(pairs["business_name"], pairs["s1_name"]):
        nt = split_script_tokens(native)
        et = split_script_tokens(to_ascii_lower(english))
        if len(nt) != len(et):
            continue
        aligned += 1
        for a, b in zip(nt, et):
            if INDIC_RE.search(a) and b.isascii():
                counts[a][b] += 1
    print(f"Name pairs with equal word counts (used for learning): {aligned:,}")

    # 4. Keep a word's most common partner if it is frequent and dominant
    mapping = {}
    for tok, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= args.min_count and n / sum(c.values()) >= 0.5:
            mapping[tok] = best
    print(f"Dictionary entries learned: {len(mapping):,}")
    print("Most frequent entries:")
    top = sorted(mapping, key=lambda t: -counts[t][mapping[t]])[:15]
    for t in top:
        print(f"   {t}  ->  {mapping[t]}   (seen {counts[t][mapping[t]]:,} times)")

    out = work / "script_dict.json"
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"\nSaved to {out}")

    # 5. How much of dev and test can it translate?
    print("\nCoverage (words translated / whole names fully translated):")
    for split in ("dev", "test"):
        paths = [work / split / f"source{i}.parquet" for i in (2, 3)]
        if not all(p.exists() for p in paths):
            continue
        names = pd.concat([load_script_names(p) for p in paths])["business_name"]
        w, f = coverage(names, mapping)
        print(f"   {split:5s} {len(names):>9,} script names   words {w:.1%}   whole names {f:.1%}")


if __name__ == "__main__":
    main()