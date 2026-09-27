"""
Data loading, text normalization, and Phase 1 exploration
for the Business Entity Resolution challenge.

Run directly to print a data summary:
    python src/data.py
    python src/data.py --data-dir "C:/path/to/student_resource/dataset"
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(r"C:\Users\adith_a9r1d5f\Downloads\Amazon_Dataset\student_resource\dataset")

# Legal suffixes stripped from names. Includes French forms even though
# France is absent from training data.
LEGAL = {
    "pvt", "private", "ltd", "limited", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llc", "llp", "plc",
    "sa", "sas", "sarl", "eurl",
}


# ---------------------------------------------------------------- loading
def load_tsv(path):
    """Read a TSV with every column as a string and empty cells kept as ''."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_split(data_dir, split):
    """Load source1/2/3 (and ground truth, if present) for 'train' or 'test'."""
    d = Path(data_dir) / split
    s1 = load_tsv(d / f"{split}_source1.tsv")
    s2 = load_tsv(d / f"{split}_source2.tsv")
    s3 = load_tsv(d / f"{split}_source3.tsv")

    gt = None
    gt_path = d / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        gt = load_tsv(gt_path)
        gt["matches"] = gt["matched_entity_ids"].apply(
            lambda x: set(i.strip() for i in x.split(",") if i.strip()) if x else set()
        )
    return s1, s2, s3, gt


# ----------------------------------------------------------- normalization
def norm(s):
    """Lowercase, strip accents, '&' -> 'and', drop punctuation, squeeze spaces."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_core(s):
    """Normalized name with legal suffixes removed."""
    return " ".join(t for t in norm(s).split() if t not in LEGAL)


# ------------------------------------------------------ Phase 1 exploration
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def explore(s1, s2, s3, gt):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    section("1. Dataset sizes and countries")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        print(f"{name}: {len(df):,} rows | columns: {list(df.columns)}")
        print("   countries:", dict(Counter(df["country"])))

    section("2. Missing / empty fields")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        empties = {c: int((df[c].str.strip() == "").sum()) for c in df.columns}
        print(f"{name}: {empties}")

    section("3. Sample records (Source 1)")
    print(s1.sample(min(5, len(s1)), random_state=0).to_string(index=False))

    if gt is None:
        print("\nNo ground truth found; skipping match statistics.")
        return

    section("4. Ground truth: singletons and match counts")
    n_matches = gt["matches"].apply(len)
    print(f"Source 1 entities in ground truth: {len(gt):,}")
    print(f"Singleton rate (no matches): {(n_matches == 0).mean():.1%}")
    print("   -> this is also the score of predicting nothing for everyone")
    print("Match-count distribution:", dict(sorted(Counter(n_matches).items())))

    all_ids = [i for m in gt["matches"] for i in m]
    print(f"Matches pointing to Source 2: {sum(i.startswith('S2-') for i in all_ids):,}")
    print(f"Matches pointing to Source 3: {sum(i.startswith('S3-') for i in all_ids):,}")

    section("5. Is each S2/S3 record owned by at most one S1 entity?")
    repeats = {i: c for i, c in Counter(all_ids).items() if c > 1}
    print(f"S2/S3 ids appearing under more than one S1 entity: {len(repeats):,}")
    if repeats:
        print("   examples:", list(repeats.items())[:5])
    else:
        print("   -> one-owner rule looks safe to use in post-processing")

    section("6. Do matches ever cross countries?")
    country = dict(zip(s1["entity_id"], s1["country"]))
    country.update(zip(s2["entity_id"], s2["country"]))
    country.update(zip(s3["entity_id"], s3["country"]))
    missing = [i for i in all_ids if i not in country]
    cross = sum(
        country.get(e) != country.get(m)
        for e, ms in zip(gt["source1_entity_id"], gt["matches"])
        for m in ms if m in country
    )
    print(f"Cross-country matched pairs: {cross:,} of {len(all_ids) - len(missing):,}")
    if missing:
        print(f"WARNING: {len(missing)} ground-truth ids not found in source files")

    section("7. Example matched records, side by side")
    records = pd.concat([s1, s2, s3]).set_index("entity_id")
    examples = gt[n_matches > 0].sample(min(8, int((n_matches > 0).sum())), random_state=1)
    for e, ms in zip(examples["source1_entity_id"], examples["matches"]):
        r = records.loc[e]
        print(f"\n{e} [{r['country']}]  {r['business_name']}  |  {r['business_address']}")
        for m in sorted(ms):
            r = records.loc[m]
            print(f"   {m} [{r['country']}]  {r['business_name']}  |  {r['business_address']}")
            print(f"      name_core: '{name_core(records.loc[e]['business_name'])}'"
                  f"  vs  '{name_core(r['business_name'])}'")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 data exploration")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="folder containing train/ and test/")
    args = parser.parse_args()

    train_dir = Path(args.data_dir) / "train"
    if not train_dir.exists():
        sys.exit(f"Could not find {train_dir}\n"
                 f"Pass the right folder, e.g.:\n"
                 f'  python src/data.py --data-dir "C:/path/to/student_resource/dataset"')

    print(f"Loading data from {args.data_dir}")
    s1, s2, s3, gt = load_split(args.data_dir, "train")
    explore(s1, s2, s3, gt)

    t1, t2, t3, _ = load_split(args.data_dir, "test")
    section("8. Test set overview")
    for name, df in [("Source 1", t1), ("Source 2", t2), ("Source 3", t3)]:
        print(f"{name}: {len(df):,} rows | countries: {dict(Counter(df['country']))}")


if __name__ == "__main__":
    main()