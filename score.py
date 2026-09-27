"""
Score a matching_results.tsv against the training ground truth with the
competition metric (macro F0.5 over Source 1 entities, singletons included).

Use it after running the full pipeline on the TRAINING set:
    python src/predict.py --split train
    python src/score.py --pred output_train/matching_results.tsv

Dev-sample entities are excluded by default, because the model was trained on
them; every other entity is scored honestly. Only entities present in the
prediction file are scored (so --countries runs work too).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_lists(path, col):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(df["source1_entity_id"],
                    df[col].map(lambda x: frozenset(i for i in x.split(",") if i))))


def f05(pred, true):
    if not true and not pred:
        return 1.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def main():
    ap = argparse.ArgumentParser(description="Macro F0.5 against training ground truth")
    ap.add_argument("--pred", default=str(PROJECT_ROOT / "output_train" / "matching_results.tsv"))
    ap.add_argument("--candidates", default=None,
                    help="candidate_pairs.tsv (default: next to --pred) for the ceiling")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--include-dev", action="store_true",
                    help="also score the dev entities the model was trained on")
    args = ap.parse_args()

    pred = read_lists(args.pred, "matched_entity_ids")
    gt = read_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv", "matched_entity_ids")
    cand_path = Path(args.candidates) if args.candidates else Path(args.pred).with_name("candidate_pairs.tsv")
    cand = read_lists(cand_path, "candidate_entity_ids") if cand_path.exists() else None

    ids = [e for e in pred if e in gt]
    if not args.include_dev:
        dev = Path(args.work_dir) / "dev" / "ground_truth.parquet"
        if dev.exists():
            dev_ids = set(pd.read_parquet(dev, columns=["source1_entity_id"])["source1_entity_id"])
            ids = [e for e in ids if e not in dev_ids]
    s1 = pd.read_parquet(Path(args.work_dir) / "train" / "source1.parquet",
                         columns=["entity_id", "country"]).set_index("entity_id")["country"]

    rows = []
    for e in ids:
        p, t = pred[e], gt[e]
        c = cand.get(e, frozenset()) if cand is not None else None
        rows.append((e, s1.get(e, "?"), f05(p, t), len(p & t), len(p), len(t),
                     (1.0 if not t else f05(c & t, t)) if c is not None else np.nan))
    df = pd.DataFrame(rows, columns=["id", "country", "f", "tp", "npred", "ntrue", "ceiling"])
    if df.empty:
        raise SystemExit("No entities to score (all were dev entities, or the files don't overlap).")

    print("=" * 70)
    print(f"FULL-DENSITY MACRO F0.5 = {df['f'].mean():.4f}   ({len(df):,} S1 entities scored)")
    print("=" * 70)
    print(f"Pair precision {df['tp'].sum() / max(df['npred'].sum(), 1):.2%}   "
          f"pair recall {df['tp'].sum() / max(df['ntrue'].sum(), 1):.2%}")
    if cand is not None:
        print(f"Blocking ceiling {df['ceiling'].mean():.4f}")
    print("\nBy country:")
    for c, g in df.groupby("country"):
        print(f"   {c:8s} {g['f'].mean():.4f}   ({len(g):,} entities)")

    single = df["ntrue"] == 0
    print(f"\nSingletons: {single.mean():.1%} of entities, "
          f"{(df.loc[single, 'npred'] == 0).mean():.1%} correctly left empty")
    nonsingle = df[~single]
    print("Entities with true matches:")
    print(f"   got no prediction at all:       {(nonsingle['npred'] == 0).mean():.1%}")
    print(f"   contain at least one wrong ID:  {(nonsingle['npred'] > nonsingle['tp']).mean():.1%}")
    print(f"   exactly right:                  {(nonsingle['f'] == 1).mean():.1%}")

    # Where is the score lost? (points lost per entity, averaged over all entities)
    lost = 1 - df["f"]
    print("\nScore lost, by cause (out of 1.0):")
    print(f"   singletons given a match:       {lost[single].sum() / len(df):.4f}")
    wrong = (~single) & (df["npred"] > df["tp"])
    print(f"   entities with a wrong ID:       {lost[wrong].sum() / len(df):.4f}")
    print(f"   entities only missing matches:  {lost[(~single) & ~wrong].sum() / len(df):.4f}")


if __name__ == "__main__":
    main()