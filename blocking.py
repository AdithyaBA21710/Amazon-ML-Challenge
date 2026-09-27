"""
Blocking (candidate generation).

Direction: every Source 2 / Source 3 record searches INTO Source 1 for its
likely owner. S1 is smaller and deduplicated, and each S2/S3 record has at
most one owner, so this is the natural way round.

Three channels, each a TF-IDF nearest-neighbour search within one country:
  name : word tokens of the core name (+ alternate name)
  addr : word/number tokens of the address   (finds trade names like "Dovacira")
  char : character 3-grams of the name without spaces
         (finds "bnpgroupcompaniesdelhi", "jexfirst", typos)
Very common tokens (e.g. "enterprises", "st", "delhi") are dropped from each
index; they don't help find a specific business and they make search slow.

Usage:
    python src/blocking.py                  # on work/dev, prints a recall report
    python src/blocking.py --k 20           # keep more candidates per channel

Output: work/<split>/candidates.parquet
    query_id, s1_id, country, <channel>_sim, <channel>_rank, [is_match]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLS = ["entity_id", "country", "business_name", "business_address",
        "name_main", "name_alt", "name_nospace", "addr_norm"]

WORD_VEC = dict(analyzer="word", token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
CHANNELS = {
    "name": dict(text=lambda df: (df["name_main"] + " " + df["name_alt"]).str.strip(),
                 vec=WORD_VEC, max_df_frac=0.005, chunk_scale=1.0),
    "addr": dict(text=lambda df: df["addr_norm"],
                 vec=WORD_VEC, max_df_frac=0.005, chunk_scale=1.0),
    "char": dict(text=lambda df: df["name_nospace"],
                 vec=dict(analyzer="char", ngram_range=(3, 3), lowercase=False,
                          sublinear_tf=True),
                 max_df_frac=0.01, chunk_scale=0.5),
}


# ------------------------------------------------------------------ search
def topk_rows(M, k):
    """For each row of sparse matrix M, return the k largest entries."""
    indptr, indices, data = M.indptr, M.indices, M.data
    rows, cols, vals, ranks = [], [], [], []
    for i in range(M.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if a == b:
            continue
        d = data[a:b]
        top = np.argpartition(-d, k)[:k] if b - a > k else np.arange(b - a)
        order = top[np.argsort(-d[top], kind="stable")]
        n = len(order)
        rows.append(np.full(n, i, dtype=np.int64))
        cols.append(indices[a:b][order])
        vals.append(d[order])
        ranks.append(np.arange(1, n + 1, dtype=np.int16))
    if not rows:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float32), np.array([], dtype=np.int16)
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(vals), np.concatenate(ranks))


def run_channel(cfg, s1_df, q_df, k, chunk_rows):
    """TF-IDF index on S1, then top-k S1 matches for each query record."""
    s1_text = cfg["text"](s1_df).fillna("").tolist()
    q_text = cfg["text"](q_df).fillna("").tolist()
    max_df = max(50, int(cfg["max_df_frac"] * len(s1_text)))
    vec = TfidfVectorizer(**cfg["vec"], min_df=1, max_df=max_df, dtype=np.float32)
    try:
        X = vec.fit_transform(s1_text)
    except ValueError:  # empty vocabulary (tiny or empty country)
        return None
    XT = X.T.tocsr()
    Q = vec.transform(q_text)

    chunk = max(100, int(chunk_rows * cfg["chunk_scale"]))
    parts = []
    for start in range(0, Q.shape[0], chunk):
        M = (Q[start:start + chunk] @ XT).tocsr()
        q, s, v, r = topk_rows(M, k)
        parts.append((q + start, s, v, r))
    return tuple(np.concatenate(x) for x in zip(*parts))


def block_country(s1c, qc, k, chunk_rows):
    merged = None
    for name, cfg in CHANNELS.items():
        t0 = time.time()
        res = run_channel(cfg, s1c, qc, k, chunk_rows)
        if res is None:
            print(f"      {name:5s}: skipped (no usable tokens after dropping common ones)")
            continue
        q, s, v, r = res
        df = pd.DataFrame({"q": q, "s": s, f"{name}_sim": v, f"{name}_rank": r})
        print(f"      {name:5s}: {len(df):>10,} pairs  ({time.time() - t0:.0f}s)", flush=True)
        merged = df if merged is None else merged.merge(df, on=["q", "s"], how="outer")
    return merged


# ------------------------------------------------------------------ report
def f05_from_recall(r):
    return np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)


def report(cands, gt, lookup, k):
    truth = gt.assign(query_id=gt["matched_entity_ids"].str.split(",")).explode("query_id")
    truth = truth[truth["query_id"].notna() & (truth["query_id"] != "")]
    n_true = len(truth)

    print("\n" + "=" * 70)
    print("BLOCKING REPORT")
    print("=" * 70)
    print(f"True (S1, S2/S3) pairs: {n_true:,}")
    print(f"Candidate pairs:        {len(cands):,}")
    print(f"Candidates per query:   {len(cands) / max(lookup['n_queries'], 1):.1f}")
    print(f"Positive rate:          {cands['is_match'].mean():.2%}")

    ks = sorted({1, 3, 5, k})
    print("\nPair recall by channel (share of true pairs found within top-k):")
    print("   channel " + "".join(f"{'k=' + str(x):>9}" for x in ks))
    matches = cands[cands["is_match"] == 1]
    for ch in CHANNELS:
        col = f"{ch}_rank"
        if col not in cands:
            continue
        print(f"   {ch:7s} " + "".join(
            f"{(matches[col] <= x).sum() / n_true:>9.2%}" for x in ks))
    union = {x: matches[[c for c in cands if c.endswith('_rank')]].le(x).any(axis=1).sum() / n_true
             for x in ks}
    print(f"   {'UNION':7s} " + "".join(f"{union[x]:>9.2%}" for x in ks))

    print("\nUnion recall by country:")
    true_by_c = truth["query_id"].map(lookup["country"]).value_counts()
    found_by_c = matches["country"].value_counts()
    for c in sorted(true_by_c.index):
        print(f"   {c:8s} {found_by_c.get(c, 0) / true_by_c[c]:.2%}")

    # Ceiling: score if the matching model were perfect on these candidates
    found = matches.groupby("s1_id").size()
    n_truth = truth.groupby("source1_entity_id").size()
    per_entity = gt["source1_entity_id"].map(
        lambda e: 1.0 if n_truth.get(e, 0) == 0 else None)
    has = per_entity.isna()
    r = gt.loc[has, "source1_entity_id"].map(lambda e: found.get(e, 0) / n_truth[e])
    per_entity.loc[has] = f05_from_recall(r.to_numpy(dtype=float))
    print(f"\nCEILING macro F0.5 (perfect model on these candidates): "
          f"{per_entity.astype(float).mean():.4f}")

    found_pairs = set(zip(matches["s1_id"], matches["query_id"]))
    missed = truth[[(a, b) not in found_pairs for a, b in
                    zip(truth["source1_entity_id"], truth["query_id"])]]
    print(f"\nMissed true pairs: {len(missed):,}. Examples:")
    rec = lookup["records"]
    for a, b in missed[["source1_entity_id", "query_id"]].sample(
            min(12, len(missed)), random_state=0).itertuples(index=False, name=None):
        ra, rb = rec.loc[a], rec.loc[b]
        print(f"\n   {a}: {ra['business_name']!r} | {ra['business_address']!r}")
        print(f"   {b}: {rb['business_name']!r} | {rb['business_address']!r}")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Blocking / candidate generation")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--split", default="dev", help="dev or test")
    ap.add_argument("--k", type=int, default=10, help="candidates kept per channel")
    ap.add_argument("--chunk-rows", type=int, default=2000,
                    help="queries per matrix multiply; lower it if you run out of memory")
    args = ap.parse_args()

    d = Path(args.work_dir) / args.split
    s1 = pd.read_parquet(d / "source1.parquet", columns=COLS)
    queries = pd.concat([pd.read_parquet(d / f"source{i}.parquet", columns=COLS)
                         for i in (2, 3)], ignore_index=True)
    print(f"{args.split}: {len(s1):,} Source 1 records, {len(queries):,} S2/S3 queries")

    all_cands = []
    for country in sorted(s1["country"].unique()):   # any label works, France included
        s1c = s1[s1["country"] == country].reset_index(drop=True)
        qc = queries[queries["country"] == country].reset_index(drop=True)
        print(f"\n[{country}] {len(s1c):,} S1 records, {len(qc):,} queries")
        if len(qc) == 0:
            continue
        m = block_country(s1c, qc, args.k, args.chunk_rows)
        if m is None or m.empty:
            continue
        m["query_id"] = qc["entity_id"].to_numpy()[m["q"].to_numpy()]
        m["s1_id"] = s1c["entity_id"].to_numpy()[m["s"].to_numpy()]
        m["country"] = country
        all_cands.append(m.drop(columns=["q", "s"]))

    orphan = ~queries["country"].isin(s1["country"].unique())
    if orphan.any():
        print(f"\nNote: {orphan.sum():,} queries have a country with no Source 1 records")

    cands = pd.concat(all_cands, ignore_index=True)
    first = ["query_id", "s1_id", "country"]
    cands = cands[first + [c for c in cands.columns if c not in first]]

    gt_path = d / "ground_truth.parquet"
    if gt_path.exists():
        gt = pd.read_parquet(gt_path)
        owner = (gt.assign(q=gt["matched_entity_ids"].str.split(","))
                   .explode("q").dropna(subset=["q"]))
        owner = owner[owner["q"] != ""].set_index("q")["source1_entity_id"]
        cands["is_match"] = (cands["query_id"].map(owner) == cands["s1_id"]).astype("int8")
        records = pd.concat([s1, queries]).set_index("entity_id")
        lookup = {"records": records, "country": records["country"],
                  "n_queries": len(queries)}
        report(cands, gt, lookup, args.k)

    out = d / "candidates.parquet"
    cands.to_parquet(out, index=False)
    print(f"\nSaved {len(cands):,} candidate pairs to {out}")


if __name__ == "__main__":
    main()