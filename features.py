"""
Pair features for the matching model.

For every candidate pair (S1 entity, S2/S3 record) from blocking, compute
numbers describing how similar the two records are. The model learns from
these which pairs are true matches.

Feature groups:
  name    : several fuzzy similarities of the cleaned names (word order
            ignored, typos allowed, spaces removed, letters sorted, ...)
  address : fuzzy address similarities and house-number agreement
  flags   : missing address, web-style name, Indian-script name
  blocking: similarity score and rank from each blocking channel
  words   : how DISTINCTIVE the words are that the two names do not share
            (an extra "services" is harmless; "developers" instead of
            "constructions" usually means a different business)
  context : how this pair compares with the query's OTHER candidates
            (the best candidate is usually the owner)

No feature uses the country label, so France is scored the same way.

Usage:
    python src/features.py                # dev split
Output:
    work/<split>/features.parquet
"""
import argparse
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from normalize import INDIC_RE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REC_COLS = ["entity_id", "business_name", "name_main", "name_alt", "name_all",
            "name_nospace", "name_is_web", "addr_norm", "addr_nums"]
FIRST_NUM_RE = re.compile(r"\d+")
IDF_CAP = 10.0       # rare words all count as "very distinctive"; keeps dev and test comparable
CPD_WORKERS = -1     # threads for rapidfuzz; predict.py sets 1 when it runs many processes


# ------------------------------------------------------------- records
def load_records(d):
    recs = pd.concat([pd.read_parquet(d / f"source{i}.parquet", columns=REC_COLS)
                      for i in (1, 2, 3)], ignore_index=True)
    return prepare_records(recs)


def prepare_records(recs):
    """Add helper columns; index by entity_id. Strings kept as plain Python objects
    so repeated lookups are cheap."""
    recs = recs[REC_COLS].copy()
    for c in REC_COLS:
        if c != "name_is_web":
            recs[c] = recs[c].fillna("").astype(str).astype(object)
    recs["name_sorted"] = recs["name_nospace"].map(lambda s: "".join(sorted(s)))
    recs["name_ntok"] = recs["name_main"].map(lambda s: len(s.split()))
    recs["addr_ntok"] = recs["addr_norm"].map(lambda s: len(s.split()))
    recs["has_script"] = recs["business_name"].map(lambda s: int(bool(INDIC_RE.search(s))))
    recs["first_num"] = recs["addr_norm"].map(
        lambda s: (m.group() if (m := FIRST_NUM_RE.search(s)) else ""))
    recs["first_tok"] = recs["name_main"].map(lambda s: s.split()[0] if s else "")
    recs["name_toks"] = recs["name_main"].map(lambda s: tuple(dict.fromkeys(s.split())))
    return recs.drop(columns=["business_name"]).set_index("entity_id")


def word_idf(recs):
    """How distinctive each name word is, measured on Source 1 names:
    rare words get high values, generic ones ('services') low values."""
    s1 = recs.index.str.startswith("S1-")
    return word_idf_from_names(recs.loc[s1, "name_main"])


def word_idf_from_names(names):
    df = Counter(t for s in names for t in dict.fromkeys(s.split()))
    n = max(len(names), 1)
    idf = {t: min(math.log(n / (c + 1)), IDF_CAP) for t, c in df.items()}
    return idf, IDF_CAP  # unseen words count as very distinctive


# ---------------------------------------------------------- similarity
def is_empty(a):
    return np.fromiter((len(x) == 0 for x in a), count=len(a), dtype=bool)


def sim(scorer, a, b):
    """Element-wise similarity of two string arrays; NaN where either is empty."""
    out = cpdist(a, b, scorer=scorer, workers=CPD_WORKERS).astype(np.float32)
    out[is_empty(a) | is_empty(b)] = np.nan
    return out


def number_features(nums_a, nums_b, first_a, first_b):
    n = len(nums_a)
    overlap = np.zeros(n, np.float32)
    jacc = np.full(n, np.nan, np.float32)
    conflict = np.zeros(n, np.float32)
    first_eq = np.full(n, np.nan, np.float32)
    for i in range(n):
        sa, sb = nums_a[i], nums_b[i]
        if sa and sb:
            A, B = set(sa.split()), set(sb.split())
            inter = len(A & B)
            overlap[i] = inter
            jacc[i] = inter / len(A | B)
            conflict[i] = float(inter == 0)
        if first_a[i] and first_b[i]:
            first_eq[i] = float(first_a[i] == first_b[i])
    return overlap, jacc, conflict, first_eq


def _tok_match(t, others):
    """Is word t present in others, allowing small typos for longer words?"""
    if t in others:
        return True
    if len(t) >= 4:
        return any(len(o) >= 4 and fuzz.ratio(t, o) >= 85 for o in others)
    return False


def word_features(toks_a, toks_b, idf, default_idf):
    """Share / max / count of distinctive words each name has that the other lacks."""
    n = len(toks_a)
    out = np.full((n, 7), np.nan, np.float32)
    for i in range(n):
        A, B = toks_a[i], toks_b[i]
        if not A or not B:
            continue
        wa = [idf.get(t, default_idf) for t in A]
        wb = [idf.get(t, default_idf) for t in B]
        ua = [w for t, w in zip(A, wa) if not _tok_match(t, B)]
        ub = [w for t, w in zip(B, wb) if not _tok_match(t, A)]
        sa, sb = sum(wa), sum(wb)
        out[i] = (sum(ua) / sa if sa else 0.0, sum(ub) / sb if sb else 0.0,
                  max(ua, default=0.0), max(ub, default=0.0),
                  len(ua), len(ub), sa - sum(ua))
    return out


def first_number_features(first_a, first_b):
    n = len(first_a)
    absdiff = np.full(n, np.nan, np.float32)
    digit_sim = np.full(n, np.nan, np.float32)
    for i in range(n):
        a, b = first_a[i], first_b[i]
        if a and b:
            absdiff[i] = math.log1p(abs(float(a) - float(b)))
            digit_sim[i] = fuzz.ratio(a, b)
    return absdiff, digit_sim


def chunk_features(ch, recs, idf, default_idf):
    ia = recs.index.get_indexer(ch["s1_id"])
    ib = recs.index.get_indexer(ch["query_id"])
    if (ia < 0).any() or (ib < 0).any():
        raise ValueError("Candidate ids not found in source files")

    def col(name, idx):
        return recs[name].to_numpy()[idx]

    f = {}
    na, nb = col("name_main", ia), col("name_main", ib)
    f["n_ratio"] = sim(fuzz.ratio, na, nb)
    f["n_tsort"] = sim(fuzz.token_sort_ratio, na, nb)
    f["n_tset"] = sim(fuzz.token_set_ratio, na, nb)
    f["n_partial"] = sim(fuzz.partial_ratio, na, nb)
    f["n_jw"] = sim(JaroWinkler.normalized_similarity, na, nb)

    sa, sb = col("name_nospace", ia), col("name_nospace", ib)
    f["ns_ratio"] = sim(fuzz.ratio, sa, sb)
    f["ns_partial"] = sim(fuzz.partial_ratio, sa, sb)
    f["n_charbag"] = sim(fuzz.ratio, col("name_sorted", ia), col("name_sorted", ib))
    f["nall_tset"] = sim(fuzz.token_set_ratio, col("name_all", ia), col("name_all", ib))

    alt_ab = sim(fuzz.token_set_ratio, na, col("name_alt", ib))
    alt_ba = sim(fuzz.token_set_ratio, col("name_alt", ia), nb)
    f["n_alt"] = np.fmax(alt_ab, alt_ba)
    f["n_best"] = np.fmax(f["n_tset"], f["n_alt"])

    wf = word_features(col("name_toks", ia), col("name_toks", ib), idf, default_idf)
    for j, name in enumerate(["w_unmatched_share_a", "w_unmatched_share_b",
                              "w_unmatched_max_a", "w_unmatched_max_b",
                              "w_unmatched_n_a", "w_unmatched_n_b", "w_matched_idf"]):
        f[name] = wf[:, j]

    f["n_first_tok_eq"] = (col("first_tok", ia) == col("first_tok", ib)).astype(np.float32)
    ta, tb = col("name_ntok", ia).astype(np.float32), col("name_ntok", ib).astype(np.float32)
    f["n_ntok_a"], f["n_ntok_b"], f["n_ntok_diff"] = ta, tb, np.abs(ta - tb)
    f["web_b"] = col("name_is_web", ib).astype(np.float32)
    f["script_b"] = col("has_script", ib).astype(np.float32)

    aa, ab = col("addr_norm", ia), col("addr_norm", ib)
    f["a_ratio"] = sim(fuzz.ratio, aa, ab)
    f["a_tset"] = sim(fuzz.token_set_ratio, aa, ab)
    f["a_tsort"] = sim(fuzz.token_sort_ratio, aa, ab)
    f["a_partial"] = sim(fuzz.partial_ratio, aa, ab)
    f["a_missing_b"] = is_empty(ab).astype(np.float32)
    f["a_ntok_b"] = col("addr_ntok", ib).astype(np.float32)
    (f["num_overlap"], f["num_jacc"], f["num_conflict"],
     f["num_first_eq"]) = number_features(col("addr_nums", ia), col("addr_nums", ib),
                                          col("first_num", ia), col("first_num", ib))
    f["num_first_logdiff"], f["num_first_digit_sim"] = first_number_features(
        col("first_num", ia), col("first_num", ib))

    out = pd.DataFrame(f, index=ch.index)
    for c in ch.columns:
        if c.endswith("_sim") or c.endswith("_rank"):
            out["blk_" + c] = ch[c].astype(np.float32)
    return out


# -------------------------------------------------------------- context
def add_context(df):
    """Compare each pair with the other candidates of the same query."""
    q = df["query_id"]
    df["q_ncand"] = df.groupby(q, observed=True)["n_best"].transform("size").astype(np.float32)
    df["s1_ncand"] = df.groupby(df["s1_id"], observed=True)["n_best"].transform("size").astype(np.float32)

    df["combo"] = (df["n_best"].fillna(0) + df["a_tset"].fillna(0) + df["n_charbag"].fillna(0)) / 3
    for c in ["n_best", "a_tset", "n_charbag", "combo"]:
        df[f"{c}_gap"] = (df[c] - df.groupby(q, observed=True)[c].transform("max")).astype(np.float32)

    # rank of this pair within its query by combo, and margin over the runner-up
    tmp = df[["query_id", "combo"]].sort_values(["query_id", "combo"], ascending=[True, False])
    g = tmp.groupby("query_id", observed=True)
    rank = g.cumcount()
    m1 = g["combo"].transform("first")
    m2 = tmp["combo"].where(rank == 1).groupby(tmp["query_id"], observed=True).transform("max")
    df["combo_rank"] = rank.reindex(df.index).astype(np.float32)
    m1, m2 = m1.reindex(df.index), m2.reindex(df.index)
    df["combo_margin"] = np.where(df["combo_rank"] == 0, df["combo"] - m2, df["combo"] - m1)
    df["combo_margin"] = df["combo_margin"].astype(np.float32)
    df["combo"] = df["combo"].astype(np.float32)
    return df


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Compute pair features")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="pairs per batch")
    args = ap.parse_args()
    d = Path(args.work_dir) / args.split

    t0 = time.time()
    recs = load_records(d)
    idf, default_idf = word_idf(recs)
    cands = pd.read_parquet(d / "candidates.parquet")
    print(f"{len(recs):,} records, {len(cands):,} candidate pairs "
          f"(loaded in {time.time() - t0:.0f}s)")

    parts = []
    for start in range(0, len(cands), args.chunk):
        ch = cands.iloc[start:start + args.chunk]
        parts.append(chunk_features(ch, recs, idf, default_idf))
        print(f"   {min(start + args.chunk, len(cands)):,} pairs  ({time.time() - t0:.0f}s)",
              flush=True)
    feats = pd.concat(parts)
    del parts, recs

    keep = ["query_id", "s1_id", "country"] + (["is_match"] if "is_match" in cands else [])
    df = pd.concat([cands[keep], feats], axis=1)
    del feats, cands
    df["query_id"] = df["query_id"].astype("category")
    df["s1_id"] = df["s1_id"].astype("category")

    print("Adding context features...", flush=True)
    df = add_context(df)

    out = d / "features.parquet"
    df.to_parquet(out, index=False)
    n_feat = len([c for c in df.columns if c not in keep])
    print(f"\nSaved {len(df):,} pairs x {n_feat} features to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()