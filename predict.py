"""
Run the pipeline on a split and write submission-format files.

For each country label in Source 1 (France included, no special casing):
  1. blocking : candidates for every S2/S3 record (saved as blk_<country>.parquet
                and reused on later runs; --reblock to redo)
  2. features : exactly the same features as in training
  3. model    : match probability for every candidate pair
  4. keep each S2/S3 record's most likely owner (best_<country>.parquet)
Finally the threshold from work/model_config.json is applied and two files are
written:
    matching_results.tsv   final matches          -> upload to the leaderboard
    candidate_pairs.tsv    what the model scored  -> required in the final zip
Finished countries are saved, so re-running with a new threshold only takes a
minute (use --overwrite to rescore with a new model).

Usage (Linux / EC2 recommended):
    python src/predict.py                   # test set  -> output/
    python src/predict.py --split train     # training set -> output_train/ (for tune.py)
    python src/predict.py --overwrite       # rescore after training a new model
"""
import argparse
import gc
import json
import os
import re
import time
from multiprocessing import get_context
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
from blocking import COLS, block_country, can_fork
from features import add_context, chunk_features, prepare_records, word_idf_from_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
READ_COLS = sorted(set(COLS) | set(F.REC_COLS) | {"country"})

_W = {}  # state shared with worker processes (inherited via fork)


# ------------------------------------------------------------ data helpers
def safe_name(country):
    return re.sub(r"\W+", "_", str(country))


def load_split(d, countries=None):
    s1 = pd.read_parquet(d / "source1.parquet", columns=READ_COLS)
    queries = pd.concat([pd.read_parquet(d / f"source{i}.parquet", columns=READ_COLS)
                         for i in (2, 3)], ignore_index=True)
    if countries:
        s1 = s1[s1["country"].isin(countries)].reset_index(drop=True)
        queries = queries[queries["country"].isin(countries)].reset_index(drop=True)
    return s1, queries


def country_frames(s1, queries, country):
    return (s1[s1["country"] == country].reset_index(drop=True),
            queries[queries["country"] == country].reset_index(drop=True))


def split_idf(s1):
    return word_idf_from_names(s1["name_main"].fillna("").astype(str).tolist())


def get_candidates(d, country, s1c, qc, args):
    """Blocking output for one country, cached on disk."""
    path = d / f"blk_{safe_name(country)}.parquet"
    if path.exists() and not getattr(args, "reblock", False):
        m = pd.read_parquet(path)
        print(f"   loaded {len(m):,} candidate pairs from {path.name}", flush=True)
        return m
    print("   blocking...", flush=True)
    m = block_country(s1c, qc, args.k, args.chunk_rows, args.workers)
    if m is None or m.empty:
        return None
    m["q"] = m["q"].astype(np.int32)
    m["s"] = m["s"].astype(np.int32)
    for c in m.columns:
        if c not in ("q", "s"):
            m[c] = m[c].astype(np.float32)
    m = m.sort_values(["q", "s"], kind="stable").reset_index(drop=True)
    m.to_parquet(path, index=False)
    return m


# ------------------------------------------------------- feature workers
def setup_worker_state(m_counts, m_rows, s1c, qc, idf, default_idf, features=None, extra=None):
    """m_counts: all candidate pairs of the country (for per-S1 counts);
    m_rows: the pairs to featurize (whole queries only)."""
    _W.clear()
    _W.update(m=m_rows, recs=prepare_records(pd.concat([s1c, qc], ignore_index=True)),
              idf=idf, default_idf=default_idf,
              q_ids=qc["entity_id"].to_numpy(object), s1_ids=s1c["entity_id"].to_numpy(object),
              s1_counts=m_counts["s"].value_counts(), features=features)
    if extra:
        _W.update(extra)


def features_for(bounds):
    a, b = bounds
    ch = _W["m"].iloc[a:b].copy()
    ch["query_id"] = _W["q_ids"][ch["q"].to_numpy()]
    ch["s1_id"] = _W["s1_ids"][ch["s"].to_numpy()]
    feats = chunk_features(ch, _W["recs"], _W["idf"], _W["default_idf"])
    df = add_context(pd.concat([ch[["query_id", "s1_id"]], feats], axis=1))
    # an S1 entity's candidate count is counted over the whole country
    df["s1_ncand"] = ch["s"].map(_W["s1_counts"]).to_numpy(np.float32)
    df["q"] = ch["q"].to_numpy(np.int64)
    df["s"] = ch["s"].to_numpy(np.int64)
    return df


def _init_worker(model_path):
    if model_path:
        _W["model"] = lgb.Booster(model_file=model_path)


def score_batch(bounds):
    df = features_for(bounds)
    prob = _W["model"].predict(df.reindex(columns=_W["features"]), num_threads=_W["threads"])
    return df["q"].to_numpy(), df["s"].to_numpy(), prob.astype(np.float32)


def run_batches(func, batches, workers, model_path=None, t0=None):
    parallel = workers > 1 and can_fork()
    F.CPD_WORKERS = 1 if parallel else -1
    _W["threads"] = 1 if parallel else 0
    t0 = t0 or time.time()
    out = []

    def progress(i):
        if i % 10 == 0 or i == len(batches):
            print(f"      {i}/{len(batches)} batches  ({time.time() - t0:.0f}s)", flush=True)

    if parallel:
        gc.freeze()
        with get_context("fork").Pool(workers, initializer=_init_worker,
                                      initargs=(model_path,)) as pool:
            for i, r in enumerate(pool.imap(func, batches), 1):
                out.append(r)
                progress(i)
        gc.unfreeze()
    else:
        _init_worker(model_path)
        for i, bnd in enumerate(batches, 1):
            out.append(func(bnd))
            progress(i)
    return out


def query_batches(q_sorted, size):
    """Split sorted rows into batches of ~size rows without splitting a query."""
    n, a, out = len(q_sorted), 0, []
    while a < n:
        b = min(a + size, n)
        if b < n:
            b = int(np.searchsorted(q_sorted, q_sorted[b - 1], side="right"))
        out.append((a, b))
        a = b
    return out


def join_by_group(keys, values):
    """keys sorted; returns {key: 'v1,v2,...'}"""
    if len(keys) == 0:
        return {}
    cut = np.flatnonzero(np.diff(keys)) + 1
    starts = np.r_[0, cut]
    return {keys[i]: ",".join(g) for i, g in zip(starts, np.split(values, cut))}


# ------------------------------------------------------------ per country
def run_country(country, s1c, qc, idf, default_idf, args, d, model_path, cfg):
    t0 = time.time()
    m = get_candidates(d, country, s1c, qc, args)
    if m is None:
        return (pd.DataFrame({"s1_id": [], "candidates": []}),
                pd.DataFrame({"query_id": [], "s1_id": [], "prob": []}))
    print(f"   {len(m):,} candidate pairs  ({time.time() - t0:.0f}s)", flush=True)

    setup_worker_state(m, m, s1c, qc, idf, default_idf, cfg["features"])
    q_ids, s1_ids = _W["q_ids"], _W["s1_ids"]
    batches = query_batches(m["q"].to_numpy(), args.batch)
    print(f"   features + model on {len(batches)} batches...", flush=True)
    res = run_batches(score_batch, batches, args.workers, model_path, t0)
    _W.clear()
    del m
    gc.collect()

    q = np.concatenate([r[0] for r in res])
    s = np.concatenate([r[1] for r in res])
    p = np.concatenate([r[2] for r in res])

    order = np.lexsort((q, s))                   # candidate lists per S1 entity
    cand = join_by_group(s[order], q_ids[q[order]])
    keys = np.array(list(cand.keys()), dtype=np.int64)
    cand_df = pd.DataFrame({"s1_id": s1_ids[keys] if len(keys) else [],
                            "candidates": list(cand.values())})

    order = np.lexsort((-p, q))                  # most likely owner per record
    first = np.r_[True, q[order][1:] != q[order][:-1]]
    best = order[first]
    best_df = pd.DataFrame({"query_id": q_ids[q[best]], "s1_id": s1_ids[s[best]],
                            "prob": p[best]})
    print(f"   done  ({time.time() - t0:.0f}s)", flush=True)
    return cand_df, best_df


# ------------------------------------------------------------------- main
def write_tsv(path, header, ids, values):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for i, v in zip(ids, values):
            f.write(f"{i}\t{v}\n")


def main():
    ap = argparse.ArgumentParser(description="Predict on a split")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--split", default="test", help="test (submission) or train (for tune.py)")
    ap.add_argument("--countries", nargs="*", default=None, help="only these labels")
    ap.add_argument("--out-dir", default=None,
                    help="default: output/ for test, output_<split>/ otherwise")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold in model_config.json")
    ap.add_argument("--country-threshold", nargs="*", default=[], metavar="COUNTRY=VALUE",
                    help="per-country thresholds, e.g. France=0.70 (default: same for all)")
    ap.add_argument("--k", type=int, default=10, help="must match the blocking used in training")
    ap.add_argument("--chunk-rows", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=500_000, help="pairs per feature batch")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--overwrite", action="store_true", help="rescore every country (new model)")
    ap.add_argument("--reblock", action="store_true", help="redo blocking instead of reusing it")
    args = ap.parse_args()

    work = Path(args.work_dir)
    d = work / args.split
    model_path = str(work / "model_lgb.txt")
    cfg = json.loads((work / "model_config.json").read_text())
    thr = args.threshold if args.threshold is not None else cfg["threshold"]
    country_thr = dict(cfg.get("country_thresholds", {}))
    for item in args.country_threshold:
        name, value = item.split("=")
        country_thr[name] = float(value)
    print(f"Model: {len(cfg['features'])} features, threshold {thr:.2f}, workers {args.workers}")
    if country_thr:
        print(f"Per-country thresholds: {country_thr}")

    s1, queries = load_split(d, args.countries)
    print(f"{args.split}: {len(s1):,} Source 1 records, {len(queries):,} S2/S3 records")
    idf, default_idf = split_idf(s1)

    t0 = time.time()
    cands, bests = [], []
    for country in sorted(s1["country"].unique()):   # any label, France included
        cpath = d / f"cand_{safe_name(country)}.parquet"
        bpath = d / f"best_{safe_name(country)}.parquet"
        if cpath.exists() and bpath.exists() and not args.overwrite:
            print(f"\n[{country}] already scored, loading")
            cands.append(pd.read_parquet(cpath))
            bests.append(pd.read_parquet(bpath).assign(country=country))
            continue
        s1c, qc = country_frames(s1, queries, country)
        print(f"\n[{country}] {len(s1c):,} S1 records, {len(qc):,} S2/S3 records", flush=True)
        if len(qc) == 0:
            continue
        c_df, b_df = run_country(country, s1c, qc, idf, default_idf, args, d, model_path, cfg)
        c_df.to_parquet(cpath, index=False)
        b_df.to_parquet(bpath, index=False)
        cands.append(c_df)
        bests.append(b_df.assign(country=country))
        gc.collect()

    # apply the threshold and assemble the two files
    best = pd.concat(bests, ignore_index=True) if bests else pd.DataFrame(
        {"query_id": [], "s1_id": [], "prob": [], "country": []})
    row_thr = best["country"].map(country_thr).fillna(thr).to_numpy(float)
    prob = best["prob"].to_numpy()
    keep = prob >= row_thr
    sup_t, lone_t = cfg.get("support_threshold"), cfg.get("lone_threshold")
    if sup_t is not None:   # weaker records accepted if their business has a confident match
        supported = set(best.loc[keep, "s1_id"])
        keep = keep | ((prob >= sup_t) & best["s1_id"].isin(supported).to_numpy())
    acc = best[keep]
    if lone_t is not None:  # a business's only match must be confident
        cnt = acc.groupby("s1_id")["s1_id"].transform("size").to_numpy()
        acc = acc[~((cnt == 1) & (acc["prob"].to_numpy() < lone_t))]
    if sup_t is not None or lone_t is not None:
        print(f"Entity-level rules: support {sup_t}, lone-match {lone_t}")
    acc = acc.sort_values(["s1_id", "query_id"])
    matches = acc.groupby("s1_id", sort=False)["query_id"].agg(",".join)
    cand = (pd.concat(cands, ignore_index=True) if cands
            else pd.DataFrame({"s1_id": [], "candidates": []})).set_index("s1_id")["candidates"]

    all_ids = s1["entity_id"].tolist()          # every S1 entity gets a row
    match_list = matches.reindex(all_ids).fillna("").tolist()
    cand_list = cand.reindex(all_ids).fillna("").tolist()

    default_out = "output" if args.split == "test" else f"output_{args.split}"
    out = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / default_out
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "matching_results.tsv", "matched_entity_ids", all_ids, match_list)
    write_tsv(out / "candidate_pairs.tsv", "candidate_entity_ids", all_ids, cand_list)

    n_matched = sum(1 for x in match_list if x)
    print("\n" + "=" * 70)
    print(f"Wrote {out / 'matching_results.tsv'}  and  {out / 'candidate_pairs.tsv'}")
    print(f"Threshold {thr:.3f} {country_thr or ''}: {len(all_ids):,} S1 entities, {n_matched:,} with matches, "
          f"{len(all_ids) - n_matched:,} left empty "
          f"({(len(all_ids) - n_matched) / max(len(all_ids), 1):.1%})")
    has = pd.Series([bool(x) for x in match_list], index=all_ids)
    for c, g in s1.groupby("country"):
        print(f"   {c:8s} {len(g):>9,} entities, {has.loc[g['entity_id']].mean():.1%} with matches")
    print(f"Total time {time.time() - t0:.0f}s")
    if args.split == "test":
        print("\nNext: run the validator from the student_resource folder.")
    else:
        print("\nNext: python src/tune.py")


if __name__ == "__main__":
    main()