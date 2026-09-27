"""
Pick the decision threshold, and report the realistic score, on HALF B of the
training entities (never used for training).

Needs the full-density predictions on the training set:
    python src/predict.py --split train --overwrite
Then:
    python src/tune.py              # prints the sweep and saves the best threshold
    python src/tune.py --dry-run    # prints only

Score = macro F0.5 over half-B Source 1 entities, singletons included, exactly
like the leaderboard. A record assigned to a half-B entity counts against it
even if its true owner is in half A.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_dense import half_of, owner_series
from config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def macro_f05(acc, n_true, ids):
    """acc: accepted pairs with s1_id, correct. Returns per-entity F0.5 over ids."""
    g = acc.groupby("s1_id")["correct"].agg(["sum", "size"])
    tp = g["sum"].reindex(ids, fill_value=0).to_numpy(float)
    npred = g["size"].reindex(ids, fill_value=0).to_numpy(float)
    nt = n_true.reindex(ids, fill_value=0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / npred, tp / nt
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((nt == 0) & (npred == 0), 1.0, f)
    return f, tp, npred, nt


def main():
    ap = argparse.ArgumentParser(description="Tune threshold on half B")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--dry-run", action="store_true", help="don't save the threshold")
    ap.add_argument("--support", action="store_true",
                    help="also tune the two entity-level rules (support + lone-match)")
    ap.add_argument("--by-country", action="store_true",
                    help="also tune a separate threshold per training country "
                         "(countries not in training keep the global one)")
    args = ap.parse_args()
    work = Path(args.work_dir)
    d = work / "train"

    files = sorted(d.glob("best_*.parquet"))
    if not files:
        raise SystemExit("No best_*.parquet in work/train. Run: python src/predict.py --split train")
    best = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    owner = owner_series(args.data_dir)
    best["correct"] = best["query_id"].map(owner).to_numpy() == best["s1_id"].to_numpy()

    s1 = pd.read_parquet(d / "source1.parquet", columns=["entity_id", "country"])
    s1 = s1[half_of(s1["entity_id"]) == 1]            # half B only
    ids = s1["entity_id"].to_numpy()
    n_true = owner.index.to_series().groupby(owner.values).size()
    best = best[best["s1_id"].isin(set(ids))]
    print(f"Scoring {len(ids):,} half-B entities")

    results = []
    grid = np.round(np.arange(0.30, 0.99, 0.02), 2)
    for t in grid:
        f, *_ = macro_f05(best[best["prob"] >= t], n_true, ids)
        results.append((t, f.mean()))
    t_best, score = max(results, key=lambda x: x[1])
    for t in np.round(np.arange(max(0.01, t_best - 0.02), min(0.995, t_best + 0.021), 0.005), 3):
        f, *_ = macro_f05(best[best["prob"] >= t], n_true, ids)
        if f.mean() > score:
            t_best, score = float(t), f.mean()

    print("\nThreshold sweep:")
    for t, sc in results:
        mark = "  <-" if abs(t - round(t_best, 2)) < 1e-9 else ""
        print(f"   t={t:.2f}  macro F0.5 = {sc:.4f}{mark}")

    f, tp, npred, nt = macro_f05(best[best["prob"] >= t_best], n_true, ids)
    df = pd.DataFrame({"country": s1["country"].to_numpy(), "f": f, "tp": tp,
                       "npred": npred, "nt": nt})
    print("\n" + "=" * 70)
    print(f"REALISTIC MACRO F0.5 (half B) = {score:.4f}   threshold {t_best:.3f}")
    print("=" * 70)
    print(f"Pair precision {df['tp'].sum() / max(df['npred'].sum(), 1):.2%}   "
          f"pair recall {df['tp'].sum() / max(df['nt'].sum(), 1):.2%}")
    for c, g in df.groupby("country"):
        print(f"   {c:8s} {g['f'].mean():.4f}   ({len(g):,} entities)")
    single = df["nt"] == 0
    wrong = (~single) & (df["npred"] > df["tp"])
    lost = 1 - df["f"]
    print(f"Singletons correctly left empty: {(df.loc[single, 'npred'] == 0).mean():.1%}")
    print("Score lost, by cause:")
    print(f"   singletons given a match:       {lost[single].sum() / len(df):.4f}")
    print(f"   entities with a wrong ID:       {lost[wrong].sum() / len(df):.4f}")
    print(f"   entities only missing matches:  {lost[(~single) & ~wrong].sum() / len(df):.4f}")

    support = {}
    if args.support:
        print("\nEntity-level rules (half B):")
        base = best["prob"].to_numpy() >= t_best
        supported = set(best.loc[base, "s1_id"])
        in_sup = best["s1_id"].isin(supported).to_numpy()
        prob = best["prob"].to_numpy()
        # Rule 1: accept weaker records if their business already has a confident match
        r1 = [(None, score)]
        for tl in np.round(np.arange(0.30, t_best, 0.05), 2):
            acc = best[base | ((prob >= tl) & in_sup)]
            r1.append((float(tl), macro_f05(acc, n_true, ids)[0].mean()))
        tl_best, sc1 = max(r1, key=lambda x: x[1])
        for tl, sc in r1[1:]:
            print(f"   support threshold {tl:.2f}: {sc:.4f}")
        keep1 = base | ((prob >= tl_best) & in_sup) if tl_best is not None else base
        # Rule 2: drop a business's only match if it is not confident enough
        acc1 = best[keep1]
        r2 = [(None, sc1)]
        cnt = acc1.groupby("s1_id")["s1_id"].transform("size").to_numpy()
        for ts in np.round(np.arange(t_best + 0.01, 0.995, 0.02), 3):
            drop = (cnt == 1) & (acc1["prob"].to_numpy() < ts)
            r2.append((float(ts), macro_f05(acc1[~drop], n_true, ids)[0].mean()))
        ts_best, sc2 = max(r2, key=lambda x: x[1])
        for ts, sc in r2[1:]:
            print(f"   lone-match threshold {ts:.3f}: {sc:.4f}")
        print(f"   RESULT: support {tl_best}, lone-match {ts_best}: {sc2:.4f} "
              f"(threshold only: {score:.4f}, gain {sc2 - score:+.4f})")
        if tl_best is not None:
            support["support_threshold"] = tl_best
        if ts_best is not None:
            support["lone_threshold"] = ts_best

    country_thr = {}
    if args.by_country:
        print("\nPer-country thresholds (half B):")
        s1_country = pd.Series(s1["country"].to_numpy(), index=ids)
        best_c = best.assign(country=best["s1_id"].map(s1_country))
        fine_grid = np.round(np.arange(0.30, 0.99, 0.005), 3)
        parts_f = []
        for c in sorted(s1["country"].unique()):
            c_ids = ids[s1["country"].to_numpy() == c]
            b = best_c[best_c["country"] == c]
            scores = [(t, macro_f05(b[b["prob"] >= t], n_true, c_ids)[0].mean()) for t in fine_grid]
            t_c, sc_c = max(scores, key=lambda x: x[1])
            sc_glob = macro_f05(b[b["prob"] >= t_best], n_true, c_ids)[0].mean()
            country_thr[c] = float(t_c)
            parts_f.append(macro_f05(b[b["prob"] >= t_c], n_true, c_ids)[0])
            print(f"   {c:8s} best threshold {t_c:.3f}: {sc_c:.4f}  (global {t_best:.3f}: {sc_glob:.4f})")
        combined = np.concatenate(parts_f).mean()
        print(f"   Combined half-B score with per-country thresholds: {combined:.4f} "
              f"(global only: {score:.4f}, gain {combined - score:+.4f})")

    if not args.dry_run:
        cfg_path = work / "model_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.update(threshold=float(t_best), threshold_tuned=True, halfB_score=float(score))
        if args.by_country:
            cfg["country_thresholds"] = country_thr
        if args.support:
            cfg.pop("support_threshold", None)
            cfg.pop("lone_threshold", None)
            cfg.update(support)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"\nSaved threshold {t_best:.3f} to {cfg_path}")
        print("Next: python src/predict.py --overwrite   (test set, new model)")


if __name__ == "__main__":
    main()