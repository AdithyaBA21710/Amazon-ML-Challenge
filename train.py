"""
Train the matching model and measure the real dev score.

Steps:
  1. Cross-validation: LightGBM is trained 5 times, each time leaving out one
     fifth of the S1 entities, and predicts the part it did not see. Every
     pair therefore gets an honest "out-of-fold" (OOF) probability.
  2. Decision rules on those probabilities:
       - one-owner rule: each S2/S3 record keeps only its most likely owner
       - threshold: that owner is accepted only if probability >= t
     t is chosen to maximise the competition metric (macro F0.5) directly.
  3. Report: dev score vs the blocking ceiling, per country, singletons,
     feature importance, and examples of mistakes.
  4. Train a final model on all dev pairs and save it with the threshold,
     ready for the test set.

Usage:
    python src/train.py
Output:
    work/model_lgb.txt, work/model_config.json, work/dev/oof_predictions.parquet
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_COLS = ["query_id", "s1_id", "country", "is_match"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127,
              min_data_in_leaf=200, feature_fraction=0.85, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1)


# ---------------------------------------------------------------- scoring
def macro_f05(pred_pairs, gt):
    """pred_pairs: DataFrame(s1_id, is_match) of accepted pairs. gt: dev ground truth."""
    n_true = gt["matched_entity_ids"].map(lambda x: len(x.split(",")) if x else 0)
    n_true.index = gt["source1_entity_id"]
    g = pred_pairs.groupby("s1_id", observed=True)["is_match"].agg(["sum", "size"])
    tp = g["sum"].reindex(n_true.index, fill_value=0).to_numpy(float)
    npred = g["size"].reindex(n_true.index, fill_value=0).to_numpy(float)
    nt = n_true.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / npred, tp / nt
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((nt == 0) & (npred == 0), 1.0, f)
    return f, n_true.index


def decide(df, prob_col, t):
    """One-owner rule + threshold. Returns the accepted pairs."""
    best = df.loc[df.groupby("query_id", observed=True)[prob_col].idxmax()]
    return best[best[prob_col] >= t]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Train matcher and evaluate on dev")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-frac", type=float, default=0.3,
                    help="share of non-matching pairs used for training (speed)")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    work = Path(args.work_dir)
    d = work / "dev"

    t0 = time.time()
    df = pd.read_parquet(d / "features.parquet")
    gt = pd.read_parquet(d / "ground_truth.parquet")
    features = [c for c in df.columns if c not in ID_COLS]
    print(f"{len(df):,} pairs, {len(features)} features, "
          f"{df['is_match'].mean():.2%} positive  ({time.time() - t0:.0f}s)")

    # ---- 1. cross-validation, grouped by S1 entity
    rng = np.random.default_rng(args.seed)
    oof = np.zeros(len(df), np.float32)
    groups = df["s1_id"].cat.codes.to_numpy()
    y = df["is_match"].to_numpy()
    X = df[features]
    best_iters = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups)):
        keep = (y[tr] == 1) | (rng.random(len(tr)) < args.neg_frac)
        tr = tr[keep]
        dtr = lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        model = lgb.train({**PARAMS, "seed": args.seed}, dtr, args.rounds,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        best_iters.append(model.best_iteration)
        print(f"   fold {k + 1}/{args.folds}: {model.best_iteration} trees  "
              f"({time.time() - t0:.0f}s)", flush=True)
    df["prob"] = oof

    # ---- 2. threshold sweep on the real metric
    print("\nThreshold sweep (one-owner rule applied):")
    best_row = df.loc[df.groupby("query_id", observed=True)["prob"].idxmax()]
    results = []
    for t in np.round(np.arange(0.10, 0.96, 0.05), 2):
        acc = best_row[best_row["prob"] >= t]
        f, _ = macro_f05(acc, gt)
        results.append((t, f.mean()))
        print(f"   t={t:.2f}  macro F0.5 = {f.mean():.4f}")
    t_best, score = max(results, key=lambda x: x[1])
    fine = np.round(np.arange(max(0.01, t_best - 0.05), min(0.99, t_best + 0.05), 0.01), 2)
    for t in fine:
        f, _ = macro_f05(best_row[best_row["prob"] >= t], gt)
        if f.mean() > score:
            t_best, score = t, f.mean()

    # ---- 3. report
    acc = best_row[best_row["prob"] >= t_best]
    f, ids = macro_f05(acc, gt)
    per = pd.DataFrame({"f": f}, index=ids)
    s1_country = df.drop_duplicates("s1_id").set_index("s1_id")["country"].astype(str)
    per["country"] = per.index.map(s1_country)
    n_true = gt.set_index("source1_entity_id")["matched_entity_ids"].map(
        lambda x: len(x.split(",")) if x else 0)
    per["singleton"] = n_true.reindex(per.index).to_numpy() == 0

    print("\n" + "=" * 70)
    print(f"DEV MACRO F0.5 = {score:.4f}   (threshold {t_best:.2f})")
    print("=" * 70)
    print(f"Accepted pairs: {len(acc):,}   precision {acc['is_match'].mean():.2%}   "
          f"recall {acc['is_match'].sum() / n_true.sum():.2%}")
    print("By country:")
    for c, v in per.groupby("country")["f"].mean().items():
        print(f"   {c:8s} {v:.4f}")
    print(f"Singletons: {per.loc[per['singleton'], 'f'].mean():.2%} correctly left empty")

    imp = pd.Series(model.feature_importance("gain"), index=features)
    print("\nTop 15 features (last fold, by gain):")
    for name, v in (imp / imp.sum()).sort_values(ascending=False).head(15).items():
        print(f"   {name:18s} {v:.1%}")

    recs = pd.concat([pd.read_parquet(d / f"source{i}.parquet",
                                      columns=["entity_id", "business_name", "business_address"])
                      for i in (1, 2, 3)]).set_index("entity_id")

    def show(rows, title):
        print(f"\n{title}")
        for q, s, p in rows[["query_id", "s1_id", "prob"]].itertuples(index=False):
            a, b = recs.loc[s], recs.loc[q]
            print(f"   p={p:.2f}  {s}: {a['business_name']!r} | {a['business_address']!r}")
            print(f"            {q}: {b['business_name']!r} | {b['business_address']!r}")

    fp = acc[acc["is_match"] == 0]
    show(fp.sample(min(8, len(fp)), random_state=0), "False merges (accepted, but wrong):")
    missed = df[(df["is_match"] == 1) & ~df.index.isin(acc.index)]
    show(missed.sample(min(8, len(missed)), random_state=0),
         "Missed matches (true pair in candidates, not accepted):")

    df[["query_id", "s1_id", "is_match", "prob"]].to_parquet(
        d / "oof_predictions.parquet", index=False)

    # ---- 4. final model on all dev pairs
    print("\nTraining final model on all dev pairs...", flush=True)
    keep = (y == 1) | (rng.random(len(y)) < args.neg_frac)
    n_rounds = int(np.mean(best_iters) * 1.1)
    final = lgb.train({**PARAMS, "seed": args.seed},
                      lgb.Dataset(X[keep], y[keep]), n_rounds)
    final.save_model(str(work / "model_lgb.txt"))
    (work / "model_config.json").write_text(json.dumps(
        {"threshold": float(t_best), "features": features, "dev_score": float(score),
         "neg_frac": args.neg_frac, "rounds": n_rounds}, indent=2))
    print(f"Saved model and threshold to {work}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()