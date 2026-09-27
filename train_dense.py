"""
Train the matching model on the realistic (full-density) training set built by
build_dense.py.

10% of the sampled records are held out for early stopping and a quick check;
the final model is then trained on everything. The decision threshold is set
afterwards by tune.py on half B, which the model never sees.

Usage:
    python src/train_dense.py
Output:
    work/model_lgb.txt, work/model_config.json
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_COLS = {"q", "s", "is_match", "country", "query_id", "s1_id"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=200, feature_fraction=0.85, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1)


def main():
    ap = argparse.ArgumentParser(description="Train on full-density features")
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "work"))
    ap.add_argument("--neg-frac", type=float, default=0.5,
                    help="share of non-matching pairs used for training")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    work = Path(args.work_dir)

    t0 = time.time()
    files = sorted((work / "train").glob("dense_*.parquet"))
    if not files:
        raise SystemExit("No dense_*.parquet files found. Run src/build_dense.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    features = [c for c in df.columns if c not in ID_COLS]
    print(f"{len(df):,} pairs from {len(files)} files, {len(features)} features, "
          f"{df['is_match'].mean():.2%} positive  ({time.time() - t0:.0f}s)")

    # hold out 10% of records (whole records, so all their candidates go together)
    rng = np.random.default_rng(args.seed)
    group = pd.factorize(df["country"].astype(str) + ":" + df["q"].astype(str))[0]
    val_group = rng.random(group.max() + 1) < 0.10
    is_val = val_group[group]
    y = df["is_match"].to_numpy()
    keep = (~is_val) & ((y == 1) | (rng.random(len(df)) < args.neg_frac))
    X = df[features]

    dtr = lgb.Dataset(X[keep], y[keep], free_raw_data=True)
    dva = lgb.Dataset(X[is_val], y[is_val], reference=dtr)
    model = lgb.train({**PARAMS, "seed": args.seed}, dtr, args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(200)])
    best_iter = model.best_iteration
    print(f"Best iteration {best_iter}  ({time.time() - t0:.0f}s)")

    # quick check on held-out records: pick each record's best candidate
    val = df.loc[is_val, ["country", "q", "s", "is_match"]].copy()
    val["prob"] = model.predict(X[is_val], num_iteration=best_iter)
    print(f"Held-out AUC {roc_auc_score(val['is_match'], val['prob']):.5f}")
    top = val.sort_values("prob", ascending=False).drop_duplicates(["country", "q"])
    owned = val.groupby(["country", "q"])["is_match"].max()
    top = top.join(owned.rename("owned"), on=["country", "q"])
    print("Held-out records, keeping each record's best candidate:")
    for t in (0.5, 0.7, 0.9):
        acc = top[top["prob"] >= t]
        prec = acc["is_match"].mean() if len(acc) else float("nan")
        rec = acc["is_match"].sum() / max(top["owned"].sum(), 1)
        print(f"   threshold {t:.1f}: precision {prec:.2%}, recall {rec:.2%}")

    imp = pd.Series(model.feature_importance("gain"), index=features)
    print("\nTop 15 features by gain:")
    for name, v in (imp / imp.sum()).sort_values(ascending=False).head(15).items():
        print(f"   {name:20s} {v:.1%}")

    print("\nTraining final model on all sampled pairs...", flush=True)
    keep_all = (y == 1) | (rng.random(len(df)) < args.neg_frac)
    n_rounds = int(best_iter * 1.1)
    final = lgb.train({**PARAMS, "seed": args.seed}, lgb.Dataset(X[keep_all], y[keep_all]),
                      n_rounds)
    final.save_model(str(work / "model_lgb.txt"))

    cfg_path = work / "model_config.json"
    old = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg_path.write_text(json.dumps(
        {"threshold": old.get("threshold", 0.5), "threshold_tuned": False,
         "features": features, "trained_on": "dense", "rounds": n_rounds,
         "neg_frac": args.neg_frac}, indent=2))
    print(f"Saved model to {work / 'model_lgb.txt'}  ({time.time() - t0:.0f}s)")
    print("\nNext: python src/predict.py --split train --overwrite   then   python src/tune.py")


if __name__ == "__main__":
    main()