# Business Entity Resolution: reproduction guide

Given business records from three sources, find every Source 2 / Source 3 record
that refers to the same real-world business as each Source 1 entity.

Pipeline: **normalize → learn script dictionary → block → features → LightGBM → decide**.
No external data, APIs or lookups are used; everything is learned from the provided
training files. The only model is LightGBM (MIT license), trained from scratch.

## 1. Environment

- Linux recommended (full runs used an AWS r6i.4xlarge: 16 vCPU, 128 GB RAM).
  Windows works too, but runs single-process and needs much more time and memory.
- Python 3.12 or newer.

```bash
sudo apt install -y libgomp1          # needed by LightGBM on Linux
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Data location

Point the pipeline at the folder that contains `train/` and `test/`, either by
editing `DATA_DIR` in `src/config.py`, or by passing it to `run_all.sh`.

## 3. Run everything

```bash
bash run_all.sh /path/to/student_resource/dataset
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Total time on the machine above is roughly 3 to 3.5 hours.

Check the files with the organizers' validator:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## 4. What each step does

| Step | Script | Output |
|---|---|---|
| Normalize names/addresses (country-agnostic rules) | `src/preprocess.py`, `src/normalize.py` | `work/train`, `work/test` (Parquet) |
| Learn Indian-script → English words from training matches | `src/script_dict.py` | `work/script_dict.json` |
| Full-density blocking on the training set (3 TF-IDF channels: name words, address tokens, name char 3-grams; top 10 each; within country; S2/S3 → S1). Training entities are split into fixed halves A/B by id hash; features are built for ~20% of half-A records | `src/build_dense.py` (uses `src/blocking.py`, `src/features.py`) | `work/train/blk_*.parquet`, `work/train/dense_*.parquet` |
| Train LightGBM (early stopping on 10% held-out records) | `src/train_dense.py` | `work/model_lgb.txt`, `work/model_config.json` |
| Score every training record with the model (reuses saved blocking) | `src/predict.py --split train` | `work/train/best_*.parquet` |
| Choose the threshold on half B (never trained on) by macro F0.5; report the realistic score | `src/tune.py` | threshold in `work/model_config.json` |
| Test set: blocking + features + model; one owner per S2/S3 record; threshold; write TSVs | `src/predict.py` | `output/*.tsv` |

Other files: `src/config.py` (data path). Development tools, not needed to
reproduce the submission: `src/data.py` (exploration), `src/train.py` (early
dev-sample model), `src/score.py` (score a prediction file against training labels).

## 5. Notes

- Country is treated as an open set of labels: every step loops over whatever labels
  appear (France included), and no feature uses the country value.
- Random seeds are fixed (seed 42; halves A/B use a CRC32 hash of the entity id).
- `predict.py` caches blocking and per-country results in `work/<split>/`, so an
  interrupted run resumes; `--overwrite` rescores, `--reblock` redoes blocking.
