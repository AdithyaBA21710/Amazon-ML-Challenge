#!/usr/bin/env bash
# Reproduce the full pipeline: raw TSV files -> output/matching_results.tsv
#                                              + output/candidate_pairs.tsv
# Usage:  bash run_all.sh /path/to/student_resource/dataset
#         (or set DATA_DIR in src/config.py and run: bash run_all.sh)
set -euo pipefail
cd "$(dirname "$0")"

ARGS=()
if [ $# -ge 1 ]; then ARGS=(--data-dir "$1"); fi

echo "== 1/8 Normalize training data (no script dictionary yet)"
rm -f work/script_dict.json
python src/preprocess.py "${ARGS[@]}" --splits train --overwrite

echo "== 2/8 Learn the Indian-script -> English word dictionary from training matches"
python src/script_dict.py "${ARGS[@]}"

echo "== 3/8 Normalize train + test again, now applying the dictionary"
python src/preprocess.py "${ARGS[@]}" --overwrite

echo "== 4/8 Full-density blocking on train; features for sampled half-A records"
python src/build_dense.py "${ARGS[@]}" --overwrite --reblock

echo "== 5/8 Train LightGBM on the full-density training features"
python src/train_dense.py --rounds 1500

echo "== 6/8 Score every training record with the new model (reuses blocking)"
python src/predict.py --split train --overwrite

echo "== 7/8 Tune the threshold on half B (never trained on); realistic score"
python src/tune.py "${ARGS[@]}"

echo "== 8/8 Test set: blocking + features + model + threshold -> submission files"
python src/predict.py --overwrite --reblock

echo "Done. Files are in output/"
