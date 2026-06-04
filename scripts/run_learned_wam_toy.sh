#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

if command -v python.exe >/dev/null 2>&1; then
  PY=(python.exe)
elif command -v python >/dev/null 2>&1 && python -m pytest --version >/dev/null 2>&1; then
  PY=(python)
elif command -v py >/dev/null 2>&1; then
  PY=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  echo "No Python interpreter found on PATH" >&2
  exit 127
fi

MODEL_PATH="results/models/learned_wam_lite_toy.npz"
SEEDS=(701 702 703 704 705)

"${PY[@]}" experiments/train_learned_wam_lite.py \
  --model-path "$MODEL_PATH" \
  --seed 601 \
  --id-mismatch mild \
  --train-states 32 \
  --train-rollouts 64 \
  --val-states 12 \
  --val-rollouts 64 \
  --ood-states 6 \
  --ood-rollouts 32 \
  --max-horizon 12

"${PY[@]}" experiments/exact_rollout_law_validation.py \
  --states 5 \
  --rollouts 48 \
  --mc-trials 600 \
  --mismatch mild \
  --scorer predicted_utility \
  --dynamics-backend learned \
  --model-path "$MODEL_PATH" \
  --seeds "${SEEDS[@]}"

"${PY[@]}" experiments/score_function_comparison.py \
  --states 5 \
  --rollouts 48 \
  --mismatch mild \
  --dynamics-backend learned \
  --model-path "$MODEL_PATH" \
  --seeds "${SEEDS[@]}"

"${PY[@]}" experiments/real_vs_imagined_utility_gap.py \
  --states 5 \
  --rollouts 48 \
  --scorer predicted_utility \
  --dynamics-backend learned \
  --model-path "$MODEL_PATH" \
  --seeds "${SEEDS[@]}"

"${PY[@]}" experiments/adaptive_rollout_allocation.py \
  --states 6 \
  --rollouts 96 \
  --pilot-k 24 \
  --max-n 32 \
  --mean-budgets 1 2 4 8 16 32 \
  --mismatch severe \
  --scorer predicted_utility \
  --dynamics-backend learned \
  --model-path "$MODEL_PATH" \
  --seeds "${SEEDS[@]}"

"${PY[@]}" experiments/closed_loop_receding_horizon_eval.py \
  --episodes 12 \
  --mismatch mild \
  --dynamics-backend learned \
  --model-path "$MODEL_PATH" \
  --seeds "${SEEDS[@]}"

"${PY[@]}" experiments/learned_wam_vs_analytic_wam.py \
  --states 12 \
  --rollouts 80 \
  --mismatch mild \
  --scorer predicted_utility \
  --model-path "$MODEL_PATH" \
  --seeds "${SEEDS[@]}"

"${PY[@]}" scripts/artifact_integrity.py --fail-on-error
"${PY[@]}" scripts/result_consistency.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
