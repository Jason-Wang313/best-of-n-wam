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

"${PY[@]}" -m pytest -q
"${PY[@]}" experiments/train_learned_wam_lite.py --model-path results/models/learned_wam_lite_smoke.npz --seed 301 --id-mismatch mild --train-states 4 --train-rollouts 16 --val-states 2 --val-rollouts 16 --ood-states 2 --ood-rollouts 12 --max-horizon 12
"${PY[@]}" experiments/learned_wam_vs_analytic_wam.py --states 2 --rollouts 24 --model-path results/models/learned_wam_lite_smoke.npz --seeds 301 302 303 304 305
"${PY[@]}" experiments/exact_rollout_law_validation.py --states 8 --rollouts 64 --mc-trials 1500 --seed 101
"${PY[@]}" experiments/auc_vs_moment_hierarchy.py --states 8 --rollouts 64 --seed 102
"${PY[@]}" experiments/pilot_to_heldout_prediction.py --states 8 --rollouts 160 --splits 2 --seed 103
"${PY[@]}" experiments/score_function_comparison.py --states 8 --rollouts 64 --seed 104
"${PY[@]}" experiments/real_vs_imagined_utility_gap.py --states 8 --rollouts 64 --seed 105
"${PY[@]}" experiments/adaptive_rollout_allocation.py --states 12 --rollouts 160 --pilot-k 32 --max-n 32 --mean-budgets 1 2 4 8 16 32 --seed 106
"${PY[@]}" experiments/closed_loop_receding_horizon_eval.py --episodes 4 --seed 107
"${PY[@]}" experiments/nonstationary_dynamics_extension.py --episodes 4 --rollouts 64 --mc-trials 1200 --seed 108
"${PY[@]}" scripts/claims_status.py
