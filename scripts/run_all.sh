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
"${PY[@]}" experiments/exact_rollout_law_validation.py --states 96 --rollouts 256 --mc-trials 12000 --seed 7
"${PY[@]}" experiments/auc_vs_moment_hierarchy.py --states 96 --rollouts 256 --seed 11
"${PY[@]}" experiments/pilot_to_heldout_prediction.py --states 120 --rollouts 384 --splits 10 --seed 17
"${PY[@]}" experiments/score_function_comparison.py --states 120 --rollouts 256 --seed 23
"${PY[@]}" experiments/real_vs_imagined_utility_gap.py --states 120 --rollouts 256 --seed 29
"${PY[@]}" experiments/adaptive_rollout_allocation.py --states 144 --rollouts 384 --pilot-k 64 --max-n 64 --mean-budgets 1 2 4 8 16 32 64 --seed 31
"${PY[@]}" experiments/closed_loop_receding_horizon_eval.py --episodes 72 --seed 37
"${PY[@]}" experiments/nonstationary_dynamics_extension.py --episodes 48 --rollouts 160 --mc-trials 5000 --seed 41
"${PY[@]}" scripts/artifact_integrity.py --fail-on-error
"${PY[@]}" scripts/result_consistency.py --fail-on-error
"${PY[@]}" scripts/narrative_consistency.py --fail-on-error
"${PY[@]}" scripts/script_contracts.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
