#!/usr/bin/env bash
set -euo pipefail

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

WSL_PYTHON_EXE=0
if [[ "${PY[0]}" == "python.exe" ]] && grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
  WSL_PYTHON_EXE=1
fi

if [[ "$WSL_PYTHON_EXE" == "1" ]]; then
  SOURCE_DIR="$(pwd)/src"
  CANONICAL_RESULTS_DIR="$(pwd)/results"
  DEFAULT_SMOKE_RESULTS_DIR="$(pwd)/results/smoke"
  export WSLENV="${WSLENV:+$WSLENV:}WAM_RESULTS_DIR/p:PYTHONPATH/lp"
else
  SOURCE_DIR="$("${PY[@]}" - <<'PY'
from pathlib import Path
print(Path("src").resolve())
PY
)"
  SOURCE_DIR="${SOURCE_DIR%$'\r'}"
  CANONICAL_RESULTS_DIR="$("${PY[@]}" - <<'PY'
from pathlib import Path
print(Path("results").resolve())
PY
)"
  CANONICAL_RESULTS_DIR="${CANONICAL_RESULTS_DIR%$'\r'}"
  DEFAULT_SMOKE_RESULTS_DIR="$("${PY[@]}" - <<'PY'
from pathlib import Path
print(Path("results/smoke").resolve())
PY
)"
  DEFAULT_SMOKE_RESULTS_DIR="${DEFAULT_SMOKE_RESULTS_DIR%$'\r'}"
fi
export PYTHONPATH="${PYTHONPATH:-}:$SOURCE_DIR"
export WAM_RESULTS_DIR="${WAM_SMOKE_RESULTS_DIR:-$DEFAULT_SMOKE_RESULTS_DIR}"
if [[ "$WSL_PYTHON_EXE" == "1" ]] && command -v wslpath >/dev/null 2>&1; then
  SMOKE_MODEL_PATH="$(wslpath -w "$WAM_RESULTS_DIR")\\models\\learned_wam_lite_smoke.npz"
else
  SMOKE_MODEL_PATH="$WAM_RESULTS_DIR/models/learned_wam_lite_smoke.npz"
fi

"${PY[@]}" -m pytest -q
"${PY[@]}" experiments/train_learned_wam_lite.py --model-path "$SMOKE_MODEL_PATH" --seed 301 --id-mismatch mild --train-states 4 --train-rollouts 16 --val-states 2 --val-rollouts 16 --ood-states 2 --ood-rollouts 12 --max-horizon 12
"${PY[@]}" experiments/learned_wam_vs_analytic_wam.py --states 2 --rollouts 24 --model-path "$SMOKE_MODEL_PATH" --seeds 301 302 303 304 305
"${PY[@]}" experiments/exact_rollout_law_validation.py --states 8 --rollouts 64 --mc-trials 1500 --seed 101
"${PY[@]}" experiments/auc_vs_moment_hierarchy.py --states 8 --rollouts 64 --seed 102
"${PY[@]}" experiments/pilot_to_heldout_prediction.py --states 8 --rollouts 160 --splits 2 --seed 103
"${PY[@]}" experiments/score_function_comparison.py --states 8 --rollouts 64 --seed 104
"${PY[@]}" experiments/real_vs_imagined_utility_gap.py --states 8 --rollouts 64 --seed 105
"${PY[@]}" experiments/adaptive_rollout_allocation.py --states 12 --rollouts 160 --pilot-k 32 --max-n 32 --mean-budgets 1 2 4 8 16 32 --seed 106
"${PY[@]}" experiments/closed_loop_receding_horizon_eval.py --episodes 4 --seed 107
"${PY[@]}" experiments/nonstationary_dynamics_extension.py --episodes 4 --rollouts 64 --mc-trials 1200 --seed 108
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/test_inventory.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/artifact_integrity.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/result_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/raw_result_recompute.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/table_schema.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/source_manifest.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/runtime_environment.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/experiment_registry.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/artifact_manifest.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/model_artifact_integrity.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/figure_quality.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/write_maxout_reports.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/narrative_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/script_contracts.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claims_status.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_semantics.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claims_status.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/tracked_artifact_provenance.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claims_status.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/report_generation_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/command_result_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claims_status.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_semantics.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claims_status.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/tracked_artifact_provenance.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claims_status.py
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/report_generation_consistency.py --fail-on-error
WAM_RESULTS_DIR="$CANONICAL_RESULTS_DIR" "${PY[@]}" scripts/command_result_consistency.py --fail-on-error
