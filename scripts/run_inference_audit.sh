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

"${PY[@]}" experiments/inference_audit_framework.py \
  --states 24 \
  --rollouts 160 \
  --seeds 701 702 703 704 705 \
  --mismatches mild severe stuck_slip nonstationary

"${PY[@]}" experiments/inference_audit_framework.py \
  --dynamics-backend learned \
  --model-path results/models/learned_wam_lite_toy.npz \
  --train-if-missing \
  --states 16 \
  --rollouts 128 \
  --seeds 751 752 753 754 755 \
  --mismatches mild severe stuck_slip nonstationary

"${PY[@]}" experiments/scorer_repair_experiment.py \
  --states 24 \
  --rollouts 192 \
  --pilot-k 48 \
  --seeds 811 812 813 814 815 \
  --mismatches severe stuck_slip nonstationary

"${PY[@]}" experiments/imagination_scaling_law.py \
  --states 8 \
  --seeds 911 912 913 914 915 \
  --mismatches mild severe \
  --horizons 4 8 12 \
  --pool-sizes 32 64 128

"${PY[@]}" scripts/test_inventory.py --fail-on-error
"${PY[@]}" scripts/artifact_integrity.py --fail-on-error
"${PY[@]}" scripts/result_consistency.py --fail-on-error
"${PY[@]}" scripts/raw_result_recompute.py --fail-on-error
"${PY[@]}" scripts/table_schema.py --fail-on-error
"${PY[@]}" scripts/source_manifest.py --fail-on-error
"${PY[@]}" scripts/runtime_environment.py --fail-on-error
"${PY[@]}" scripts/experiment_registry.py --fail-on-error
"${PY[@]}" scripts/artifact_manifest.py --fail-on-error
"${PY[@]}" scripts/model_artifact_integrity.py --fail-on-error
"${PY[@]}" scripts/figure_quality.py --fail-on-error
"${PY[@]}" scripts/narrative_consistency.py --fail-on-error
"${PY[@]}" scripts/script_contracts.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_semantics.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/command_result_consistency.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_semantics.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/command_result_consistency.py --fail-on-error
