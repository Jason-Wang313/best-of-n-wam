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
bash scripts/run_smoke.sh
bash scripts/run_learned_wam_toy.sh
bash scripts/run_multi_env.sh
"${PY[@]}" experiments/nonstationary_dynamics_extension.py --episodes 48 --rollouts 160 --mc-trials 1000 --seed 41
bash scripts/run_benchmark_full.sh
bash scripts/run_visual_optional.sh
bash scripts/run_inference_audit.sh
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
"${PY[@]}" scripts/write_maxout_reports.py
"${PY[@]}" scripts/narrative_consistency.py --fail-on-error
"${PY[@]}" scripts/script_contracts.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_semantics.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
"${PY[@]}" scripts/tracked_artifact_provenance.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/report_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/command_result_consistency.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_semantics.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
"${PY[@]}" scripts/tracked_artifact_provenance.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/report_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/command_result_consistency.py --fail-on-error
"${PY[@]}" scripts/write_maxout_reports.py
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
"${PY[@]}" scripts/write_maxout_reports.py
"${PY[@]}" scripts/narrative_consistency.py --fail-on-error
"${PY[@]}" scripts/script_contracts.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_semantics.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
"${PY[@]}" scripts/tracked_artifact_provenance.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/report_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/command_result_consistency.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_semantics.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_evidence_quality.py --fail-on-error
"${PY[@]}" scripts/tracked_artifact_provenance.py --fail-on-error
"${PY[@]}" scripts/claims_status.py
"${PY[@]}" scripts/claim_ledger_integrity.py --fail-on-error
"${PY[@]}" scripts/claim_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/report_generation_consistency.py --fail-on-error
"${PY[@]}" scripts/command_result_consistency.py --fail-on-error
