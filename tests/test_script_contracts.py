from pathlib import Path

from wam_inference_value.script_contracts import (
    CORE_GATE_SEQUENCE,
    audit_core_script,
    ordered_subsequence,
    ScriptContractCheck,
)


def test_ordered_subsequence_requires_order():
    text = "alpha beta gamma"

    assert ordered_subsequence(text, ["alpha", "gamma"])
    assert not ordered_subsequence(text, ["gamma", "alpha"])


def test_core_script_contract_flags_missing_gate(tmp_path: Path):
    script = tmp_path / "scripts" / "run_smoke.sh"
    script.parent.mkdir()
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python.exe --version\n"
        "python3 --version\n"
        "experiments/exact_rollout_law_validation.py\n"
        + "\n".join(CORE_GATE_SEQUENCE[:-1])
        + "\n",
        encoding="utf-8",
    )
    checks: list[ScriptContractCheck] = []

    audit_core_script(tmp_path, "scripts/run_smoke.sh", ["experiments/exact_rollout_law_validation.py"], checks)

    failures = {check.name for check in checks if not check.ok}
    assert "run_smoke_gate_sequence" in failures
    assert "run_smoke_double_claim_ledger" in failures
