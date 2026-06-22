from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments import v5_prospective_evidence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v5_prospective_smoke_generates_all_gated_artifacts(tmp_path: Path) -> None:
    summary = v5_prospective_evidence.run(smoke=True, output_root=tmp_path)

    assert summary["gate_passed"] is True
    assert summary["low_ram_design"]["parallel_jobs"] == 1
    assert summary["low_ram_design"]["materializes_optional_robotics_envs"] is False
    assert summary["prospective_audit"]["gate_passed"] is True
    assert summary["label_budget_sample_complexity"]["gate_passed"] is True
    assert summary["selector_gauntlet"]["gate_passed"] is True
    assert summary["equal_compute_frontier"]["gate_passed"] is True
    assert summary["closed_loop_validation"]["gate_passed"] is True

    out = tmp_path / "results" / "v5_smoke"
    for name in [
        "prospective_audit_predictions.csv",
        "prospective_audit_predictions.sha256",
        "prospective_audit_outcomes.csv",
        "prospective_audit_summary.json",
        "label_budget_sample_complexity.csv",
        "selector_gauntlet.csv",
        "equal_compute_frontier.csv",
        "closed_loop_validation.csv",
        "prospective_evidence_summary.json",
    ]:
        assert (out / name).exists(), name


def test_prospective_predictions_are_hashed_into_outcomes(tmp_path: Path) -> None:
    v5_prospective_evidence.run(smoke=True, output_root=tmp_path)
    out = tmp_path / "results" / "v5_smoke"
    digest = (out / "prospective_audit_predictions.sha256").read_text(encoding="utf-8").strip()
    assert digest == v5_prospective_evidence.sha256(out / "prospective_audit_predictions.csv")

    outcomes = read_csv(out / "prospective_audit_outcomes.csv")
    assert outcomes
    assert {row["prediction_sha256"] for row in outcomes} == {digest}
    assert {"allow_high_n", "block_high_n", "stop_early", "request_labels"} & {
        row["decision"] for row in outcomes
    }


def test_label_budget_includes_zero_label_and_request_label_behavior(tmp_path: Path) -> None:
    v5_prospective_evidence.run(smoke=True, output_root=tmp_path)
    payload = load_json(tmp_path / "results" / "v5_smoke" / "label_budget_sample_complexity_summary.json")

    assert payload["zero_label_included"] is True
    zero = [row for row in payload["budget_summaries"] if row["label_budget"] == 0][0]
    assert "request_labels" in zero["decision_counts"]


def test_selector_gauntlet_labels_oracle_as_non_deployable(tmp_path: Path) -> None:
    v5_prospective_evidence.run(smoke=True, output_root=tmp_path)
    rows = read_csv(tmp_path / "results" / "v5_smoke" / "selector_gauntlet.csv")

    oracle_rows = [row for row in rows if row["selector"] == "oracle_real_utility"]
    assert oracle_rows
    assert {row["deployable"] for row in oracle_rows} == {"False"}
    assert {row["oracle_row"] for row in oracle_rows} == {"True"}


def test_equal_compute_frontier_reports_cpu_units_and_oracle_regret(tmp_path: Path) -> None:
    v5_prospective_evidence.run(smoke=True, output_root=tmp_path)
    rows = read_csv(tmp_path / "results" / "v5_smoke" / "equal_compute_frontier.csv")

    assert rows
    assert all(int(row["cpu_units"]) >= int(row["rollouts_used"]) for row in rows)
    assert "oracle_upper_bound" in {row["strategy"] for row in rows}


def test_closed_loop_validation_has_required_policies(tmp_path: Path) -> None:
    v5_prospective_evidence.run(smoke=True, output_root=tmp_path)
    rows = read_csv(tmp_path / "results" / "v5_smoke" / "closed_loop_validation.csv")

    policies = {row["policy"] for row in rows}
    assert {"n1", "raw_high_n", "random_high_n", "audit_policy", "oracle_upper_bound"}.issubset(policies)
    summary = load_json(tmp_path / "results" / "v5_smoke" / "closed_loop_validation_summary.json")
    assert summary["episodes"] > 0
    assert "audit_minus_raw_ci" in summary
