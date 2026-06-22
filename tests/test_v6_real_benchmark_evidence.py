from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments import v6_real_benchmark_evidence


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v6_real_benchmark_smoke_is_gated_and_low_ram(tmp_path: Path) -> None:
    summary = v6_real_benchmark_evidence.run(output_root=tmp_path, source_root=ROOT, smoke=True)

    assert summary["gate_passed"] is True
    assert summary["low_ram_design"]["uses_existing_curve_csvs"] is True
    assert summary["low_ram_design"]["reruns_simulators"] is False
    assert summary["low_ram_design"]["stores_candidate_tensors"] is False
    assert summary["real_benchmark_audit"]["family_count"] >= 4
    assert summary["real_benchmark_audit"]["pool_count"] >= 20
    assert summary["real_benchmark_audit"]["decision_accuracy"] >= 0.8
    assert summary["real_benchmark_audit"]["false_allow_rate"] <= 0.02

    out = tmp_path / "results" / "v6_smoke"
    for name in [
        "split_manifest.json",
        "real_benchmark_audit_predictions.csv",
        "real_benchmark_audit_predictions.sha256",
        "real_benchmark_audit_outcomes.csv",
        "cross_family_transfer.csv",
        "selector_metric_ablation.csv",
        "robustness_grid.csv",
        "finite_sample_audit_theory.csv",
        "summary.json",
    ]:
        assert (out / name).exists(), name


def test_v6_predictions_are_hash_locked_into_outcomes(tmp_path: Path) -> None:
    v6_real_benchmark_evidence.run(output_root=tmp_path, source_root=ROOT, smoke=True)
    out = tmp_path / "results" / "v6_smoke"
    digest = (out / "real_benchmark_audit_predictions.sha256").read_text(encoding="utf-8").strip()
    assert digest == v6_real_benchmark_evidence.sha256(out / "real_benchmark_audit_predictions.csv")

    outcomes = read_csv(out / "real_benchmark_audit_outcomes.csv")
    assert outcomes
    assert {row["prediction_sha256"] for row in outcomes} == {digest}
    assert {"allow_high_n", "request_labels", "stop_early"} & {row["decision"] for row in outcomes}


def test_v6_ablation_reports_oracle_as_non_deployable(tmp_path: Path) -> None:
    v6_real_benchmark_evidence.run(output_root=tmp_path, source_root=ROOT, smoke=True)
    rows = read_csv(tmp_path / "results" / "v6_smoke" / "selector_metric_ablation.csv")

    oracle = [row for row in rows if row["strategy"] == "oracle_upper_bound"]
    audit = [row for row in rows if row["strategy"] == "audit_with_abstention"]
    assert oracle and audit
    assert oracle[0]["deployable"] == "False"
    assert audit[0]["deployable"] == "True"
    assert float(audit[0]["mean_rollout_units"]) < float([row for row in rows if row["strategy"] == "raw_high_n"][0]["mean_rollout_units"])


def test_v6_finite_sample_theory_has_conservative_bound(tmp_path: Path) -> None:
    v6_real_benchmark_evidence.run(output_root=tmp_path, source_root=ROOT, smoke=True)
    payload = load_json(tmp_path / "results" / "v6_smoke" / "summary.json")

    assert payload["finite_sample_theory"]["epsilon_0_05_delta_0_05_labels"] > 1000
    assert "abstention" in payload["finite_sample_theory"]["interpretation"]
