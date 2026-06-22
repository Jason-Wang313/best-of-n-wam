from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments import v5_core_evidence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_core_evidence_smoke_generates_gated_artifacts(tmp_path: Path) -> None:
    summary = v5_core_evidence.run(tmp_path, smoke=True)

    assert summary["gate_passed"] is True
    assert summary["low_ram_design"]["parallel_jobs"] == 1
    assert summary["low_ram_design"]["streamed_census_csv"] is True
    assert summary["exact_law_hardening"]["all_passed"] is True
    assert summary["auc_correlation_insufficiency"]["gate_passed"] is True
    assert summary["finite_pool_census"]["gate_passed"] is True
    assert summary["impossibility_boundary"]["gate_passed"] is True

    out = tmp_path / "results" / "v5_smoke"
    assert (out / "exact_law_hardening.csv").exists()
    assert (out / "auc_correlation_insufficiency.csv").exists()
    assert (out / "finite_pool_census.csv").exists()
    assert (out / "impossibility_boundary.csv").exists()
    assert (out / "summary.json").exists()


def test_auc_correlation_counterexample_matches_common_metrics(tmp_path: Path) -> None:
    v5_core_evidence.run(tmp_path, smoke=True)
    payload = load_json(tmp_path / "results" / "v5_smoke" / "auc_correlation_insufficiency.json")

    diffs = payload["matched_metric_abs_differences"]
    assert max(diffs.values()) <= 1e-12
    assert payload["n2_gap"] <= 1e-12
    assert payload["high_n_gap"] >= 0.5


def test_finite_pool_census_row_count_is_exact(tmp_path: Path) -> None:
    summary = v5_core_evidence.run(tmp_path, smoke=True)
    census = summary["finite_pool_census"]

    assert census["row_count"] == census["expected_row_count"]
    assert census["counts_sum"] == census["expected_row_count"]
    assert {"helps", "harms"}.issubset(set(census["classification_counts"]))


def test_scoretailbench_manifest_reproduces_pool_hashes(tmp_path: Path) -> None:
    v5_core_evidence.run(tmp_path, smoke=True)
    root = tmp_path / "scoretailbench_v5_smoke"
    manifest = load_json(root / "manifest.json")

    assert manifest["claims_not_supported"]
    assert len(manifest["pools"]) == 4
    assert (root / "baselines" / "finite_selected_utility_curves.json").exists()
    for row in manifest["pools"]:
        pool_path = root / row["path"]
        assert pool_path.exists()
        assert v5_core_evidence.sha256(pool_path) == row["sha256"]
        with pool_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        assert len(records) == row["candidates"]
        assert {"score", "imagined_utility", "real_utility"}.issubset(records[0])


def test_impossibility_boundary_forces_refusal_case(tmp_path: Path) -> None:
    v5_core_evidence.run(tmp_path, smoke=True)
    payload = load_json(tmp_path / "results" / "v5_smoke" / "impossibility_boundary_summary.json")

    assert payload["observable_features_identical"] is True
    assert payload["score_only_selected_candidate_equal"] is True
    assert payload["selected_real_utility_gap"] >= 0.8
    assert payload["gate_passed"] is True
