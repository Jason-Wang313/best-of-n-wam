from __future__ import annotations

import json
from pathlib import Path

from experiments import v6_frozen_evidence


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v6_frozen_evidence_smoke_writes_paper_inputs(tmp_path: Path) -> None:
    summary = v6_frozen_evidence.run(smoke=True, output_root=tmp_path, source_root=ROOT)

    assert summary["gate_passed"] is True
    assert summary["family_count"] >= 4
    assert summary["pool_count"] >= 20
    assert summary["decision_accuracy"] >= 0.8
    assert summary["false_allow_rate"] <= 0.02

    macros = tmp_path / "v6_results_macros.tex"
    assert macros.exists()
    macro_text = macros.read_text(encoding="utf-8")
    for name in [
        "VSixWAMFamilies",
        "VSixWAMPools",
        "VSixWAMDecisionAccuracy",
        "VSixWAMPredictionHash",
        "VSixWAMFiniteLabelsEpsFiveDeltaFive",
    ]:
        assert f"\\newcommand{{\\{name}}}" in macro_text

    assert (tmp_path / "paper" / "v6_summary_table.tex").exists()
    assert (tmp_path / "paper" / "v6_ablation_table.tex").exists()
    table_text = (tmp_path / "paper" / "v6_summary_table.tex").read_text(encoding="utf-8")
    assert "Existing simulated rollout-pool curves only" in table_text
    assert "no-real-robot" in table_text
    assert "broad-SOTA claim" in table_text

    for figure_name in [
        "v6_cross_family_accuracy.pdf",
        "v6_cross_family_decided_rate.pdf",
        "v6_compute_ablation.pdf",
        "v6_robustness_grid.pdf",
    ]:
        figure = tmp_path / "paper_figures" / "v6" / figure_name
        assert figure.exists(), figure_name
        assert figure.stat().st_size > 1000

    frozen_summary = load_json(tmp_path / "results" / "v6_frozen_evidence_smoke" / "summary.json")
    assert frozen_summary["gate_passed"] is True
