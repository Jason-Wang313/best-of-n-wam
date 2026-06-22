from __future__ import annotations

import json
from pathlib import Path

from experiments import v5_frozen_evidence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_frozen_evidence_smoke_writes_paper_inputs(tmp_path: Path) -> None:
    summary = v5_frozen_evidence.run(smoke=True, output_root=tmp_path)

    assert summary["gate_passed"] is True
    assert summary["prospective_pools"] > 0
    assert summary["census_rows"] > 0
    assert summary["closed_loop_audit_return"] > summary["closed_loop_raw_return"]

    macros = tmp_path / "v5_results_macros.tex"
    assert macros.exists()
    macro_text = macros.read_text(encoding="utf-8")
    for name in [
        "VFiveWAMExactHardeningCases",
        "VFiveWAMCensusRows",
        "VFiveWAMProspectivePools",
        "VFiveWAMPredictionHash",
        "VFiveWAMClosedLoopAuditReturn",
    ]:
        assert f"\\newcommand{{\\{name}}}" in macro_text

    summary_table = tmp_path / "paper" / "v5_summary_table.tex"
    compute_table = tmp_path / "paper" / "v5_compute_frontier_table.tex"
    assert summary_table.exists()
    assert compute_table.exists()

    table_text = summary_table.read_text(encoding="utf-8")
    assert "CPU enumeration gives complete coverage" in table_text
    assert "not robot success" in table_text
    assert "scale theater" not in table_text
    for unsupported in ["validated on real robots", "achieves SOTA", "full RoboCasa-wide validation"]:
        assert unsupported not in table_text

    for figure_name in [
        "v5_census_regimes.pdf",
        "v5_closed_loop_returns.pdf",
        "v5_equal_compute_frontier.pdf",
        "v5_label_budget.pdf",
    ]:
        figure = tmp_path / "paper_figures" / "v5" / figure_name
        assert figure.exists(), figure_name
        assert figure.stat().st_size > 1000

    frozen_summary = load_json(tmp_path / "results" / "v5_frozen_evidence_smoke" / "summary.json")
    assert frozen_summary["gate_passed"] is True
    assert frozen_summary["prospective_prediction_hash"][:10] in macro_text
