import sys
from pathlib import Path

from wam_inference_value.report_generation_consistency import audit_report_generation_consistency


REPORTS = ["claims_report.md", "final_decision_report.md"]


def write_reports(reports_dir: Path, *, suffix: str = "stable") -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "claims_report.md").write_text(
        f"# Claims Report\n\n- verified: `1`\n- marker: {suffix}\n",
        encoding="utf-8",
    )
    (reports_dir / "final_decision_report.md").write_text(
        f"# Final Decision Report\n\n## Command Results\n\n- marker: {suffix}\n",
        encoding="utf-8",
    )


def write_generator(path: Path, *, mutate: bool) -> None:
    suffix = "mutated" if mutate else "stable"
    path.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "reports = Path(os.environ['WAM_REPORTS_DIR'])\n"
        "reports.mkdir(parents=True, exist_ok=True)\n"
        f"(reports / 'claims_report.md').write_text('# Claims Report\\n\\n- verified: `1`\\n- marker: {suffix}\\n', encoding='utf-8')\n"
        f"(reports / 'final_decision_report.md').write_text('# Final Decision Report\\n\\n## Command Results\\n\\n- marker: {suffix}\\n', encoding='utf-8')\n"
        "print('max-out reports written')\n",
        encoding="utf-8",
    )


def test_report_generation_consistency_accepts_byte_stable_generator(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    generator = tmp_path / "stable_report_generator.py"
    write_reports(reports_dir)
    write_generator(generator, mutate=False)

    payload = audit_report_generation_consistency(
        tmp_path,
        reports_dir,
        report_names=REPORTS,
        command=[sys.executable, str(generator)],
    )

    assert payload["verified"] is True
    assert payload["n_issues"] == 0


def test_report_generation_consistency_rejects_mutating_generator(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    generator = tmp_path / "mutating_report_generator.py"
    write_reports(reports_dir)
    write_generator(generator, mutate=True)

    payload = audit_report_generation_consistency(
        tmp_path,
        reports_dir,
        report_names=REPORTS,
        command=[sys.executable, str(generator)],
    )

    failures = {check["name"] for check in payload["issues"]}
    assert "generated_reports_byte_stable" in failures
    assert payload["changed_reports"] == REPORTS
