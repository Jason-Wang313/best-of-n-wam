from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


GENERATED_REPORTS = [
    "maxout_initial_audit.md",
    "maxout_completion_audit.md",
    "reviewer_risk_assessment.md",
    "ablation_report.md",
    "falsification_report.md",
    "claims_report.md",
    "paper_result_summary.md",
    "final_decision_report.md",
]


@dataclass(frozen=True)
class ReportGenerationCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ReportGenerationCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ReportGenerationCheck(name=name, ok=bool(ok), detail=detail))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def default_command(root: Path) -> list[str]:
    return [sys.executable, str(root / "scripts" / "write_maxout_reports.py")]


def audit_report_generation_consistency(
    root: Path,
    reports_dir: Path | None = None,
    *,
    report_names: Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    timeout_s: int = 180,
) -> dict[str, Any]:
    root = root.resolve()
    reports_dir = (reports_dir or root / "reports").resolve()
    names = list(report_names or GENERATED_REPORTS)
    paths = [reports_dir / name for name in names]
    before = {name: read_bytes(path) for name, path in zip(names, paths)}

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    env["WAM_REPORTS_DIR"] = str(reports_dir)
    cmd = list(command) if command is not None else default_command(root)
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )

    after = {name: read_bytes(path) for name, path in zip(names, paths)}
    checks: list[ReportGenerationCheck] = []
    add(checks, "generator_exit_zero", proc.returncode == 0, f"returncode={proc.returncode}")
    add(checks, "reports_list_nonempty", len(names) > 0, f"reports={len(names)}")

    missing = [name for name, path in zip(names, paths) if not path.exists()]
    empty = [name for name, data in after.items() if not data]
    changed = [name for name in names if before[name] != after[name]]
    unresolved = [
        name
        for name, data in after.items()
        if b"{fmt(" in data or b"{claims_payload" in data or b"`missing`" in data
    ]

    add(checks, "generated_reports_exist", not missing, f"missing={missing}")
    add(checks, "generated_reports_nonempty", not empty, f"empty={empty}")
    add(checks, "generated_reports_byte_stable", not changed, f"changed={changed}")
    add(checks, "stdout_confirms_generation", "max-out reports written" in proc.stdout, f"stdout_bytes={len(proc.stdout.encode('utf-8'))}")
    add(checks, "no_known_template_markers", not unresolved, f"unresolved={unresolved}")
    add(checks, "final_decision_has_command_results", b"## Command Results" in after.get("final_decision_report.md", b""), "final_decision_report.md has command results")
    add(checks, "claims_report_has_counts", b"- verified:" in after.get("claims_report.md", b""), "claims_report.md has status counts")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "report_generation_consistency",
        "verified": len(issues) == 0,
        "n_reports": len(names),
        "n_files_checked": len(names),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "report_sha256": {name: sha256_bytes(after[name]) for name in names},
        "changed_reports": changed,
        "generator_returncode": proc.returncode,
        "command": cmd,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def report_generation_consistency_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Report Generation Consistency Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Reports checked: {payload.get('n_reports')}",
        f"- Files checked: {payload.get('n_files_checked')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Rerunning `scripts/write_maxout_reports.py` leaves all canonical generated narrative reports byte-stable.")
    lines.append("")
    return "\n".join(lines)
