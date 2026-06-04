from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COLLECTED_RE = re.compile(r"(?P<count>\d+)\s+tests?\s+collected")


@dataclass(frozen=True)
class TestInventoryCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[TestInventoryCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(TestInventoryCheck(name=name, ok=bool(ok), detail=detail))


def parse_pytest_collect_stdout(stdout: str) -> dict[str, Any]:
    nodeids = [line.strip() for line in stdout.splitlines() if line.strip().startswith("tests/") and "::" in line]
    trailer_count = None
    for match in COLLECTED_RE.finditer(stdout):
        trailer_count = int(match.group("count"))
    return {
        "nodeids": nodeids,
        "n_nodeids": len(nodeids),
        "trailer_count": trailer_count,
    }


def audit_pytest_collection(*, stdout: str, returncode: int) -> dict[str, Any]:
    parsed = parse_pytest_collect_stdout(stdout)
    nodeids = parsed["nodeids"]
    unique_nodeids = sorted(set(nodeids))
    duplicates = sorted({nodeid for nodeid in nodeids if nodeids.count(nodeid) > 1})
    checks: list[TestInventoryCheck] = []

    add(checks, "pytest_collect_exit_zero", returncode == 0, f"returncode={returncode}")
    add(checks, "nodeids_present", len(nodeids) > 0, f"nodeids={len(nodeids)}")
    add(checks, "nodeids_unique", not duplicates, f"duplicates={duplicates[:10]}")
    add(checks, "minimum_test_count", len(nodeids) >= 90, f"nodeids={len(nodeids)}")
    if parsed["trailer_count"] is not None:
        add(checks, "trailer_count_matches_nodeids", parsed["trailer_count"] == len(nodeids), f"trailer={parsed['trailer_count']}, nodeids={len(nodeids)}")
    else:
        add(checks, "trailer_count_present", False, "pytest collect output did not include collected-count trailer")
    required_prefixes = [
        "tests/test_abstract_claim_support.py::",
        "tests/test_claim_reference_integrity.py::",
        "tests/test_claim_scope_audit.py::",
        "tests/test_command_result_consistency.py::",
        "tests/test_model_artifact_integrity.py::",
        "tests/test_publication_scope.py::",
        "tests/test_script_contracts.py::",
        "tests/test_theorem_binary.py::",
    ]
    missing_required = [prefix for prefix in required_prefixes if not any(nodeid.startswith(prefix) for nodeid in nodeids)]
    add(checks, "required_test_families_present", not missing_required, f"missing={missing_required}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "test_inventory",
        "verified": len(issues) == 0,
        "n_tests": len(nodeids),
        "n_unique_tests": len(unique_nodeids),
        "trailer_count": parsed["trailer_count"],
        "n_checks": len(checks),
        "n_issues": len(issues),
        "nodeids": nodeids,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def collect_pytest_inventory(root: Path, timeout_s: int = 180) -> dict[str, Any]:
    root = root.resolve()
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    payload = audit_pytest_collection(stdout=proc.stdout, returncode=proc.returncode)
    payload["command"] = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    return payload


def test_inventory_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Test Inventory Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Tests collected: {payload.get('n_tests')}",
        f"- Unique tests: {payload.get('n_unique_tests')}",
        f"- Trailer count: {payload.get('trailer_count')}",
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
        lines.append("Pytest collection succeeded, node IDs are unique, the collected-count trailer matches the node ID count, and required test families are present.")
    lines.append("")
    return "\n".join(lines)
