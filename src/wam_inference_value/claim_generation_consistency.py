from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ClaimGenerationCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ClaimGenerationCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ClaimGenerationCheck(name=name, ok=bool(ok), detail=detail))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def parse_json_bytes(data: bytes) -> dict[str, Any]:
    if not data:
        return {}
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def default_command(root: Path) -> list[str]:
    return [sys.executable, str(root / "scripts" / "claims_status.py")]


def audit_claim_generation_consistency(
    root: Path,
    results_dir: Path | None = None,
    *,
    command: Sequence[str] | None = None,
    timeout_s: int = 180,
) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    json_path = results_dir / "claims_status.json"
    md_path = results_dir / "claims_status.md"
    before_json = read_bytes(json_path)
    before_md = read_bytes(md_path)

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    env["WAM_RESULTS_DIR"] = str(results_dir)
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

    after_json = read_bytes(json_path)
    after_md = read_bytes(md_path)
    payload = parse_json_bytes(after_json)
    claims = payload.get("claims") if isinstance(payload.get("claims"), list) else []
    checks: list[ClaimGenerationCheck] = []

    add(checks, "generator_exit_zero", proc.returncode == 0, f"returncode={proc.returncode}")
    add(checks, "claims_json_exists", json_path.exists() and len(after_json) > 0, f"bytes={len(after_json)}")
    add(checks, "claims_md_exists", md_path.exists() and len(after_md) > 0, f"bytes={len(after_md)}")
    add(checks, "claims_json_byte_stable", before_json == after_json, f"before={sha256_bytes(before_json)}, after={sha256_bytes(after_json)}")
    add(checks, "claims_md_byte_stable", before_md == after_md, f"before={sha256_bytes(before_md)}, after={sha256_bytes(after_md)}")
    add(checks, "claims_json_parses", bool(payload), f"keys={sorted(payload)[:8]}")
    add(checks, "claims_all_verified", int(payload.get("num_verified") or 0) == len(claims) and len(claims) > 0, f"verified={payload.get('num_verified')}, claims={len(claims)}")
    add(checks, "claims_no_overclaims", len(payload.get("overclaims") or []) == 0, f"overclaims={len(payload.get('overclaims') or [])}")
    add(checks, "stdout_contains_claims_status", "# Claims Status" in proc.stdout, f"stdout_bytes={len(proc.stdout.encode('utf-8'))}")
    markdown_claim_rows = sum(1 for line in after_md.decode("utf-8", errors="ignore").splitlines() if line.startswith("- Claim "))
    add(checks, "markdown_claim_count_matches_json", markdown_claim_rows == len(claims), f"md_claims={markdown_claim_rows}, json_claims={len(claims)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "claim_generation_consistency",
        "verified": len(issues) == 0,
        "n_files_checked": 2,
        "n_claims": len(claims),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "json_sha256": sha256_bytes(after_json),
        "md_sha256": sha256_bytes(after_md),
        "generator_returncode": proc.returncode,
        "command": cmd,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
    }


def claim_generation_consistency_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Generation Consistency Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Files checked: {payload.get('n_files_checked')}",
        f"- Claims: {payload.get('n_claims')}",
        f"- Checks: {payload.get('n_checks')}",
        f"- Issues: {payload.get('n_issues')}",
        f"- claims_status.json SHA-256: `{payload.get('json_sha256')}`",
        f"- claims_status.md SHA-256: `{payload.get('md_sha256')}`",
        "",
    ]
    issues = payload.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- `{issue.get('name')}`: {issue.get('detail')}")
    else:
        lines.append("Rerunning `scripts/claims_status.py` leaves `results/claims_status.json` and `results/claims_status.md` byte-stable, with all claims verified and no overclaims.")
    lines.append("")
    return "\n".join(lines)
