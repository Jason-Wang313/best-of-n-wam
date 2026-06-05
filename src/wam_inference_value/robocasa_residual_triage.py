from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluation import results_dir, write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _category(env_id: str) -> str:
    name = env_id.split("/", 1)[-1]
    if name.startswith("PickPlace"):
        return "pick_place"
    if name.startswith("Open"):
        return "open"
    if name.startswith("Close"):
        return "close"
    if name.startswith("Turn"):
        return "turn"
    if name.startswith("Move"):
        return "move"
    if name.startswith("Manipulate"):
        return "manipulate"
    if any(token in name for token in ("Clean", "Clear", "Wash", "Rinse", "Dry")):
        return "cleaning"
    if any(token in name for token in ("Cook", "Heat", "Boil", "Bake", "Oven", "Microwave", "Toast")):
        return "cooking"
    return "long_horizon_or_compositional"


def _error_class(error: str) -> str:
    text = str(error or "")
    if not text:
        return ""
    if "missing 1 required positional argument" in text or "unexpected keyword argument" in text:
        return "constructor_signature_failure"
    if "NotImplementedError" in text:
        return "not_implemented"
    if "ValueError" in text:
        return "value_error"
    if "Timeout" in text or "timed out" in text.lower():
        return "timeout"
    return "other_error"


def _micro_attempt_rows(root_results: Path) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    tables_dir = root_results / "tables"
    for path in sorted(tables_dir.glob("benchmark_robocasa_micro_rollout*.csv")):
        for row in _read_csv_rows(path):
            env_id = str(row.get("env_id") or "")
            if not env_id:
                continue
            attempts.setdefault(env_id, []).append(
                {
                    "source": path.name,
                    "reset_ok": _parse_bool(row.get("reset_ok")),
                    "rollout_ok": _parse_bool(row.get("rollout_ok")),
                    "nondegenerate": _parse_bool(row.get("nondegenerate")),
                    "seconds": row.get("seconds"),
                    "error": str(row.get("error") or ""),
                    "error_class": _error_class(str(row.get("error") or "")),
                }
            )
    for path in sorted(root_results.glob("benchmark_robocasa_micro_rollout_*.json")):
        payload = _load_json(path)
        if not payload.get("timed_out"):
            continue
        for env_id in payload.get("env_ids") or []:
            attempts.setdefault(str(env_id), []).append(
                {
                    "source": path.name,
                    "reset_ok": False,
                    "rollout_ok": False,
                    "nondegenerate": False,
                    "seconds": payload.get("wall_clock_seconds"),
                    "error": "timed out",
                    "error_class": "timeout",
                }
            )
    return attempts


def _status_for(env_id: str, rollout_pool: set[str], micro_covered: set[str], attempts: list[dict[str, Any]]) -> tuple[str, str]:
    if env_id in rollout_pool:
        return "rollout_pool_covered", "verified rollout-pool/smoke artifact"
    if env_id in micro_covered:
        return "micro_nondegenerate_covered", "verified micro-rollout nondegenerate artifact"
    if not attempts:
        return "unattempted", "no committed micro or rollout-pool artifact"
    if any(bool(row.get("nondegenerate")) for row in attempts):
        return "attempted_nondegenerate_not_catalogued", "nondegenerate attempt exists but catalog did not count it"
    classes = Counter(str(row.get("error_class") or "degenerate_or_failed") for row in attempts)
    if classes.get("timeout"):
        return "timed_out", f"timeout attempts={classes['timeout']}"
    if classes.get("constructor_signature_failure"):
        return "constructor_signature_failure", f"constructor signature failures={classes['constructor_signature_failure']}"
    if classes.get("not_implemented"):
        return "not_implemented", f"not implemented attempts={classes['not_implemented']}"
    if classes.get("value_error"):
        return "value_error", f"value errors={classes['value_error']}"
    return "degenerate_or_failed", f"attempts={len(attempts)}"


def build_robocasa_residual_triage(root: Path, root_results: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    root_results = (root_results or results_dir()).resolve()
    catalog = _load_json(root_results / "benchmark_robocasa_catalog_probe.json")
    registry_rows = _read_csv_rows(root_results / "tables" / "benchmark_robocasa_catalog_registry.csv")
    rollout_pool = {str(env_id) for env_id in catalog.get("verified_artifact_env_ids") or []}
    micro_covered = {str(env_id) for env_id in catalog.get("micro_rollout_env_ids") or []}
    any_covered = rollout_pool | micro_covered
    attempts_by_env = _micro_attempt_rows(root_results)

    rows: list[dict[str, Any]] = []
    for registry in registry_rows:
        env_id = str(registry.get("env_id") or "")
        if not env_id:
            continue
        attempts = attempts_by_env.get(env_id, [])
        status, reason = _status_for(env_id, rollout_pool, micro_covered, attempts)
        rows.append(
            {
                "env_id": env_id,
                "category": str(registry.get("category") or _category(env_id)),
                "coverage_status": status,
                "reason": reason,
                "attempt_count": len(attempts),
                "attempt_sources": ";".join(sorted({str(row.get("source")) for row in attempts if row.get("source")})),
                "covered_by_rollout_pool": env_id in rollout_pool,
                "covered_by_micro": env_id in micro_covered,
                "covered_by_any": env_id in any_covered,
            }
        )

    status_counts = Counter(str(row["coverage_status"]) for row in rows)
    category_status: dict[str, Counter[str]] = {}
    for row in rows:
        category_status.setdefault(str(row["category"]), Counter())[str(row["coverage_status"])] += 1
    unattempted_by_category = {
        category: int(counter.get("unattempted", 0))
        for category, counter in sorted(category_status.items())
        if counter.get("unattempted", 0)
    }
    table_path = root_results / "tables" / "benchmark_robocasa_residual_triage.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env_id",
        "category",
        "coverage_status",
        "reason",
        "attempt_count",
        "attempt_sources",
        "covered_by_rollout_pool",
        "covered_by_micro",
        "covered_by_any",
    ]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "benchmark_robocasa_residual_triage",
        "verified": bool(rows),
        "registry_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "unattempted_by_category": unattempted_by_category,
        "rollout_pool_covered": len(rollout_pool),
        "micro_nondegenerate_covered": len(micro_covered),
        "any_covered": len(any_covered),
        "attempted_not_covered": sum(
            int(row["coverage_status"] not in {"rollout_pool_covered", "micro_nondegenerate_covered", "unattempted"})
            for row in rows
        ),
        "unattempted": int(status_counts.get("unattempted", 0)),
        "table_path": str(table_path),
        "note": "RoboCasa residual triage only. Constructor failures, timeouts, and degenerate probes are not validation evidence.",
    }
    write_json(root_results / "benchmark_robocasa_residual_triage.json", summary)
    return summary


def robocasa_residual_triage_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RoboCasa Residual Triage",
        "",
        f"- verified audit: `{payload.get('verified')}`",
        f"- registry task IDs: `{payload.get('registry_count')}`",
        f"- rollout-pool covered: `{payload.get('rollout_pool_covered')}`",
        f"- micro nondegenerate covered: `{payload.get('micro_nondegenerate_covered')}`",
        f"- any covered: `{payload.get('any_covered')}`",
        f"- attempted but not covered: `{payload.get('attempted_not_covered')}`",
        f"- unattempted: `{payload.get('unattempted')}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in (payload.get("status_counts") or {}).items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Unattempted By Category", ""])
    for category, count in (payload.get("unattempted_by_category") or {}).items():
        lines.append(f"- `{category}`: `{count}`")
    lines.extend(
        [
            "",
            "This report is a triage artifact only. It does not promote failed, timed-out, or unattempted RoboCasa task IDs to benchmark evidence.",
        ]
    )
    return "\n".join(lines) + "\n"
