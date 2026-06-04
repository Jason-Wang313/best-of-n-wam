from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODEL_SUFFIXES = {".npz", ".joblib"}


@dataclass(frozen=True)
class ModelArtifactCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[ModelArtifactCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ModelArtifactCheck(name=name, ok=bool(ok), detail=detail))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_array_stats(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    record = {"shape": list(arr.shape), "dtype": str(arr.dtype), "size": int(arr.size), "numeric": np.issubdtype(arr.dtype, np.number)}
    if record["numeric"]:
        finite = np.isfinite(arr)
        record["finite_count"] = int(finite.sum())
        record["all_finite"] = bool(finite.all())
        if arr.size:
            finite_values = arr[finite]
            record["min"] = float(finite_values.min()) if finite_values.size else None
            record["max"] = float(finite_values.max()) if finite_values.size else None
    return record


def inspect_npz_model(path: Path) -> dict[str, Any]:
    arrays: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=True) as payload:
        for name in payload.files:
            arrays.append({"name": name, **numeric_array_stats(payload[name])})
    return {
        "load_ok": True,
        "loader": "numpy.load",
        "n_arrays": len(arrays),
        "total_elements": int(sum(array.get("size", 0) for array in arrays)),
        "numeric_arrays": int(sum(1 for array in arrays if array.get("numeric"))),
        "all_finite": all(array.get("all_finite", True) for array in arrays),
        "arrays": arrays,
    }


def find_predictor_paths(obj: Any, *, prefix: str = "model", depth: int = 0) -> list[str]:
    if hasattr(obj, "predict"):
        return [prefix]
    if depth >= 3:
        return []
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_predictor_paths(value, prefix=f"{prefix}.{key}", depth=depth + 1))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(find_predictor_paths(value, prefix=f"{prefix}[{index}]", depth=depth + 1))
    return found


def inspect_joblib_model(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("joblib") is None:
        return {"load_ok": False, "loader": "joblib", "error": "joblib unavailable"}
    import joblib

    model = joblib.load(path)
    model_type = f"{type(model).__module__}.{type(model).__qualname__}"
    predictor_paths = find_predictor_paths(model)
    attrs = [name for name in ("predict", "estimators_", "n_features_in_", "classes_", "feature_importances_") if hasattr(model, name)]
    return {
        "load_ok": True,
        "loader": "joblib.load",
        "model_type": model_type,
        "attributes": attrs,
        "predictor_paths": predictor_paths,
        "has_predict": bool(predictor_paths),
    }


def inspect_model_file(root: Path, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "suffix": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        if path.suffix.lower() == ".npz":
            record.update(inspect_npz_model(path))
        elif path.suffix.lower() == ".joblib":
            record.update(inspect_joblib_model(path))
        else:
            record.update({"load_ok": False, "error": "unsupported suffix"})
    except Exception as exc:  # pragma: no cover - defensive, reported in JSON.
        record.update({"load_ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def scan_model_artifacts(root: Path, models_dir: Path | None = None) -> list[dict[str, Any]]:
    root = root.resolve()
    models_dir = (models_dir or root / "results" / "models").resolve()
    if not models_dir.exists():
        return []
    records = []
    for path in sorted(models_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES:
            records.append(inspect_model_file(root, path))
    return records


def audit_model_artifacts(root: Path, models_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    models_dir = (models_dir or root / "results" / "models").resolve()
    checks: list[ModelArtifactCheck] = []
    records = scan_model_artifacts(root, models_dir)
    by_suffix: dict[str, int] = {}
    for record in records:
        suffix = str(record.get("suffix"))
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
    unreadable = [record["path"] for record in records if not record.get("load_ok")]
    empty = [record["path"] for record in records if int(record.get("bytes") or 0) <= 0]
    npz_records = [record for record in records if record.get("suffix") == ".npz"]
    joblib_records = [record for record in records if record.get("suffix") == ".joblib"]
    npz_empty_arrays = [record["path"] for record in npz_records if int(record.get("n_arrays") or 0) <= 0]
    npz_nonfinite = [record["path"] for record in npz_records if not record.get("all_finite")]
    npz_tiny = [record["path"] for record in npz_records if int(record.get("total_elements") or 0) <= 0]
    joblib_without_predict = [record["path"] for record in joblib_records if not record.get("has_predict")]
    total_bytes = int(sum(int(record.get("bytes") or 0) for record in records))

    add(checks, "model_artifacts_present", len(records) >= 45, f"models={len(records)}")
    add(checks, "model_artifact_npz_breadth", len(npz_records) >= 30, f"npz={len(npz_records)}")
    add(checks, "model_artifact_joblib_breadth", len(joblib_records) >= 10, f"joblib={len(joblib_records)}")
    add(checks, "model_artifact_total_bytes", total_bytes >= 50_000_000, f"bytes={total_bytes}")
    add(checks, "model_artifacts_load", not unreadable, f"unreadable={unreadable[:10]}, count={len(unreadable)}")
    add(checks, "model_artifacts_nonempty_files", not empty, f"empty={empty}")
    add(checks, "npz_model_arrays_present", not npz_empty_arrays, f"empty_arrays={npz_empty_arrays[:10]}, count={len(npz_empty_arrays)}")
    add(checks, "npz_model_arrays_finite", not npz_nonfinite, f"nonfinite={npz_nonfinite[:10]}, count={len(npz_nonfinite)}")
    add(checks, "npz_model_arrays_nonempty", not npz_tiny, f"tiny={npz_tiny[:10]}, count={len(npz_tiny)}")
    add(checks, "joblib_models_predictable", not joblib_without_predict, f"without_predict={joblib_without_predict[:10]}, count={len(joblib_without_predict)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "model_artifact_integrity",
        "verified": len(issues) == 0,
        "models_dir": str(models_dir),
        "n_models": len(records),
        "counts_by_suffix": by_suffix,
        "total_bytes": total_bytes,
        "n_npz_arrays": int(sum(int(record.get("n_arrays") or 0) for record in npz_records)),
        "n_npz_numeric_arrays": int(sum(int(record.get("numeric_arrays") or 0) for record in npz_records)),
        "n_npz_elements": int(sum(int(record.get("total_elements") or 0) for record in npz_records)),
        "n_joblib_predictors": int(sum(1 for record in joblib_records if record.get("has_predict"))),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "unreadable_models": unreadable,
        "empty_models": empty,
        "npz_empty_arrays": npz_empty_arrays,
        "npz_nonfinite": npz_nonfinite,
        "joblib_without_predict": joblib_without_predict,
        "records": records,
    }


def model_artifact_integrity_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Model Artifact Integrity Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Models: {payload.get('n_models')}",
        f"- Counts by suffix: {payload.get('counts_by_suffix')}",
        f"- Total bytes: {payload.get('total_bytes')}",
        f"- NPZ arrays: {payload.get('n_npz_arrays')}",
        f"- NPZ elements: {payload.get('n_npz_elements')}",
        f"- Joblib predictors: {payload.get('n_joblib_predictors')}",
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
        lines.append("Committed model artifacts are loadable, nonempty, finite where numeric, and usable as predictors where stored as joblib estimators.")
    lines.append("")
    return "\n".join(lines)
