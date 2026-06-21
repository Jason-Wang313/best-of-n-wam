from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Callable, Iterable

for _thread_env in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_env, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for path in [SRC, EXPERIMENTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _add_optional_libero_paths(root: Path = ROOT) -> None:
    default_config = root.parent / "external_benchmarks" / ".libero"
    if default_config.exists():
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(default_config))
    candidates = [
        os.environ.get("LIBERO_SOURCE_PATH"),
        os.environ.get("WAM_LIBERO_SOURCE_PATH"),
        os.environ.get("LIBERO_SOURCE"),
        str(root.parent / "external_benchmarks" / "LIBERO"),
    ]
    for raw in candidates:
        if not raw:
            continue
        base = Path(raw).expanduser()
        for candidate in [base]:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))


_add_optional_libero_paths()

from benchmark_libero_wam import (  # noqa: E402
    DEFAULT_N_VALUES,
    TARGETS,
    RidgeWAM,
    TaskData,
    TaskSpec,
    _model_metrics,
    ci95,
    collect_task_data,
    normalized_utility,
    parse_task_spec,
    save_model,
)
from wam_inference_value.benchmarks.libero_adapter import is_libero_available  # noqa: E402
from wam_inference_value.theorem import simulate_best_of_n, utility_best_of_n_finite  # noqa: E402


RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
REPORTS = ROOT / "reports"
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FALLBACK_SUITE_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


@dataclass(frozen=True)
class Layout:
    out_dir: Path
    chunk_dir: Path
    row_dir: Path
    summary: Path
    status_summary: Path
    manifest: Path
    event_log: Path
    model: Path
    train_validation: Path
    eval_rollouts: Path
    curves: Path
    exact_law: Path
    seed_metrics: Path
    paper_table: Path
    report: Path
    root_summary: Path


@dataclass
class Preflight:
    disk_free_gb: float
    memory_available_gb: float | None
    min_disk_free_gb: float
    min_memory_available_gb: float
    ok: bool
    issues: list[str]


def set_low_priority() -> dict[str, Any]:
    """Best-effort process de-prioritization for laptop-friendly long runs."""

    if sys.platform.startswith("win"):
        try:
            import ctypes

            below_normal = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = bool(ctypes.windll.kernel32.SetPriorityClass(handle, below_normal))
            return {"attempted": True, "ok": ok, "mode": "windows_below_normal"}
        except Exception as exc:
            return {"attempted": True, "ok": False, "mode": "windows_below_normal", "error": f"{type(exc).__name__}: {exc}"}
    try:
        os.nice(10)
        return {"attempted": True, "ok": True, "mode": "posix_nice_10"}
    except Exception as exc:
        return {"attempted": True, "ok": False, "mode": "posix_nice_10", "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class CorrStats:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        self.n += int(x.size)
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_x2 += float(np.sum(x * x))
        self.sum_y2 += float(np.sum(y * y))
        self.sum_xy += float(np.sum(x * y))

    def corr(self) -> float:
        if self.n < 2:
            return 0.0
        cov = self.sum_xy - self.sum_x * self.sum_y / self.n
        vx = self.sum_x2 - self.sum_x * self.sum_x / self.n
        vy = self.sum_y2 - self.sum_y * self.sum_y / self.n
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        return float(cov / np.sqrt(vx * vy))


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_").lower()


def layout(output_root: Path, paper_root: Path, tag: str) -> Layout:
    safe_tag = safe_slug(tag or "libero_full_suite_serial")
    out_dir = output_root / "results" / safe_tag
    canonical = safe_tag == "libero_full_suite_serial"
    paper_table_path = (
        paper_root / "libero_full_suite_serial_table.tex"
        if canonical
        else out_dir / "libero_full_suite_serial_table.tex"
    )
    root_summary = (
        output_root / "results" / "benchmark_libero_full_suite_serial.json"
        if canonical
        else out_dir / "summary.json"
    )
    report_path = (
        output_root / "reports" / "libero_full_suite_serial_report.md"
        if canonical
        else out_dir / "libero_full_suite_serial_report.md"
    )
    return Layout(
        out_dir=out_dir,
        chunk_dir=out_dir / "chunks",
        row_dir=out_dir / "task_rows",
        summary=out_dir / "summary.json",
        status_summary=out_dir / "status_summary.json",
        manifest=out_dir / "progress_manifest.json",
        event_log=out_dir / "events.jsonl",
        model=out_dir / "benchmark_libero_full_suite_serial_ridge_wam.npz",
        train_validation=out_dir / "benchmark_libero_full_suite_serial_train_validation.csv",
        eval_rollouts=out_dir / "benchmark_libero_full_suite_serial_eval_rollouts.csv",
        curves=out_dir / "benchmark_libero_full_suite_serial_curves.csv",
        exact_law=out_dir / "benchmark_libero_full_suite_serial_exact_law.csv",
        seed_metrics=out_dir / "benchmark_libero_full_suite_serial_seed_metrics.csv",
        paper_table=paper_table_path,
        report=report_path,
        root_summary=root_summary,
    )


def append_event(paths: Layout, event: str, payload: dict[str, Any] | None = None) -> None:
    paths.event_log.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **(payload or {}),
    }
    with paths.event_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_sanitize(row), sort_keys=True) + "\n")


def available_memory_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available / (1024**3))
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.ullAvailPhys / (1024**3))
        except Exception:
            return None
    return None


def preflight(paths: Layout, min_disk_free_gb: float, min_memory_available_gb: float) -> Preflight:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(paths.out_dir)
    disk_free_gb = float(disk.free / (1024**3))
    memory_gb = available_memory_gb()
    issues: list[str] = []
    if disk_free_gb < float(min_disk_free_gb):
        issues.append(f"disk_free_gb={disk_free_gb:.2f} < {min_disk_free_gb:.2f}")
    if memory_gb is not None and memory_gb < float(min_memory_available_gb):
        issues.append(f"memory_available_gb={memory_gb:.2f} < {min_memory_available_gb:.2f}")
    return Preflight(
        disk_free_gb=disk_free_gb,
        memory_available_gb=memory_gb,
        min_disk_free_gb=float(min_disk_free_gb),
        min_memory_available_gb=float(min_memory_available_gb),
        ok=not issues,
        issues=issues,
    )


def discover_task_specs(suites: Iterable[str], *, allow_fallback: bool = True) -> tuple[list[TaskSpec], dict[str, Any]]:
    specs: list[TaskSpec] = []
    suite_counts: dict[str, int] = {}
    used_live_libero = False
    try:
        from libero.libero import benchmark

        for suite in suites:
            bm_cls = benchmark.get_benchmark(str(suite))
            bm = bm_cls(task_order_index=0)
            n_tasks = int(bm.get_num_tasks())
            suite_counts[str(suite)] = n_tasks
            specs.extend(TaskSpec(suite=str(suite), task_index=i) for i in range(n_tasks))
        used_live_libero = True
    except Exception as exc:
        if not allow_fallback:
            raise
        for suite in suites:
            n_tasks = int(FALLBACK_SUITE_COUNTS.get(str(suite), 10))
            suite_counts[str(suite)] = n_tasks
            specs.extend(TaskSpec(suite=str(suite), task_index=i) for i in range(n_tasks))
        return specs, {
            "used_live_libero": False,
            "fallback_task_counts": True,
            "suite_counts": suite_counts,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return specs, {
        "used_live_libero": used_live_libero,
        "fallback_task_counts": False,
        "suite_counts": suite_counts,
        "reason": "queried installed LIBERO benchmark registry",
    }


def resolve_task_specs(args: argparse.Namespace) -> tuple[list[TaskSpec], dict[str, Any]]:
    if args.tasks:
        specs = [parse_task_spec(raw) for raw in args.tasks]
        return specs, {"explicit_tasks": [spec.key for spec in specs], "fallback_task_counts": False}
    suites = list(args.suites)
    if args.include_libero_90 and "libero_90" not in suites:
        suites.append("libero_90")
    return discover_task_specs(suites, allow_fallback=args.allow_fallback_task_list)


def task_slug(spec: TaskSpec, ordinal: int) -> str:
    return f"{ordinal:04d}_{safe_slug(spec.key)}"


def chunk_paths(paths: Layout, spec: TaskSpec, ordinal: int) -> dict[str, Path]:
    slug = task_slug(spec, ordinal)
    return {
        "npz": paths.chunk_dir / f"{slug}.npz",
        "rows": paths.row_dir / f"{slug}.rows.json",
        "train_validation_csv": paths.row_dir / f"{slug}.train_validation.csv",
        "eval_csv": paths.row_dir / f"{slug}.eval_rollouts.csv",
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    tmp.replace(path)


def save_task_chunk(paths: Layout, spec: TaskSpec, ordinal: int, data: TaskData) -> dict[str, Path]:
    files = chunk_paths(paths, spec, ordinal)
    for directory in [paths.chunk_dir, paths.row_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    tmp_npz = files["npz"].with_suffix(".npz.tmp")
    with tmp_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_x=data.train_x,
            train_y=data.train_y,
            val_x=data.val_x,
            val_y=data.val_y,
            eval_x=data.eval_x,
        )
    tmp_npz.replace(files["npz"])
    rows_payload = {
        "task_key": data.task_key,
        "task_name": data.task_name,
        "task_index": int(data.task_index),
        "train_rows": data.train_rows,
        "val_rows": data.val_rows,
        "eval_rows": data.eval_rows,
    }
    write_json(files["rows"], rows_payload)
    write_rows_csv(files["train_validation_csv"], [*data.train_rows, *data.val_rows])
    write_rows_csv(files["eval_csv"], data.eval_rows)
    return files


def task_completed(paths: Layout, spec: TaskSpec, ordinal: int) -> bool:
    files = chunk_paths(paths, spec, ordinal)
    return all(path.exists() and path.stat().st_size > 0 for path in files.values())


def load_rows(path: Path) -> dict[str, Any]:
    return read_json(path)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=float) for key in data.files}


def initialize_manifest(paths: Layout, specs: list[TaskSpec], task_info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    existing = read_json(paths.manifest)
    existing_tasks = {
        str(row.get("task_key")): row
        for row in existing.get("tasks", [])
        if isinstance(row, dict) and row.get("task_key")
    }
    rows = []
    for ordinal, spec in enumerate(specs):
        old = existing_tasks.get(spec.key, {})
        files = chunk_paths(paths, spec, ordinal)
        status = "completed" if task_completed(paths, spec, ordinal) else str(old.get("status") or "pending")
        if status == "running":
            status = "pending"
        rows.append(
            {
                "ordinal": ordinal,
                "task_key": spec.key,
                "suite": spec.suite,
                "task_index": int(spec.task_index),
                "status": status,
                "chunk_npz": str(files["npz"]),
                "rows_json": str(files["rows"]),
                "train_validation_csv": str(files["train_validation_csv"]),
                "eval_csv": str(files["eval_csv"]),
                "started_at": old.get("started_at"),
                "completed_at": old.get("completed_at"),
                "seconds": old.get("seconds"),
                "error": old.get("error"),
            }
        )
    manifest = {
        "experiment": "benchmark_libero_full_suite_serial",
        "attempted": True,
        "mode": "serial_low_ram",
        "task_count": len(specs),
        "task_source": task_info,
        "runner_contract": {
            "one_task_at_a_time": True,
            "checkpoint_resume": True,
            "append_only_task_chunks": True,
            "stores_candidate_tensors_in_manifest": False,
            "parallel_jobs": 1,
            "single_thread_env_defaults": True,
            "low_priority_requested": bool(args.low_priority),
            "sleep_between_tasks_seconds": float(args.sleep_between_tasks),
        },
        "parameters": {
            "suites": list(args.suites),
            "explicit_tasks": list(args.tasks or []),
            "include_libero_90": bool(args.include_libero_90),
            "train_states": int(args.train_states),
            "train_rollouts": int(args.train_rollouts),
            "val_states": int(args.val_states),
            "val_rollouts": int(args.val_rollouts),
            "eval_states": int(args.eval_states),
            "eval_rollouts": int(args.eval_rollouts),
            "horizon": int(args.horizon),
            "n_values": [int(n) for n in args.n_values],
            "mc_trials": int(args.mc_trials),
            "seed": int(args.seed),
            "low_priority": bool(args.low_priority),
            "sleep_between_tasks": float(args.sleep_between_tasks),
        },
        "tasks": rows,
    }
    write_json(paths.manifest, manifest)
    return manifest


def update_task_status(paths: Layout, task_key: str, **updates: Any) -> None:
    manifest = read_json(paths.manifest)
    tasks = manifest.get("tasks") or []
    for row in tasks:
        if row.get("task_key") == task_key:
            row.update(updates)
            break
    write_json(paths.manifest, manifest)


def completed_task_rows(paths: Layout) -> list[dict[str, Any]]:
    manifest = read_json(paths.manifest)
    return [
        row
        for row in (manifest.get("tasks") or [])
        if row.get("status") == "completed"
        and Path(str(row.get("chunk_npz"))).exists()
        and Path(str(row.get("rows_json"))).exists()
    ]


def manifest_counts(manifest: dict[str, Any]) -> dict[str, int]:
    tasks = manifest.get("tasks") or []
    return {
        "task_count": int(manifest.get("task_count") or len(tasks)),
        "completed": sum(1 for row in tasks if row.get("status") == "completed"),
        "pending": sum(1 for row in tasks if row.get("status") == "pending"),
        "running": sum(1 for row in tasks if row.get("status") == "running"),
        "failed": sum(1 for row in tasks if row.get("status") == "failed"),
    }


def status_summary(paths: Layout, preflight_payload: Preflight) -> dict[str, Any]:
    manifest = read_json(paths.manifest)
    tasks = manifest.get("tasks") or []
    counts = manifest_counts(manifest)
    next_pending = next((row for row in tasks if row.get("status") == "pending"), None)
    complete = counts["task_count"] > 0 and counts["completed"] == counts["task_count"]
    return {
        "experiment": "benchmark_libero_full_suite_serial",
        "attempted": True,
        "available": bool(manifest),
        "verified": False,
        "complete": bool(complete),
        "status_only": True,
        "task_count": counts["task_count"],
        "completed_task_count": counts["completed"],
        "pending_task_count": counts["pending"],
        "running_task_count": counts["running"],
        "failed_task_count": counts["failed"],
        "next_pending_task": None
        if next_pending is None
        else {
            "ordinal": next_pending.get("ordinal"),
            "task_key": next_pending.get("task_key"),
            "suite": next_pending.get("suite"),
            "task_index": next_pending.get("task_index"),
        },
        "manifest_path": str(paths.manifest),
        "preflight": asdict(preflight_payload),
        "low_ram_contract": {
            "one_task_at_a_time": True,
            "parallel_jobs": 1,
            "single_thread_env_defaults": True,
            "checkpoint_resume": True,
            "append_only_task_chunks": True,
            "stores_candidate_tensors_in_manifest": False,
        },
        "claim_boundaries": {
            "real_robot": False,
            "modern_vla_scale_sota": False,
            "full_policy_success": False,
        },
    }


def wait_for_preflight(paths: Layout, args: argparse.Namespace, first: Preflight) -> Preflight:
    timeout_s = float(args.wait_for_preflight_seconds)
    if first.ok or timeout_s <= 0.0:
        return first
    interval_s = max(1.0, float(args.preflight_poll_seconds))
    deadline = time.time() + timeout_s
    attempt = 0
    latest = first
    append_event(
        paths,
        "preflight_wait_start",
        {"timeout_seconds": timeout_s, "poll_seconds": interval_s, "initial_issues": first.issues},
    )
    while not latest.ok and time.time() < deadline:
        sleep_s = min(interval_s, max(0.0, deadline - time.time()))
        if sleep_s <= 0.0:
            break
        time.sleep(sleep_s)
        attempt += 1
        latest = preflight(paths, args.min_disk_free_gb, args.min_available_ram_gb)
        append_event(
            paths,
            "preflight_wait_poll",
            {
                "attempt": attempt,
                "ok": latest.ok,
                "disk_free_gb": latest.disk_free_gb,
                "memory_available_gb": latest.memory_available_gb,
                "issues": latest.issues,
            },
        )
    append_event(paths, "preflight_wait_end", {"ok": latest.ok, "attempts": attempt, "issues": latest.issues})
    return latest


def compute_feature_stats(task_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, int]:
    n = 0
    sums: np.ndarray | None = None
    sums_sq: np.ndarray | None = None
    for row in task_rows:
        arrays = load_arrays(Path(str(row["chunk_npz"])))
        x = arrays["train_x"]
        if sums is None:
            sums = np.zeros(x.shape[1], dtype=float)
            sums_sq = np.zeros(x.shape[1], dtype=float)
        sums += np.sum(x, axis=0)
        sums_sq += np.sum(x * x, axis=0)
        n += int(x.shape[0])
    if sums is None or sums_sq is None or n <= 0:
        raise ValueError("no training rows available for streaming ridge fit")
    mean = sums / n
    var = np.maximum(sums_sq / n - mean * mean, 0.0)
    scale = np.sqrt(var)
    scale[scale < 1e-8] = 1.0
    return mean, scale, n


def fit_ridge_streaming(task_rows: list[dict[str, Any]], alpha: float) -> RidgeWAM:
    mean, scale, _ = compute_feature_stats(task_rows)
    dim = int(mean.size)
    xtx = np.zeros((dim + 1, dim + 1), dtype=float)
    xty = np.zeros((dim + 1, len(TARGETS)), dtype=float)
    for row in task_rows:
        arrays = load_arrays(Path(str(row["chunk_npz"])))
        z = (arrays["train_x"] - mean) / scale
        z = np.column_stack([np.ones(z.shape[0]), z])
        xtx += z.T @ z
        xty += z.T @ arrays["train_y"]
    reg = float(alpha) * np.eye(dim + 1)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(xtx + reg, xty)
    return RidgeWAM(mean=mean, scale=scale, weights=weights, target_names=TARGETS)


def streamed_model_metrics(model: RidgeWAM, task_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    count = 0
    abs_error = np.zeros(len(TARGETS), dtype=float)
    utility_corr = CorrStats()
    progress_corr = CorrStats()
    physics_corr = CorrStats()
    for row in task_rows:
        arrays = load_arrays(Path(str(row["chunk_npz"])))
        pred = model.predict(arrays["val_x"])
        y = arrays["val_y"]
        count += int(y.shape[0])
        abs_error += np.sum(np.abs(pred - y), axis=0)
        learned_physics = pred[:, 1] + float(args.success_bonus) * pred[:, 4] - float(args.energy_penalty) * pred[:, 3]
        utility_corr.update(pred[:, 0], y[:, 0])
        progress_corr.update(pred[:, 1], y[:, 1])
        physics_corr.update(learned_physics, y[:, 0])
    if count <= 0:
        raise ValueError("no validation rows available")
    return {
        "utility_mae": float(abs_error[0] / count),
        "utility_corr": utility_corr.corr(),
        "learned_physics_score_corr": physics_corr.corr(),
        "progress_mae": float(abs_error[1] / count),
        "progress_corr": progress_corr.corr(),
        "final_distance_mae": float(abs_error[2] / count),
        "energy_mae": float(abs_error[3] / count),
        "success_mae": float(abs_error[4] / count),
    }


def group_eval_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["task_key"]), int(row["seed"]), int(row["state_id"]))
        groups.setdefault(key, []).append(row)
    return groups


def finalize(paths: Layout, args: argparse.Namespace, preflight_payload: Preflight) -> dict[str, Any]:
    task_rows = completed_task_rows(paths)
    n_values = [int(n) for n in args.n_values]
    max_n = max(n_values)
    if len(task_rows) < int(args.min_tasks):
        summary = {
            "experiment": "benchmark_libero_full_suite_serial",
            "attempted": True,
            "available": False,
            "verified": False,
            "complete": False,
            "completed_task_count": len(task_rows),
            "required_min_tasks": int(args.min_tasks),
            "reason": "not enough completed task chunks to finalize",
            "preflight": asdict(preflight_payload),
            "manifest_path": str(paths.manifest),
            "note": "Runner can resume from progress_manifest.json; no full-suite claim is promoted until enough task chunks complete.",
        }
        write_outputs(paths, summary, table_only=True)
        return summary

    import pandas as pd

    model = fit_ridge_streaming(task_rows, alpha=args.ridge_alpha)
    model_metrics = streamed_model_metrics(model, task_rows, args)
    save_model(
        model,
        paths.model,
        {
            "experiment": "benchmark_libero_full_suite_serial",
            "tasks": [row["task_key"] for row in task_rows],
            "model_type": "streamed_libero_ridge_state_action_sequence_wam",
            "horizon": int(args.horizon),
            "train_states_per_task": int(args.train_states),
            "train_rollouts": int(args.train_rollouts),
            "serial_low_ram": True,
        },
    )

    train_val_rows: list[dict[str, Any]] = []
    eval_detail_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    seed_metrics: list[dict[str, Any]] = []
    task_metrics: list[dict[str, Any]] = []
    for task_row in task_rows:
        arrays = load_arrays(Path(str(task_row["chunk_npz"])))
        rows_payload = load_rows(Path(str(task_row["rows_json"])))
        train_val_rows.extend(rows_payload.get("train_rows") or [])
        train_val_rows.extend(rows_payload.get("val_rows") or [])
        task_eval_rows = list(rows_payload.get("eval_rows") or [])
        pred = model.predict(arrays["eval_x"])
        for raw_row, p in zip(task_eval_rows, pred):
            out = dict(raw_row)
            out.update(
                {
                    "predicted_utility": float(p[0]),
                    "predicted_progress": float(p[1]),
                    "predicted_final_distance": float(p[2]),
                    "predicted_energy": float(p[3]),
                    "predicted_success": float(p[4]),
                    "learned_physics_score": float(p[1] + args.success_bonus * p[4] - args.energy_penalty * p[3]),
                }
            )
            eval_detail_rows.append(out)

        task_seed_metrics: list[dict[str, Any]] = []
        for (task_key, seed, state_id), sub_rows in group_eval_rows(eval_detail_rows[-len(task_eval_rows) :]).items():
            real_utility = np.asarray([float(row["utility"]) for row in sub_rows], dtype=float)
            norm_utility = normalized_utility(real_utility)
            success = np.asarray([float(row["success"]) for row in sub_rows], dtype=float)
            energy = np.asarray([float(row["energy"]) for row in sub_rows], dtype=float)
            progress = np.asarray([float(row["progress"]) for row in sub_rows], dtype=float)
            final_distance = np.asarray([float(row["final_distance"]) for row in sub_rows], dtype=float)
            reward = np.asarray([float(row["total_reward"]) for row in sub_rows], dtype=float)
            pred_utility = np.asarray([float(row["predicted_utility"]) for row in sub_rows], dtype=float)
            pred_physics = np.asarray([float(row["learned_physics_score"]) for row in sub_rows], dtype=float)
            pred_progress = np.asarray([float(row["predicted_progress"]) for row in sub_rows], dtype=float)
            pred_success = np.asarray([float(row["predicted_success"]) for row in sub_rows], dtype=float)
            rng = np.random.default_rng(int(seed) + 37 * int(state_id))
            scorers = {
                "random": rng.normal(size=len(sub_rows)),
                "learned_wam": pred_utility,
                "learned_physics_score": pred_physics,
                "learned_energy_regularized": pred_utility - float(args.learned_energy_regularizer) * energy,
                "predicted_progress": pred_progress,
                "predicted_success": pred_success,
                "low_energy": -energy,
                "distance_progress": progress - final_distance,
                "benchmark_reward": reward,
                "oracle_real_utility": real_utility,
            }
            high_n_values: dict[str, float] = {}
            for scorer, scores in scorers.items():
                raw_curve = utility_best_of_n_finite(scores, real_utility, n_values)
                norm_curve = utility_best_of_n_finite(scores, norm_utility, n_values)
                succ_curve = utility_best_of_n_finite(scores, success, n_values)
                for n in n_values:
                    curve_rows.append(
                        {
                            "task_key": task_key,
                            "seed": int(seed),
                            "state_id": int(state_id),
                            "scorer": scorer,
                            "N": int(n),
                            "real_utility": float(raw_curve[n]),
                            "normalized_real_utility": float(norm_curve[n]),
                            "success": float(succ_curve[n]),
                        }
                    )
                high_n_values[scorer] = float(norm_curve[max_n])
                if scorer in {"learned_wam", "learned_physics_score", "learned_energy_regularized", "oracle_real_utility"}:
                    for n in n_values:
                        mc = simulate_best_of_n(
                            scores,
                            real_utility,
                            n,
                            int(args.mc_trials),
                            int(seed) + 100 * n + 13 * int(state_id),
                        )
                        exact_rows.append(
                            {
                                "task_key": task_key,
                                "seed": int(seed),
                                "state_id": int(state_id),
                                "scorer": scorer,
                                "N": int(n),
                                "utility_exact": float(raw_curve[n]),
                                "utility_mc": float(mc),
                                "utility_abs_error": float(abs(raw_curve[n] - mc)),
                            }
                        )
            learned_deltas = {
                "learned_wam": high_n_values.get("learned_wam", np.nan) - high_n_values.get("random", np.nan),
                "learned_physics_score": high_n_values.get("learned_physics_score", np.nan) - high_n_values.get("random", np.nan),
                "learned_energy_regularized": high_n_values.get("learned_energy_regularized", np.nan)
                - high_n_values.get("random", np.nan),
            }
            best_name = max(learned_deltas, key=lambda k: learned_deltas[k] if np.isfinite(learned_deltas[k]) else -np.inf)
            metric_row = {
                "task_key": task_key,
                "seed": int(seed),
                "state_id": int(state_id),
                f"learned_wam_minus_random_N{max_n}": float(learned_deltas["learned_wam"]),
                f"learned_physics_minus_random_N{max_n}": float(learned_deltas["learned_physics_score"]),
                f"learned_energy_regularized_minus_random_N{max_n}": float(learned_deltas["learned_energy_regularized"]),
                f"best_learned_minus_random_N{max_n}": float(learned_deltas[best_name]),
                "best_learned_scorer": best_name,
                f"oracle_minus_random_N{max_n}": float(high_n_values.get("oracle_real_utility", np.nan) - high_n_values.get("random", np.nan)),
                f"oracle_minus_best_learned_N{max_n}": float(high_n_values.get("oracle_real_utility", np.nan) - high_n_values.get(best_name, np.nan)),
                f"benchmark_reward_minus_random_N{max_n}": float(high_n_values.get("benchmark_reward", np.nan) - high_n_values.get("random", np.nan)),
            }
            seed_metrics.append(metric_row)
            task_seed_metrics.append(metric_row)
        if task_seed_metrics:
            task_df = pd.DataFrame(task_seed_metrics)
            task_metrics.append(
                {
                    "task_key": task_row["task_key"],
                    "eval_rollout_pools": int(len(task_df)),
                    f"best_learned_minus_random_N{max_n}_mean": float(task_df[f"best_learned_minus_random_N{max_n}"].mean()),
                }
            )
        gc.collect()

    write_rows_csv(paths.train_validation, train_val_rows)
    write_rows_csv(paths.eval_rollouts, eval_detail_rows)
    write_rows_csv(paths.curves, curve_rows)
    write_rows_csv(paths.exact_law, exact_rows)
    write_rows_csv(paths.seed_metrics, seed_metrics)

    seed_df = pd.DataFrame(seed_metrics)
    confidence_intervals = {
        key: ci95(seed_df[key].to_numpy())
        for key in seed_df.columns
        if key not in {"task_key", "seed", "state_id", "best_learned_scorer"}
    }
    learned_ci_keys = {
        "learned_wam": f"learned_wam_minus_random_N{max_n}",
        "learned_physics_score": f"learned_physics_minus_random_N{max_n}",
        "learned_energy_regularized": f"learned_energy_regularized_minus_random_N{max_n}",
    }
    promoted_scorer = max(
        learned_ci_keys,
        key=lambda scorer: (confidence_intervals.get(learned_ci_keys[scorer]) or {}).get("lo")
        if (confidence_intervals.get(learned_ci_keys[scorer]) or {}).get("lo") is not None
        else -np.inf,
    )
    confidence_intervals[f"best_learned_minus_random_N{max_n}"] = confidence_intervals[learned_ci_keys[promoted_scorer]]
    exact_mae = float(np.mean([float(row["utility_abs_error"]) for row in exact_rows])) if exact_rows else None
    promoted_ci = confidence_intervals.get(f"best_learned_minus_random_N{max_n}") or {}
    manifest = read_json(paths.manifest)
    total_tasks = int(manifest.get("task_count") or len(task_rows))
    completed_task_count = len(task_rows)
    complete = completed_task_count == total_tasks
    verified = (
        completed_task_count >= int(args.min_tasks)
        and exact_mae is not None
        and exact_mae < float(args.max_exact_mae)
        and promoted_ci.get("n", 0) >= int(args.min_eval_pools)
        and promoted_ci.get("lo") is not None
        and promoted_ci["lo"] > 0.0
        and model_metrics["utility_corr"] > 0.0
    )
    summary = {
        "experiment": "benchmark_libero_full_suite_serial",
        "attempted": True,
        "available": True,
        "verified": bool(verified),
        "complete": bool(complete),
        "scope": "full configured LIBERO suites, serial low-RAM rollout-pool WAM audit",
        "task_count": total_tasks,
        "completed_task_count": completed_task_count,
        "incomplete_task_count": max(0, total_tasks - completed_task_count),
        "completed_tasks": [row["task_key"] for row in task_rows],
        "suites": list(args.suites),
        "include_libero_90": bool(args.include_libero_90),
        "low_ram_contract": {
            "one_task_at_a_time": True,
            "parallel_jobs": 1,
            "single_thread_env_defaults": True,
            "low_priority_requested": bool(args.low_priority),
            "low_priority": getattr(args, "_low_priority_result", {"attempted": False}),
            "sleep_between_tasks_seconds": float(args.sleep_between_tasks),
            "checkpoint_resume": True,
            "append_only_task_chunks": True,
            "aggregate_rebuilt_from_chunks": True,
            "stores_candidate_tensors_in_manifest": False,
        },
        "preflight": asdict(preflight_payload),
        "model_path": str(paths.model),
        "model_type": "streamed_libero_ridge_state_action_sequence_wam",
        "train_states_per_task": int(args.train_states),
        "train_rollouts": int(args.train_rollouts),
        "validation_states_per_task": int(args.val_states),
        "validation_rollouts": int(args.val_rollouts),
        "eval_states_per_task": int(args.eval_states),
        "eval_rollouts": int(args.eval_rollouts),
        "horizon": int(args.horizon),
        "n_values": n_values,
        "max_n": max_n,
        "train_samples": int(sum(load_arrays(Path(str(row["chunk_npz"])))["train_x"].shape[0] for row in task_rows)),
        "validation_samples": int(sum(load_arrays(Path(str(row["chunk_npz"])))["val_x"].shape[0] for row in task_rows)),
        "eval_samples": int(len(eval_detail_rows)),
        "eval_rollout_pools": int(len(seed_metrics)),
        "model_metrics": model_metrics,
        "exact_law_utility_mae": exact_mae,
        "confidence_intervals": confidence_intervals,
        "promoted_scorer": promoted_scorer,
        "task_metrics": task_metrics,
        "artifacts": {
            "manifest": str(paths.manifest),
            "event_log": str(paths.event_log),
            "train_validation": str(paths.train_validation),
            "eval_rollouts": str(paths.eval_rollouts),
            "curves": str(paths.curves),
            "exact_law": str(paths.exact_law),
            "seed_metrics": str(paths.seed_metrics),
            "paper_table": str(paths.paper_table),
            "report": str(paths.report),
        },
        "claim_boundaries": {
            "real_robot": False,
            "modern_vla_scale_sota": False,
            "full_policy_success": False,
            "evidence_type": "CPU serial rollout-pool audit, not real-robot or VLA-scale SOTA evidence",
        },
    }
    write_outputs(paths, summary, table_only=False)
    return summary


def fmt_percent(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.1f}\\%"
    except (TypeError, ValueError):
        return "--"


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def paper_table(summary: dict[str, Any]) -> str:
    ci = (summary.get("confidence_intervals") or {}).get(f"best_learned_minus_random_N{summary.get('max_n', 8)}") or {}
    status = "complete" if summary.get("complete") else "in progress"
    verified = "verified" if summary.get("verified") else "not promoted"
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{p{0.30\\linewidth}p{0.28\\linewidth}p{0.34\\linewidth}}",
            "\\toprule",
            "LIBERO serial check & Result & Claim boundary \\\\",
            "\\midrule",
            f"Configured suite coverage & {summary.get('completed_task_count', 0)} / {summary.get('task_count', 0)} tasks; {status} & Full-suite means configured LIBERO task coverage, not hardware. \\\\",
            f"Execution contract & one task at a time; jobs=1; low priority & Low-RAM CPU scheduling, not GPU-scale training. \\\\",
            f"Checkpointing & manifest + immutable task chunks & Resume after sleep/reboot without discarding completed tasks. \\\\",
            f"Rollout-pool audit & {summary.get('eval_rollout_pools', 0)} pools; N up to {summary.get('max_n', '--')} & Dense rollout-pool utility, not solved-policy success. \\\\",
            f"Learned-vs-random CI & lower {fmt_float(ci.get('lo'))}; status {verified} & Promoted only if the frozen CI and exact-law gates pass. \\\\",
            f"Exact-law check & MAE {fmt_float(summary.get('exact_law_utility_mae'))} & Finite tied-pool accounting, not real-robot validation. \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Low-RAM serial LIBERO full-suite rollout-pool audit. The table separates full configured-suite CPU breadth from nonclaims about real robots, modern VLA-scale SOTA, or full policy success.}",
            "\\label{tab:libero-full-suite-serial}",
            "\\end{table}",
            "",
        ]
    )


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LIBERO Full-Suite Serial Benchmark",
        "",
        f"- status: `{'verified' if summary.get('verified') else 'not_promoted'}`",
        f"- complete: `{summary.get('complete')}`",
        f"- completed tasks: `{summary.get('completed_task_count')}` / `{summary.get('task_count')}`",
        f"- eval rollout pools: `{summary.get('eval_rollout_pools')}`",
        f"- one task at a time: `{((summary.get('low_ram_contract') or {}).get('one_task_at_a_time'))}`",
        f"- parallel jobs: `{((summary.get('low_ram_contract') or {}).get('parallel_jobs'))}`",
        f"- low priority: `{((summary.get('low_ram_contract') or {}).get('low_priority'))}`",
        f"- sleep between tasks: `{((summary.get('low_ram_contract') or {}).get('sleep_between_tasks_seconds'))}`",
        f"- checkpoint manifest: `{((summary.get('artifacts') or {}).get('manifest'))}`",
        f"- event log: `{((summary.get('artifacts') or {}).get('event_log'))}`",
        "",
        "This artifact is a CPU serial rollout-pool WAM audit over configured LIBERO suites. It is not real-world robot evidence, not modern VLA-scale SOTA evidence, and not a full solved-policy success claim.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(paths: Layout, summary: dict[str, Any], *, table_only: bool) -> None:
    write_json(paths.summary, summary)
    write_json(paths.root_summary, summary)
    try:
        table_path = paths.paper_table.resolve()
        out_path = paths.out_dir.resolve()
        table_inside_output = table_path == out_path or out_path in table_path.parents
    except OSError:
        table_inside_output = False
    publish_table = table_inside_output or bool(summary.get("complete"))
    if publish_table:
        paths.paper_table.parent.mkdir(parents=True, exist_ok=True)
        paths.paper_table.write_text(paper_table(summary), encoding="utf-8")
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(report_markdown(summary), encoding="utf-8")
    if not table_only:
        append_event(paths, "finalized", {"verified": summary.get("verified"), "complete": summary.get("complete")})


def write_status_output(paths: Layout, summary: dict[str, Any]) -> None:
    write_json(paths.status_summary, summary)


def collect_serial(
    paths: Layout,
    specs: list[TaskSpec],
    args: argparse.Namespace,
    collector: Callable[[TaskSpec, int, int, argparse.Namespace], TaskData],
) -> None:
    n_tasks = len(specs)
    completed_this_run = 0
    for ordinal, spec in enumerate(specs):
        if task_completed(paths, spec, ordinal) and not args.force:
            update_task_status(paths, spec.key, status="completed", error=None)
            append_event(paths, "task_reused", {"task_key": spec.key, "ordinal": ordinal})
            continue
        if args.stop_after_tasks is not None and completed_this_run >= int(args.stop_after_tasks):
            append_event(paths, "stop_after_tasks", {"completed_this_run": completed_this_run})
            break
        pf = preflight(paths, args.min_disk_free_gb, args.min_available_ram_gb)
        pf = wait_for_preflight(paths, args, pf)
        if not pf.ok:
            update_task_status(paths, spec.key, status="pending", error="; ".join(pf.issues))
            append_event(paths, "preflight_failed_before_task", {"task_key": spec.key, "issues": pf.issues})
            if args.fail_on_preflight:
                raise RuntimeError("; ".join(pf.issues))
            break
        started = time.time()
        update_task_status(
            paths,
            spec.key,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            error=None,
        )
        append_event(paths, "task_start", {"task_key": spec.key, "ordinal": ordinal})
        try:
            data = collector(spec, ordinal, n_tasks, args)
            files = save_task_chunk(paths, spec, ordinal, data)
            elapsed = time.time() - started
            update_task_status(
                paths,
                spec.key,
                status="completed",
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                seconds=float(elapsed),
                error=None,
            )
            append_event(
                paths,
                "task_complete",
                {
                    "task_key": spec.key,
                    "ordinal": ordinal,
                    "seconds": float(elapsed),
                    "chunk_npz": str(files["npz"]),
                },
            )
            completed_this_run += 1
            if float(args.sleep_between_tasks) > 0.0:
                append_event(paths, "sleep_between_tasks", {"seconds": float(args.sleep_between_tasks)})
                time.sleep(float(args.sleep_between_tasks))
        except Exception as exc:
            update_task_status(paths, spec.key, status="failed", error=f"{type(exc).__name__}: {exc}")
            append_event(paths, "task_failed", {"task_key": spec.key, "ordinal": ordinal, "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise
        finally:
            gc.collect()


def run(
    args: argparse.Namespace,
    *,
    collector: Callable[[TaskSpec, int, int, argparse.Namespace], TaskData] = collect_task_data,
    availability_checker: Callable[[], tuple[bool, str]] = is_libero_available,
) -> dict[str, Any]:
    paths = layout(Path(args.output_root), Path(args.paper_root), args.output_tag)
    for directory in [paths.out_dir, paths.chunk_dir, paths.row_dir, paths.paper_table.parent, paths.report.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    args._low_priority_result = set_low_priority() if bool(args.low_priority) else {"attempted": False}
    append_event(paths, "priority", getattr(args, "_low_priority_result", {"attempted": False}))
    pf = preflight(paths, args.min_disk_free_gb, args.min_available_ram_gb)
    append_event(paths, "preflight", asdict(pf))
    if args.status:
        summary = status_summary(paths, pf)
        write_status_output(paths, summary)
        return summary
    if args.finalize_only:
        specs, task_info = resolve_task_specs(args)
        if args.max_tasks is not None:
            specs = specs[: int(args.max_tasks)]
            task_info = {**task_info, "max_tasks": int(args.max_tasks)}
        manifest = initialize_manifest(paths, specs, task_info, args)
        append_event(paths, "finalize_only", {"task_count": len(specs), "manifest_task_count": manifest.get("task_count")})
        return finalize(paths, args, pf)
    pf = wait_for_preflight(paths, args, pf)
    if not pf.ok:
        summary = {
            "experiment": "benchmark_libero_full_suite_serial",
            "attempted": True,
            "available": False,
            "verified": False,
            "complete": False,
            "reason": "preflight failed",
            "preflight": asdict(pf),
            "claim_boundaries": {
                "real_robot": False,
                "modern_vla_scale_sota": False,
                "full_policy_success": False,
            },
        }
        write_outputs(paths, summary, table_only=True)
        if args.fail_on_preflight:
            raise SystemExit(2)
        return summary
    specs, task_info = resolve_task_specs(args)
    if args.max_tasks is not None:
        specs = specs[: int(args.max_tasks)]
        task_info = {**task_info, "max_tasks": int(args.max_tasks)}
    manifest = initialize_manifest(paths, specs, task_info, args)
    if args.list_tasks:
        summary = {
            "experiment": "benchmark_libero_full_suite_serial",
            "attempted": True,
            "available": True,
            "verified": False,
            "complete": False,
            "task_count": len(specs),
            "completed_task_count": len(completed_task_rows(paths)),
            "tasks": [spec.key for spec in specs],
            "task_source": task_info,
            "manifest_path": str(paths.manifest),
            "preflight": asdict(pf),
            "claim_boundaries": {
                "real_robot": False,
                "modern_vla_scale_sota": False,
                "full_policy_success": False,
            },
        }
        write_outputs(paths, summary, table_only=True)
        return summary
    ok, reason = availability_checker()
    if not ok:
        summary = {
            "experiment": "benchmark_libero_full_suite_serial",
            "attempted": True,
            "available": False,
            "verified": False,
            "complete": False,
            "task_count": len(specs),
            "completed_task_count": len(completed_task_rows(paths)),
            "reason": reason,
            "task_source": task_info,
            "manifest_path": str(paths.manifest),
            "preflight": asdict(pf),
            "claim_boundaries": {
                "real_robot": False,
                "modern_vla_scale_sota": False,
                "full_policy_success": False,
            },
            "note": "Run with a LIBERO-compatible interpreter and LIBERO_SOURCE_PATH to collect task chunks.",
        }
        write_outputs(paths, summary, table_only=True)
        return summary
    append_event(paths, "collection_start", {"task_count": len(specs), "manifest_task_count": manifest.get("task_count")})
    collect_serial(paths, specs, args, collector)
    return finalize(paths, args, pf)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-RAM serial LIBERO full-suite rollout-pool WAM benchmark.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--paper-root", type=Path, default=PAPER)
    parser.add_argument("--output-tag", default="libero_full_suite_serial")
    parser.add_argument("--suites", nargs="*", default=list(DEFAULT_SUITES))
    parser.add_argument("--include-libero-90", action="store_true")
    parser.add_argument("--allow-fallback-task-list", action="store_true", default=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--stop-after-tasks", type=int, default=None)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fail-on-preflight", action="store_true")
    parser.add_argument("--low-priority", dest="low_priority", action="store_true", default=True)
    parser.add_argument("--no-low-priority", dest="low_priority", action="store_false")
    parser.add_argument("--sleep-between-tasks", type=float, default=0.0)
    parser.add_argument("--min-disk-free-gb", type=float, default=2.0)
    parser.add_argument("--min-available-ram-gb", type=float, default=1.5)
    parser.add_argument("--wait-for-preflight-seconds", type=float, default=0.0)
    parser.add_argument("--preflight-poll-seconds", type=float, default=30.0)
    parser.add_argument("--train-states", type=int, default=2)
    parser.add_argument("--train-rollouts", type=int, default=8)
    parser.add_argument("--val-states", type=int, default=1)
    parser.add_argument("--val-rollouts", type=int, default=8)
    parser.add_argument("--eval-states", type=int, default=2)
    parser.add_argument("--eval-rollouts", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--mc-trials", type=int, default=600)
    parser.add_argument("--n-values", nargs="*", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--action-scale", type=float, default=0.65)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--eef-weight", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--energy-penalty", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--learned-energy-regularizer", type=float, default=0.03)
    parser.add_argument("--max-exact-mae", type=float, default=0.04)
    parser.add_argument("--min-tasks", type=int, default=4)
    parser.add_argument("--min-eval-pools", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    preflight_payload = summary.get("preflight") or {}
    next_task = (summary.get("next_pending_task") or {}).get("task_key")
    ram = preflight_payload.get("memory_available_gb")
    ram_text = "unknown" if ram is None else f"{float(ram):.2f}GB"
    print(
        "LIBERO serial benchmark: "
        f"available={summary.get('available')} complete={summary.get('complete')} "
        f"verified={summary.get('verified')} tasks={summary.get('completed_task_count')}/{summary.get('task_count')} "
        f"preflight={preflight_payload.get('ok')} ram={ram_text} next={next_task}"
    )


if __name__ == "__main__":
    main()
