from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import benchmark_libero_full_suite_serial as serial


def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return serial.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--paper-root",
            str(tmp_path / "paper"),
            "--tasks",
            "libero_object/0",
            "libero_object/1",
            "--train-states",
            "1",
            "--train-rollouts",
            "6",
            "--val-states",
            "1",
            "--val-rollouts",
            "6",
            "--eval-states",
            "3",
            "--eval-rollouts",
            "6",
            "--horizon",
            "2",
            "--n-values",
            "1",
            "2",
            "4",
            "--mc-trials",
            "800",
            "--min-tasks",
            "2",
            "--min-eval-pools",
            "2",
            "--max-exact-mae",
            "1.0",
            "--min-available-ram-gb",
            "0",
            "--min-disk-free-gb",
            "0",
            "--no-low-priority",
            *extra,
        ]
    )


def _target_row(utility: float, energy: float) -> list[float]:
    progress = utility
    final_distance = 1.0 - utility
    success = 1.0 if utility > 0.8 else 0.0
    return [utility, progress, final_distance, energy, success]


def _features(utilities: np.ndarray, task_index: int, n_tasks: int) -> np.ndarray:
    one_hot = np.zeros((len(utilities), n_tasks), dtype=float)
    one_hot[:, task_index] = 1.0
    base = np.column_stack(
        [
            utilities,
            utilities**2,
            np.linspace(0.0, 1.0, len(utilities)),
            np.full(len(utilities), float(task_index)),
        ]
    )
    return np.concatenate([base, one_hot], axis=1)


def _rows(
    *,
    task_key: str,
    task_name: str,
    task_index: int,
    split: str,
    utilities: np.ndarray,
    state_id: int,
    seed: int,
    feature_dim: int,
) -> list[dict]:
    out = []
    for rollout_id, utility in enumerate(utilities):
        energy = 0.01 * float(rollout_id)
        target = _target_row(float(utility), energy)
        out.append(
            {
                "task_key": task_key,
                "task_name": task_name,
                "task_index": task_index,
                "split": split,
                "state_id": state_id,
                "seed": seed,
                "rollout_id": rollout_id,
                "feature_dim": feature_dim,
                "utility": target[0],
                "progress": target[1],
                "final_distance": target[2],
                "energy": energy,
                "success": bool(target[4]),
                "total_reward": target[0],
            }
        )
    return out


def fake_collect_task_data(spec: serial.TaskSpec, task_index: int, n_tasks: int, args: argparse.Namespace) -> serial.TaskData:
    train_util = np.linspace(0.0, 1.0, int(args.train_rollouts))
    val_util = np.linspace(0.05, 0.95, int(args.val_rollouts))
    eval_blocks = [np.linspace(0.0, 1.0, int(args.eval_rollouts)) for _ in range(int(args.eval_states))]
    train_x = _features(train_util, task_index, n_tasks)
    val_x = _features(val_util, task_index, n_tasks)
    eval_x = np.vstack([_features(block, task_index, n_tasks) for block in eval_blocks])
    train_y = np.asarray([_target_row(float(u), 0.01 * i) for i, u in enumerate(train_util)], dtype=float)
    val_y = np.asarray([_target_row(float(u), 0.01 * i) for i, u in enumerate(val_util)], dtype=float)
    feature_dim = train_x.shape[1]
    train_rows = _rows(
        task_key=spec.key,
        task_name=f"{spec.suite}/fake_{spec.task_index}",
        task_index=task_index,
        split="train",
        utilities=train_util,
        state_id=0,
        seed=100 + task_index,
        feature_dim=feature_dim,
    )
    val_rows = _rows(
        task_key=spec.key,
        task_name=f"{spec.suite}/fake_{spec.task_index}",
        task_index=task_index,
        split="validation",
        utilities=val_util,
        state_id=0,
        seed=200 + task_index,
        feature_dim=feature_dim,
    )
    eval_rows = []
    for state_id, block in enumerate(eval_blocks):
        eval_rows.extend(
            _rows(
                task_key=spec.key,
                task_name=f"{spec.suite}/fake_{spec.task_index}",
                task_index=task_index,
                split="eval",
                utilities=block,
                state_id=state_id,
                seed=300 + 10 * task_index + state_id,
                feature_dim=feature_dim,
            )
        )
    return serial.TaskData(
        task_key=spec.key,
        task_name=f"{spec.suite}/fake_{spec.task_index}",
        task_index=task_index,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        eval_x=eval_x,
        train_rows=train_rows,
        val_rows=val_rows,
        eval_rows=eval_rows,
    )


def test_serial_libero_runner_checkpoints_finalizes_and_resumes(tmp_path: Path) -> None:
    args = _args(tmp_path)
    summary = serial.run(args, collector=fake_collect_task_data, availability_checker=lambda: (True, "fake LIBERO"))

    assert summary["complete"] is True
    assert summary["verified"] is True
    assert summary["completed_task_count"] == 2
    assert summary["low_ram_contract"]["one_task_at_a_time"] is True
    assert summary["low_ram_contract"]["parallel_jobs"] == 1
    assert summary["low_ram_contract"]["single_thread_env_defaults"] is True
    assert summary["low_ram_contract"]["append_only_task_chunks"] is True
    assert summary["eval_rollout_pools"] == 6

    manifest = json.loads((tmp_path / "results" / "libero_full_suite_serial" / "progress_manifest.json").read_text())
    assert [row["status"] for row in manifest["tasks"]] == ["completed", "completed"]
    assert len(list((tmp_path / "results" / "libero_full_suite_serial" / "chunks").glob("*.npz"))) == 2
    assert (tmp_path / "results" / "libero_full_suite_serial" / "benchmark_libero_full_suite_serial_curves.csv").exists()

    table = (tmp_path / "paper" / "libero_full_suite_serial_table.tex").read_text(encoding="utf-8")
    assert "one task at a time" in table
    assert "not real-robot validation" in table
    assert "not GPU-scale training" in table
    assert "not solved-policy success" in table

    def should_not_collect(*_args, **_kwargs):  # pragma: no cover - only called on broken resume
        raise AssertionError("resume should reuse existing task chunks")

    resumed = serial.run(args, collector=should_not_collect, availability_checker=lambda: (True, "fake LIBERO"))
    assert resumed["complete"] is True
    assert resumed["completed_task_count"] == 2
    events = (tmp_path / "results" / "libero_full_suite_serial" / "events.jsonl").read_text(encoding="utf-8")
    assert "task_reused" in events


def test_serial_libero_runner_preflight_blocks_collection(tmp_path: Path) -> None:
    args = _args(tmp_path, "--min-available-ram-gb", "9999")
    summary = serial.run(args, collector=fake_collect_task_data, availability_checker=lambda: (True, "fake LIBERO"))

    assert summary["verified"] is False
    assert summary["complete"] is False
    assert summary["reason"] == "preflight failed"
    assert summary["preflight"]["ok"] is False
    assert not list((tmp_path / "results" / "libero_full_suite_serial" / "chunks").glob("*.npz"))


def test_serial_libero_status_reports_manifest_without_collection(tmp_path: Path) -> None:
    args = _args(tmp_path, "--stop-after-tasks", "1")
    partial = serial.run(args, collector=fake_collect_task_data, availability_checker=lambda: (True, "fake LIBERO"))
    assert partial["completed_task_count"] == 1

    def should_not_collect(*_args, **_kwargs):  # pragma: no cover - only called on broken status mode
        raise AssertionError("status mode should not collect chunks")

    status_args = _args(tmp_path, "--status")
    summary = serial.run(status_args, collector=should_not_collect, availability_checker=lambda: (True, "fake LIBERO"))

    assert summary["status_only"] is True
    assert summary["completed_task_count"] == 1
    assert summary["pending_task_count"] == 1
    assert summary["next_pending_task"]["task_key"] == "libero_object/1"


def test_wait_for_preflight_polls_until_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path, "--wait-for-preflight-seconds", "5", "--preflight-poll-seconds", "1")
    paths = serial.layout(tmp_path, tmp_path / "paper", "libero_full_suite_serial")
    calls = {"n": 0}
    blocked = serial.Preflight(
        disk_free_gb=10.0,
        memory_available_gb=0.5,
        min_disk_free_gb=0.0,
        min_memory_available_gb=0.0,
        ok=False,
        issues=["memory low"],
    )
    safe = serial.Preflight(
        disk_free_gb=10.0,
        memory_available_gb=3.0,
        min_disk_free_gb=0.0,
        min_memory_available_gb=0.0,
        ok=True,
        issues=[],
    )

    def fake_preflight(*_args, **_kwargs):
        calls["n"] += 1
        return safe

    monkeypatch.setattr(serial, "preflight", fake_preflight)
    monkeypatch.setattr(serial.time, "sleep", lambda _seconds: None)

    result = serial.wait_for_preflight(paths, args, blocked)

    assert result.ok is True
    assert calls["n"] == 1
    events = (tmp_path / "results" / "libero_full_suite_serial" / "events.jsonl").read_text(encoding="utf-8")
    assert "preflight_wait_start" in events
    assert "preflight_wait_end" in events


def test_serial_runner_waits_between_reused_and_pending_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_args = _args(tmp_path, "--stop-after-tasks", "1")
    partial = serial.run(first_args, collector=fake_collect_task_data, availability_checker=lambda: (True, "fake LIBERO"))
    assert partial["completed_task_count"] == 1

    safe = serial.Preflight(
        disk_free_gb=10.0,
        memory_available_gb=3.0,
        min_disk_free_gb=0.0,
        min_memory_available_gb=0.0,
        ok=True,
        issues=[],
    )
    blocked = serial.Preflight(
        disk_free_gb=10.0,
        memory_available_gb=0.5,
        min_disk_free_gb=0.0,
        min_memory_available_gb=0.0,
        ok=False,
        issues=["memory low"],
    )
    sequence = [safe, blocked, safe]

    def fake_preflight(*_args, **_kwargs):
        return sequence.pop(0)

    monkeypatch.setattr(serial, "preflight", fake_preflight)
    monkeypatch.setattr(serial.time, "sleep", lambda _seconds: None)

    resume_args = _args(tmp_path, "--stop-after-tasks", "1", "--wait-for-preflight-seconds", "5", "--preflight-poll-seconds", "1")
    resumed = serial.run(resume_args, collector=fake_collect_task_data, availability_checker=lambda: (True, "fake LIBERO"))

    assert resumed["complete"] is True
    assert resumed["completed_task_count"] == 2
    assert sequence == []
    events = (tmp_path / "results" / "libero_full_suite_serial" / "events.jsonl").read_text(encoding="utf-8")
    assert "task_reused" in events
    assert "preflight_wait_start" in events
    assert "preflight_wait_end" in events


def test_discover_task_specs_has_fallback_for_noninteractive_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __builtins__["__import__"]

    def blocked_import(name, *args, **kwargs):
        if name.startswith("libero"):
            raise ModuleNotFoundError("blocked LIBERO import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(__builtins__, "__import__", blocked_import)
    specs, info = serial.discover_task_specs(["libero_object"], allow_fallback=True)

    assert len(specs) == 10
    assert specs[0].key == "libero_object/0"
    assert info["fallback_task_counts"] is True
