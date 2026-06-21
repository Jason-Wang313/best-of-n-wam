from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments import libero_main_paper_summary as libero_main


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_libero_main_summary_derives_suite_and_worst_task_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    seed_metrics_path = tmp_path / "seed_metrics.csv"
    out_csv = tmp_path / "summary.csv"
    out_json = tmp_path / "summary_out.json"
    table_path = tmp_path / "paper" / "libero_main_suite_table.tex"

    summary_path.write_text(
        json.dumps(
            {
                "complete": True,
                "verified": True,
                "completed_task_count": 2,
                "task_count": 2,
                "eval_rollout_pools": 4,
                "max_n": 8,
                "promoted_scorer": "learned_wam",
                "claim_boundaries": {
                    "real_robot": False,
                    "modern_vla_scale_sota": False,
                    "full_policy_success": False,
                },
                "task_metrics": [
                    {"task_key": "libero_spatial/0", "best_learned_minus_random_N8_mean": 0.2},
                    {"task_key": "libero_object/0", "best_learned_minus_random_N8_mean": 0.4},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        seed_metrics_path,
        [
            {
                "task_key": "libero_spatial/0",
                "learned_wam_minus_random_N8": 0.1,
                "best_learned_minus_random_N8": 0.9,
            },
            {
                "task_key": "libero_spatial/0",
                "learned_wam_minus_random_N8": 0.3,
                "best_learned_minus_random_N8": 0.9,
            },
            {
                "task_key": "libero_object/0",
                "learned_wam_minus_random_N8": 0.3,
                "best_learned_minus_random_N8": 0.9,
            },
            {
                "task_key": "libero_object/0",
                "learned_wam_minus_random_N8": 0.5,
                "best_learned_minus_random_N8": 0.9,
            },
        ],
    )

    payload = libero_main.generate(summary_path, seed_metrics_path, out_csv, out_json, table_path)

    assert payload["complete"] is True
    all_row = next(row for row in payload["suite_rows"] if row["suite"] == "__all__")
    spatial_row = next(row for row in payload["suite_rows"] if row["suite"] == "libero_spatial")
    assert all_row["tasks"] == 2
    assert all_row["rollout_pools"] == 4
    assert all_row["promoted_scorer"] == "learned_wam"
    assert all_row["mean_learned_wam_minus_random_N8"] == 0.3
    assert spatial_row["worst_task"] == "libero_spatial/0"
    assert out_csv.exists()
    table = table_path.read_text(encoding="utf-8")
    assert "All configured" in table
    assert "CI lower / worst" in table
    assert "promoted learned scorer minus random" in table
    assert "not real-robot validation" in table
    assert "not VLA-scale SOTA" in table
    assert "not solved-policy success" in table
