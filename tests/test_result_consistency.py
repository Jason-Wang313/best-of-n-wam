import json

from wam_inference_value.result_consistency import (
    ConsistencyCheck,
    audit_rollout_pool_wam,
    ci_objects,
    pool_count,
)


def test_ci_objects_find_nested_confidence_intervals():
    payload = {
        "outer": {
            "confidence_intervals": {
                "gain": {"n": 5, "mean": 1.0, "lo": 0.5, "hi": 1.5, "std": 0.2}
            }
        }
    }

    found = ci_objects(payload)

    assert len(found) == 1
    assert found[0][0] == "outer.confidence_intervals.gain"
    assert found[0][1]["n"] == 5


def test_pool_count_uses_task_seed_state_keys():
    rows = [
        {"env_id": "task_a", "seed": "1", "state_id": "0", "scorer": "random"},
        {"env_id": "task_a", "seed": "1", "state_id": "0", "scorer": "oracle"},
        {"env_id": "task_a", "seed": "1", "state_id": "1", "scorer": "random"},
        {"env_id": "task_b", "seed": "1", "state_id": "0", "scorer": "random"},
    ]

    assert pool_count(rows, "env_id") == 3


def test_rollout_pool_wam_flags_summary_table_mismatch(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    curves = tmp_path / "curves.csv"
    exact = tmp_path / "exact.csv"
    data = tmp_path / "data.csv"
    eval_rows = tmp_path / "eval.csv"
    seed_metrics = tmp_path / "seed.csv"
    curves.write_text(
        "env_id,seed,state_id,scorer,N,real_utility\n"
        "task_a,1,0,random,1,0.0\n"
        "task_a,1,0,random,2,0.1\n",
        encoding="utf-8",
    )
    exact.write_text("env_id,state_id,N,exact_utility\n" "task_a,0,1,0.0\n" "task_a,0,2,0.1\n", encoding="utf-8")
    data.write_text("split,env_id,utility\ntrain,task_a,0.0\nvalidation,task_a,0.0\n", encoding="utf-8")
    eval_rows.write_text("env_id,seed,state_id,utility\n" "task_a,1,0,0.0\n", encoding="utf-8")
    seed_metrics.write_text("env_id,seed,metric\n" "task_a,1,0.0\n", encoding="utf-8")
    payload = {
        "env_id": "task_a",
        "train_samples": 1,
        "validation_samples": 1,
        "eval_samples": 2,
        "eval_states": 1,
        "n_values": [1, 2],
        "curves_path": str(curves),
        "exact_path": str(exact),
        "data_path": str(data),
        "eval_path": str(eval_rows),
        "seed_metrics_path": str(seed_metrics),
    }
    (results / "toy_wam.json").write_text(json.dumps(payload), encoding="utf-8")
    checks: list[ConsistencyCheck] = []

    audit_rollout_pool_wam(tmp_path, results, checks, "toy_wam.json", "toy")

    failures = {check.name: check.detail for check in checks if not check.ok}
    assert "toy_eval_rows" in failures
    assert failures["toy_eval_rows"] == "rows=1, json=2"
