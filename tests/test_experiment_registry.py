from pathlib import Path

from wam_inference_value.experiment_registry import ExperimentRegistryEntry, audit_experiment_registry


def make_file(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_experiment_registry_accepts_complete_entry(tmp_path: Path) -> None:
    make_file(tmp_path / "experiments" / "demo.py", "print('ok')\n")
    make_file(tmp_path / "scripts" / "run_demo.sh", "python experiments/demo.py --tag good\n")
    make_file(tmp_path / "results" / "demo.json", '{"verified": true, "experiment": "demo"}\n')
    make_file(tmp_path / "results" / "tables" / "demo.csv", "a\n1\n")
    make_file(tmp_path / "results" / "figures" / "demo.png", "png-ish\n")
    entries = [
        ExperimentRegistryEntry(
            "demo",
            "core_analytic",
            "experiments/demo.py",
            "demo.json",
            ("scripts/run_demo.sh",),
            wrapper_snippets=("--tag good",),
            table_globs=("demo.csv",),
            figure_globs=("demo.png",),
            require_verified_true=True,
        )
    ]

    payload = audit_experiment_registry(tmp_path, tmp_path / "results", entries)

    assert payload["records"][0]["ok"] is True
    assert payload["n_failed_records"] == 0


def test_experiment_registry_detects_missing_result(tmp_path: Path) -> None:
    make_file(tmp_path / "experiments" / "demo.py", "print('ok')\n")
    make_file(tmp_path / "scripts" / "run_demo.sh", "python experiments/demo.py\n")
    entries = [ExperimentRegistryEntry("demo", "core_analytic", "experiments/demo.py", "demo.json", ("scripts/run_demo.sh",))]

    payload = audit_experiment_registry(tmp_path, tmp_path / "results", entries)

    assert payload["records"][0]["ok"] is False
    assert payload["records"][0]["result_exists"] is False


def test_experiment_registry_detects_wrapper_drift(tmp_path: Path) -> None:
    make_file(tmp_path / "experiments" / "demo.py", "print('ok')\n")
    make_file(tmp_path / "scripts" / "run_demo.sh", "python other.py\n")
    make_file(tmp_path / "results" / "demo.json", '{"experiment": "demo"}\n')
    entries = [ExperimentRegistryEntry("demo", "core_analytic", "experiments/demo.py", "demo.json", ("scripts/run_demo.sh",))]

    payload = audit_experiment_registry(tmp_path, tmp_path / "results", entries)

    assert payload["records"][0]["ok"] is False
    assert payload["records"][0]["run_scripts"][0]["missing_snippets"] == ["experiments/demo.py"]


def test_experiment_registry_detects_unverified_result(tmp_path: Path) -> None:
    make_file(tmp_path / "experiments" / "demo.py", "print('ok')\n")
    make_file(tmp_path / "scripts" / "run_demo.sh", "python experiments/demo.py\n")
    make_file(tmp_path / "results" / "demo.json", '{"verified": false, "experiment": "demo"}\n')
    entries = [ExperimentRegistryEntry("demo", "core_analytic", "experiments/demo.py", "demo.json", ("scripts/run_demo.sh",), require_verified_true=True)]

    payload = audit_experiment_registry(tmp_path, tmp_path / "results", entries)

    assert payload["records"][0]["ok"] is False
    assert payload["records"][0]["verified_field_ok"] is False
