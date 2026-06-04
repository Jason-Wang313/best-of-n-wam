from pathlib import Path

import numpy as np

from wam_inference_value.model_artifact_integrity import audit_model_artifacts


def test_model_artifact_integrity_accepts_loadable_npz(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    np.savez(models / "model.npz", weights=np.array([[1.0, 2.0]]), bias=np.array([0.5]))

    payload = audit_model_artifacts(tmp_path, models)
    record = payload["records"][0]

    assert record["load_ok"] is True
    assert record["all_finite"] is True
    assert record["n_arrays"] == 2


def test_model_artifact_integrity_detects_nonfinite_npz(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    np.savez(models / "bad.npz", weights=np.array([1.0, np.nan]))

    payload = audit_model_artifacts(tmp_path, models)

    assert "npz_model_arrays_finite" in {issue["name"] for issue in payload["issues"]}
    assert payload["npz_nonfinite"] == ["models/bad.npz"]


def test_model_artifact_integrity_detects_unreadable_npz(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "bad.npz").write_text("not an npz", encoding="utf-8")

    payload = audit_model_artifacts(tmp_path, models)

    assert "model_artifacts_load" in {issue["name"] for issue in payload["issues"]}
    assert payload["unreadable_models"] == ["models/bad.npz"]
