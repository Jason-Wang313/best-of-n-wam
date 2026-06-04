from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wam_inference_value.figure_quality import audit_figure_quality


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, array)


def test_figure_quality_accepts_nonblank_png(tmp_path: Path) -> None:
    results = tmp_path / "results"
    image = np.zeros((320, 320, 3), dtype=float)
    image[40:280, 40:280, 0] = 0.8
    image[80:240, 80:240, 1] = 0.4
    _write_image(results / "figures" / "good.png", image)

    audit = audit_figure_quality(tmp_path, results, expected_figure_names=set(), min_figures=1, min_width=300, min_height=300)

    assert audit["n_figures"] == 1
    assert "figure_files_readable" not in {issue["name"] for issue in audit["issues"]}
    assert "figure_pixel_variance_nonblank" not in {issue["name"] for issue in audit["issues"]}


def test_figure_quality_detects_blank_png(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_image(results / "figures" / "blank.png", np.ones((320, 320, 3), dtype=float))

    audit = audit_figure_quality(tmp_path, results, expected_figure_names=set(), min_figures=1, min_width=300, min_height=300)

    issue_names = {issue["name"] for issue in audit["issues"]}
    assert "figure_pixel_variance_nonblank" in issue_names
    assert "figure_nonwhite_content_present" in issue_names


def test_figure_quality_detects_tiny_png(tmp_path: Path) -> None:
    results = tmp_path / "results"
    image = np.zeros((64, 64, 3), dtype=float)
    image[:, :, 2] = 0.7
    _write_image(results / "figures" / "tiny.png", image)

    audit = audit_figure_quality(tmp_path, results, expected_figure_names=set(), min_figures=1, min_width=300, min_height=300)

    assert "figure_dimensions_publication_usable" in {issue["name"] for issue in audit["issues"]}
