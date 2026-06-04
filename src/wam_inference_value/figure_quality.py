from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np


EXPECTED_FIGURE_NAMES = {
    "exp1_exact_vs_mc_success.png",
    "exp1_exact_vs_mc_success_learned.png",
    "exp2_auc_vs_moment_error.png",
    "exp3_pilot_to_heldout_mae.png",
    "exp4_score_function_curves.png",
    "exp4_score_function_curves_learned.png",
    "exp5_imagined_vs_real_gap.png",
    "exp5_imagined_vs_real_gap_learned.png",
    "exp6_adaptive_allocation.png",
    "exp6_adaptive_allocation_learned.png",
    "exp7_closed_loop_success.png",
    "exp7_closed_loop_success_learned.png",
    "exp8_nonstationary_shift.png",
    "exp10_falsification_bad_scorer.png",
    "learned_wam_vs_analytic_wam.png",
    "multi_env_inference_curves.png",
    "benchmark_maniskill_curves.png",
    "benchmark_gym_robotics_curves.png",
    "benchmark_metaworld_curves.png",
    "benchmark_robosuite_curves.png",
    "benchmark_visual_wam_lite_curves.png",
    "benchmark_gym_robotics_visual_wam_curves.png",
    "inference_audit_tail_alignment.png",
    "inference_audit_tail_alignment_learned.png",
    "imagination_scaling_frontier.png",
}


@dataclass(frozen=True)
class FigureQualityCheck:
    name: str
    ok: bool
    detail: str


def add(checks: list[FigureQualityCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(FigureQualityCheck(name=name, ok=bool(ok), detail=detail))


def scan_pngs(results_dir: Path) -> list[Path]:
    figures_dir = results_dir / "figures"
    if not figures_dir.exists():
        return []
    return sorted(path for path in figures_dir.glob("*.png") if path.is_file())


def _rgb_array(path: Path) -> np.ndarray:
    array = np.asarray(mpimg.imread(path))
    if array.ndim == 2:
        return array[..., None]
    if array.ndim == 3 and array.shape[-1] >= 3:
        return array[..., :3]
    raise ValueError(f"unsupported image shape {array.shape}")


def image_quality_record(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "readable": False,
        "ok": False,
    }
    try:
        rgb = _rgb_array(path).astype(float)
    except Exception as exc:
        record["detail"] = f"read_error={type(exc).__name__}: {exc}"
        return record

    height, width = rgb.shape[:2]
    if rgb.max(initial=0.0) > 1.0:
        normalized = rgb / 255.0
    else:
        normalized = rgb
    nonwhite_mask = np.any(np.abs(normalized - 1.0) > 0.02, axis=-1) if normalized.ndim == 3 else np.abs(normalized - 1.0) > 0.02
    record.update(
        {
            "readable": True,
            "width": int(width),
            "height": int(height),
            "pixel_std": float(np.std(normalized)),
            "pixel_span": float(np.max(normalized) - np.min(normalized)),
            "nonwhite_fraction": float(np.mean(nonwhite_mask)),
        }
    )
    return record


def audit_figure_quality(
    root: Path,
    results_dir: Path | None = None,
    *,
    expected_figure_names: set[str] | None = None,
    min_figures: int = 30,
    min_width: int = 300,
    min_height: int = 300,
    min_bytes: int = 1000,
    min_pixel_std: float = 0.02,
    min_pixel_span: float = 0.10,
    min_nonwhite_fraction: float = 0.02,
) -> dict[str, Any]:
    root = root.resolve()
    results_dir = (results_dir or root / "results").resolve()
    checks: list[FigureQualityCheck] = []
    paths = scan_pngs(results_dir)
    records = [image_quality_record(path, root) for path in paths]

    expected_names = EXPECTED_FIGURE_NAMES if expected_figure_names is None else expected_figure_names
    missing_expected = sorted(expected_names - {path.name for path in paths})
    unreadable = [record["path"] for record in records if not record.get("readable")]
    too_small = [record["path"] for record in records if int(record.get("width") or 0) < min_width or int(record.get("height") or 0) < min_height]
    tiny_files = [record["path"] for record in records if int(record.get("bytes") or 0) < min_bytes]
    low_variance = [record["path"] for record in records if float(record.get("pixel_std") or 0.0) < min_pixel_std]
    low_span = [record["path"] for record in records if float(record.get("pixel_span") or 0.0) < min_pixel_span]
    blankish = [record["path"] for record in records if float(record.get("nonwhite_fraction") or 0.0) < min_nonwhite_fraction]

    add(checks, "figure_files_present", len(paths) >= min_figures, f"figures={len(paths)}, required={min_figures}")
    add(checks, "expected_publication_figures_present", not missing_expected, f"missing={missing_expected}")
    add(checks, "figure_files_readable", not unreadable, f"unreadable={unreadable[:10]}, count={len(unreadable)}")
    add(checks, "figure_dimensions_publication_usable", not too_small, f"too_small={too_small[:10]}, count={len(too_small)}")
    add(checks, "figure_files_nontrivial_size", not tiny_files, f"tiny={tiny_files[:10]}, count={len(tiny_files)}")
    add(checks, "figure_pixel_variance_nonblank", not low_variance, f"low_variance={low_variance[:10]}, count={len(low_variance)}")
    add(checks, "figure_pixel_span_nonflat", not low_span, f"low_span={low_span[:10]}, count={len(low_span)}")
    add(checks, "figure_nonwhite_content_present", not blankish, f"blankish={blankish[:10]}, count={len(blankish)}")

    issues = [check for check in checks if not check.ok]
    return {
        "experiment": "figure_quality",
        "verified": len(issues) == 0,
        "results_dir": str(results_dir),
        "n_figures": len(paths),
        "n_expected_figures": len(expected_names),
        "n_checks": len(checks),
        "n_issues": len(issues),
        "thresholds": {
            "min_figures": min_figures,
            "min_width": min_width,
            "min_height": min_height,
            "min_bytes": min_bytes,
            "min_pixel_std": min_pixel_std,
            "min_pixel_span": min_pixel_span,
            "min_nonwhite_fraction": min_nonwhite_fraction,
        },
        "figure_records": records,
        "checks": [check.__dict__ for check in checks],
        "issues": [check.__dict__ for check in issues],
        "missing_expected_figures": missing_expected,
        "unreadable_figures": unreadable,
        "too_small_figures": too_small,
        "tiny_figure_files": tiny_files,
        "low_variance_figures": low_variance,
        "low_span_figures": low_span,
        "blankish_figures": blankish,
    }


def figure_quality_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Figure Quality Report",
        "",
        f"- Verified: {payload.get('verified')}",
        f"- Figures audited: {payload.get('n_figures')}",
        f"- Expected publication figures: {payload.get('n_expected_figures')}",
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
        lines.append("Canonical PNG figures are present, readable, nonblank, nonflat, and large enough for publication-style inspection.")
    lines.append("")
    return "\n".join(lines)
