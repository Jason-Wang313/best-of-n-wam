from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from wam_inference_value.theorem import auc_kappa, n2_auc_identity, utility_best_of_n_finite


N_VALUES = [1, 2, 4, 8, 16, 32, 64]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson_or_none(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def curve_columns(curve: dict[int, float], prefix: str = "f") -> dict[str, float]:
    return {f"{prefix}{N}": float(curve[N]) for N in N_VALUES}


def run_exact_law_hardening(out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cases = [
        {
            "case": "two_point_aligned",
            "scores": [0.0, 1.0],
            "utilities": [0.0, 1.0],
            "expected": lambda N: 1.0 - 0.5**N,
        },
        {
            "case": "two_point_anti_aligned",
            "scores": [0.0, 1.0],
            "utilities": [1.0, 0.0],
            "expected": lambda N: 0.5**N,
        },
        {
            "case": "all_equal_scores",
            "scores": [0.0, 0.0, 0.0],
            "utilities": [-1.0, 2.0, 4.0],
            "expected": lambda N: 5.0 / 3.0,
        },
        {
            "case": "top_tie_group",
            "scores": [0.0, 1.0, 1.0],
            "utilities": [0.0, 1.0, 3.0],
            "expected": lambda N: 2.0 * (1.0 - (1.0 / 3.0) ** N),
        },
        {
            "case": "negative_top_tail",
            "scores": [0.0, 1.0],
            "utilities": [1.0, -2.0],
            "expected": lambda N: -2.0 + 3.0 * 0.5**N,
        },
        {
            "case": "all_success_binary",
            "scores": [0.0, 1.0, 2.0, 3.0],
            "utilities": [1.0, 1.0, 1.0, 1.0],
            "expected": lambda N: 1.0,
        },
        {
            "case": "no_success_binary",
            "scores": [0.0, 1.0, 2.0, 3.0],
            "utilities": [0.0, 0.0, 0.0, 0.0],
            "expected": lambda N: 0.0,
        },
    ]
    max_abs_error = 0.0
    for case in cases:
        curve = utility_best_of_n_finite(case["scores"], case["utilities"], N_VALUES)
        for N in N_VALUES:
            expected = float(case["expected"](N))
            actual = float(curve[N])
            abs_error = abs(actual - expected)
            max_abs_error = max(max_abs_error, abs_error)
            rows.append(
                {
                    "case": case["case"],
                    "N": N,
                    "expected": expected,
                    "actual": actual,
                    "abs_error": abs_error,
                    "pass": abs_error <= 1e-12,
                }
            )

    csv_path = out / "exact_law_hardening.csv"
    write_csv(csv_path, rows, ["case", "N", "expected", "actual", "abs_error", "pass"])
    summary = {
        "case_count": len(cases),
        "row_count": len(rows),
        "max_abs_error": max_abs_error,
        "all_passed": all(bool(row["pass"]) for row in rows),
        "artifact": str(csv_path),
    }
    write_json(out / "exact_law_hardening.json", summary)
    return summary


def success_vector(m: int, positive_ranks_1_indexed: Iterable[int]) -> np.ndarray:
    success = np.zeros(m, dtype=float)
    for rank in positive_ranks_1_indexed:
        success[int(rank) - 1] = 1.0
    return success


def run_auc_correlation_insufficiency(out: Path) -> dict[str, Any]:
    scores = np.arange(1, 9, dtype=float)
    examples = [
        ("tail_safe", success_vector(8, [1, 6, 7, 8])),
        ("tail_unsafe", success_vector(8, [4, 5, 6, 7])),
    ]
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[int, float]] = {}
    metrics: dict[str, dict[str, float | None]] = {}
    for name, success in examples:
        curve = utility_best_of_n_finite(scores, success, N_VALUES)
        curves[name] = curve
        p = float(np.mean(success))
        auc = float(auc_kappa(scores, success))
        corr = pearson_or_none(scores, success)
        metrics[name] = {
            "p": p,
            "auc": auc,
            "n2_auc_identity": n2_auc_identity(p, auc),
            "n2_finite": curve[2],
            "mean_score": float(np.mean(scores)),
            "mean_utility": float(np.mean(success)),
            "score_utility_corr": corr,
            "high_n_selected_utility": curve[64],
        }
        rows.append(
            {
                "example": name,
                "positive_ranks": ";".join(str(i + 1) for i, value in enumerate(success) if value == 1.0),
                **metrics[name],
                **curve_columns(curve),
            }
        )

    high_n_gap = abs(curves["tail_safe"][64] - curves["tail_unsafe"][64])
    n2_gap = abs(curves["tail_safe"][2] - curves["tail_unsafe"][2])
    matched = {
        key: abs(float(metrics["tail_safe"][key]) - float(metrics["tail_unsafe"][key]))
        for key in ["p", "auc", "n2_auc_identity", "n2_finite", "mean_score", "mean_utility", "score_utility_corr"]
    }
    csv_path = out / "auc_correlation_insufficiency.csv"
    write_csv(
        csv_path,
        rows,
        [
            "example",
            "positive_ranks",
            "p",
            "auc",
            "n2_auc_identity",
            "n2_finite",
            "mean_score",
            "mean_utility",
            "score_utility_corr",
            "high_n_selected_utility",
            *[f"f{N}" for N in N_VALUES],
        ],
    )
    summary = {
        "matched_metric_abs_differences": matched,
        "n2_gap": n2_gap,
        "high_n_gap": high_n_gap,
        "gate_high_n_gap_threshold": 0.5,
        "gate_passed": high_n_gap >= 0.5 and max(matched.values()) <= 1e-12,
        "artifact": str(csv_path),
    }
    write_json(out / "auc_correlation_insufficiency.json", summary)
    return summary


def score_pattern(m: int, kind: str) -> np.ndarray:
    if kind == "strict":
        return np.arange(m, dtype=float)
    if kind == "two_tier":
        return np.floor(np.arange(m, dtype=float) * 2.0 / m)
    if kind == "three_tier":
        return np.floor(np.arange(m, dtype=float) * 3.0 / m)
    raise ValueError(f"unknown score pattern: {kind}")


def classify_curve(values: list[float]) -> str:
    eps = 1e-9
    diffs = np.diff(np.asarray(values, dtype=float))
    high_gain = values[-1] - values[0]
    if max(values) - min(values) <= eps:
        return "flat"
    if np.any(diffs > eps) and np.any(diffs < -eps):
        return "nonmonotonic"
    if high_gain > 0.05:
        return "helps"
    if high_gain < -0.05:
        return "harms"
    return "saturates_or_small"


def iter_census_rows(max_m: int) -> Iterable[dict[str, Any]]:
    for m in range(4, max_m + 1):
        for pattern in ["strict", "two_tier", "three_tier"]:
            scores = score_pattern(m, pattern)
            for mask in range(1, (1 << m) - 1):
                success = np.asarray([(mask >> i) & 1 for i in range(m)], dtype=float)
                curve = utility_best_of_n_finite(scores, success, N_VALUES)
                values = [curve[N] for N in N_VALUES]
                corr = pearson_or_none(scores, success)
                auc = auc_kappa(scores, success)
                classification = classify_curve(values)
                yield {
                    "m": m,
                    "score_pattern": pattern,
                    "success_mask": mask,
                    "p": float(np.mean(success)),
                    "auc": None if not math.isfinite(auc) else float(auc),
                    "score_utility_corr": corr,
                    "classification": classification,
                    "high_n_gain": float(curve[64] - curve[1]),
                    "max_curve": float(max(values)),
                    "min_curve": float(min(values)),
                    **curve_columns(curve),
                }


def run_finite_pool_census(out: Path, max_m: int) -> dict[str, Any]:
    csv_path = out / "finite_pool_census.csv"
    fieldnames = [
        "m",
        "score_pattern",
        "success_mask",
        "p",
        "auc",
        "score_utility_corr",
        "classification",
        "high_n_gain",
        "max_curve",
        "min_curve",
        *[f"f{N}" for N in N_VALUES],
    ]
    counters: Counter[str] = Counter()
    by_m: Counter[str] = Counter()

    def counted_rows() -> Iterable[dict[str, Any]]:
        for row in iter_census_rows(max_m):
            counters[str(row["classification"])] += 1
            by_m[str(row["m"])] += 1
            yield row

    row_count = write_csv(csv_path, counted_rows(), fieldnames)
    expected = 3 * sum((1 << m) - 2 for m in range(4, max_m + 1))
    summary = {
        "max_m": max_m,
        "row_count": row_count,
        "expected_row_count": expected,
        "classification_counts": dict(sorted(counters.items())),
        "rows_by_m": dict(sorted(by_m.items(), key=lambda item: int(item[0]))),
        "counts_sum": int(sum(counters.values())),
        "gate_passed": row_count == expected and sum(counters.values()) == expected,
        "artifact": str(csv_path),
    }
    write_json(out / "finite_pool_census_summary.json", summary)
    return summary


def run_impossibility_boundary(out: Path) -> dict[str, Any]:
    observable_scores = np.asarray([0.10, 0.75, 0.95, 1.00], dtype=float)
    imagined_utility = np.asarray([0.20, 0.70, 0.90, 0.95], dtype=float)
    uncertainty = np.asarray([0.10, 0.10, 0.10, 0.10], dtype=float)
    real_safe = np.asarray([0.10, 0.45, 0.85, 0.95], dtype=float)
    real_unsafe = np.asarray([0.95, 0.80, 0.45, 0.05], dtype=float)
    rows = []
    for world, real in [("safe_tail", real_safe), ("unsafe_tail", real_unsafe)]:
        selected = int(np.argmax(observable_scores))
        curve = utility_best_of_n_finite(observable_scores, real, N_VALUES)
        rows.append(
            {
                "world": world,
                "selected_candidate_score_only": selected,
                "selected_real_utility": float(real[selected]),
                "observable_scores": ";".join(f"{x:.3f}" for x in observable_scores),
                "imagined_utility": ";".join(f"{x:.3f}" for x in imagined_utility),
                "uncertainty": ";".join(f"{x:.3f}" for x in uncertainty),
                "real_utility": ";".join(f"{x:.3f}" for x in real),
                "recommended_audit_action": "allow" if world == "safe_tail" else "block_or_request_labels",
                **curve_columns(curve),
            }
        )
    csv_path = out / "impossibility_boundary.csv"
    write_csv(
        csv_path,
        rows,
        [
            "world",
            "selected_candidate_score_only",
            "selected_real_utility",
            "observable_scores",
            "imagined_utility",
            "uncertainty",
            "real_utility",
            "recommended_audit_action",
            *[f"f{N}" for N in N_VALUES],
        ],
    )
    summary = {
        "observable_features_identical": True,
        "score_only_selected_candidate_equal": rows[0]["selected_candidate_score_only"]
        == rows[1]["selected_candidate_score_only"],
        "selected_real_utility_gap": abs(rows[0]["selected_real_utility"] - rows[1]["selected_real_utility"]),
        "gate_passed": rows[0]["selected_candidate_score_only"] == rows[1]["selected_candidate_score_only"]
        and abs(rows[0]["selected_real_utility"] - rows[1]["selected_real_utility"]) >= 0.8,
        "artifact": str(csv_path),
    }
    write_json(out / "impossibility_boundary_summary.json", summary)
    return summary


def write_scoretailbench(root: Path) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    pools_dir = root / "pools"
    splits_dir = root / "splits"
    baselines_dir = root / "baselines"
    pools_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    pool_specs = {
        "matched_auc_tail_safe": success_vector(8, [1, 6, 7, 8]),
        "matched_auc_tail_unsafe": success_vector(8, [4, 5, 6, 7]),
        "impossibility_safe_tail": np.asarray([0.10, 0.45, 0.85, 0.95], dtype=float),
        "impossibility_unsafe_tail": np.asarray([0.95, 0.80, 0.45, 0.05], dtype=float),
    }
    baseline_curves: dict[str, dict[str, float]] = {}
    manifest_pools = []
    for pool_id, real_utility in pool_specs.items():
        m = len(real_utility)
        if pool_id.startswith("matched_auc"):
            score = np.arange(1, m + 1, dtype=float)
            imagined = score / float(m)
            uncertainty = np.full(m, 0.10)
            family = "auc_insufficiency"
        else:
            score = np.asarray([0.10, 0.75, 0.95, 1.00], dtype=float)
            imagined = np.asarray([0.20, 0.70, 0.90, 0.95], dtype=float)
            uncertainty = np.full(m, 0.10)
            family = "impossibility_boundary"
        rows = [
            {
                "pool_id": pool_id,
                "family": family,
                "split": "v5_core",
                "candidate_id": idx,
                "score": float(score[idx]),
                "imagined_utility": float(imagined[idx]),
                "uncertainty": float(uncertainty[idx]),
                "real_utility": float(real_utility[idx]),
            }
            for idx in range(m)
        ]
        csv_path = pools_dir / f"{pool_id}.csv"
        write_csv(
            csv_path,
            rows,
            ["pool_id", "family", "split", "candidate_id", "score", "imagined_utility", "uncertainty", "real_utility"],
        )
        curve = utility_best_of_n_finite(score, real_utility, N_VALUES)
        baseline_curves[pool_id] = {str(N): float(curve[N]) for N in N_VALUES}
        manifest_pools.append(
            {
                "pool_id": pool_id,
                "family": family,
                "split": "v5_core",
                "candidates": m,
                "sha256": sha256(csv_path),
                "path": f"pools/{pool_id}.csv",
            }
        )

    split_payload = {
        "v5_core": {
            "description": "Tiny deterministic pools for V5 core reproduction; not the full paper benchmark.",
            "pool_ids": sorted(pool_specs),
        }
    }
    write_json(splits_dir / "v5_core.json", split_payload)
    write_json(baselines_dir / "finite_selected_utility_curves.json", {"N_values": N_VALUES, "curves": baseline_curves})
    readme = """# ScoreTailBench V5 Core

This is a tiny deterministic benchmark package generated by
`experiments/v5_core_evidence.py`. It is a low-RAM reproduction artifact for the
V5 core claims, not the full benchmark suite.

Each pool contains candidate scores, imagined utility, uncertainty, and real
utility. Baseline finite selected-utility curves are generated directly from the
same CSV files using the exact finite tied-pool law.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "name": "ScoreTailBench V5 Core",
        "version": "0.1.0",
        "scope": "tiny deterministic V5 core reproduction artifact",
        "claims_not_supported": [
            "real robot validation",
            "GPU-scale training",
            "broad robotics SOTA",
            "universal WAM training recipe",
        ],
        "N_values": N_VALUES,
        "pools": manifest_pools,
        "splits": ["splits/v5_core.json"],
        "baselines": ["baselines/finite_selected_utility_curves.json"],
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def run(output_root: Path = ROOT, *, smoke: bool = False) -> dict[str, Any]:
    out = output_root / ("results/v5_smoke" if smoke else "results/v5")
    out.mkdir(parents=True, exist_ok=True)
    max_m = 7 if smoke else 8
    exact = run_exact_law_hardening(out)
    auc = run_auc_correlation_insufficiency(out)
    census = run_finite_pool_census(out, max_m=max_m)
    impossibility = run_impossibility_boundary(out)
    bench_root = output_root / ("scoretailbench_v5_smoke" if smoke else "scoretailbench")
    bench_manifest = write_scoretailbench(bench_root)
    summary = {
        "mode": "smoke" if smoke else "canonical",
        "low_ram_design": {
            "parallel_jobs": 1,
            "streamed_census_csv": True,
            "max_census_m": max_m,
            "full_dense_tensor_materialization": False,
        },
        "exact_law_hardening": exact,
        "auc_correlation_insufficiency": auc,
        "finite_pool_census": census,
        "impossibility_boundary": impossibility,
        "scoretailbench": {
            "root": str(bench_root),
            "pool_count": len(bench_manifest["pools"]),
            "manifest": str(bench_root / "manifest.json"),
        },
        "gate_passed": all(
            [
                exact["all_passed"],
                auc["gate_passed"],
                census["gate_passed"],
                impossibility["gate_passed"],
                len(bench_manifest["pools"]) >= 4,
            ]
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="write smaller smoke artifacts under results/v5_smoke")
    args = parser.parse_args()
    summary = run(ROOT, smoke=args.smoke)
    mode = summary["mode"]
    print(
        f"v5 core evidence complete ({mode}): "
        f"census_rows={summary['finite_pool_census']['row_count']} "
        f"high_n_gap={summary['auc_correlation_insufficiency']['high_n_gap']:.3f} "
        f"gate={summary['gate_passed']}"
    )


if __name__ == "__main__":
    main()
