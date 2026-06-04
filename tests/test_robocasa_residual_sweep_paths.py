from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_sweep_module():
    path = ROOT / "experiments" / "benchmark_robocasa_residual_frontier_sweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_robocasa_residual_frontier_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residual_sweep_summary_tag_preserves_canonical_default() -> None:
    module = _load_sweep_module()

    canonical = module._summary_paths("")
    tagged = module._summary_paths("Simple Residual Probe")

    assert canonical["summary"].name == "benchmark_robocasa_residual_frontier_sweep.json"
    assert canonical["table"].name == "benchmark_robocasa_residual_frontier_sweep_chunks.csv"
    assert canonical["report"].name == "robocasa_residual_frontier_sweep_report.md"
    assert tagged["summary"].name == "benchmark_robocasa_residual_frontier_sweep_simple_residual_probe.json"
    assert tagged["table"].name == "benchmark_robocasa_residual_frontier_sweep_simple_residual_probe_chunks.csv"
    assert tagged["report"].name == "robocasa_residual_frontier_sweep_simple_residual_probe_report.md"
