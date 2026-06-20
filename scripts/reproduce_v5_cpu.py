from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import v5_core_evidence, v5_prospective_evidence


def main() -> None:
    tracemalloc.start()
    start = time.perf_counter()
    core = v5_core_evidence.run(ROOT, smoke=True)
    prospective = v5_prospective_evidence.run(smoke=True, output_root=ROOT)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "command": "python scripts/reproduce_v5_cpu.py",
        "mode": "smoke",
        "elapsed_seconds": elapsed,
        "peak_python_heap_mb": peak / (1024 * 1024),
        "core_summary_path": "results/v5_smoke/summary.json",
        "prospective_summary_path": "results/v5_smoke/prospective_evidence_summary.json",
        "scoretailbench_path": "scoretailbench_v5_smoke/manifest.json",
        "core_gate_passed": bool(core["gate_passed"]),
        "prospective_gate_passed": bool(prospective["gate_passed"]),
        "gate_passed": bool(core["gate_passed"] and prospective["gate_passed"]),
        "note": "tracemalloc reports Python heap, not full process RSS.",
    }
    path = ROOT / "results" / "v5_smoke" / "v5_reproduction_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
