from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import v6_frozen_evidence, v6_real_benchmark_evidence


OUT = ROOT / "results" / "v6_smoke"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    tracemalloc.start()
    evidence = v6_real_benchmark_evidence.run(output_root=ROOT, source_root=ROOT, smoke=True)
    frozen = v6_frozen_evidence.run(output_root=ROOT, source_root=ROOT, smoke=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "elapsed_seconds": time.perf_counter() - start,
        "peak_python_heap_mb": peak / (1024 * 1024),
        "gate_passed": bool(evidence["gate_passed"] and frozen["gate_passed"]),
        "families": evidence["real_benchmark_audit"]["family_count"],
        "pools": evidence["real_benchmark_audit"]["pool_count"],
        "decision_accuracy": evidence["real_benchmark_audit"]["decision_accuracy"],
        "false_allow_rate": evidence["real_benchmark_audit"]["false_allow_rate"],
        "low_ram_design": evidence["low_ram_design"],
        "note": "tracemalloc reports Python heap, not full process RSS.",
    }
    write_json(OUT / "v6_reproduction_report.json", report)
    print(
        "v6 cpu reproduction complete: "
        f"gate={report['gate_passed']} "
        f"families={report['families']} pools={report['pools']} "
        f"peak_heap_mb={report['peak_python_heap_mb']:.2f}"
    )
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
