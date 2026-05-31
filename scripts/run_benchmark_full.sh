#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

if command -v python.exe >/dev/null 2>&1; then
  PY=(python.exe)
elif command -v python >/dev/null 2>&1 && python -m pytest --version >/dev/null 2>&1; then
  PY=(python)
elif command -v py >/dev/null 2>&1; then
  PY=(py -3)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  echo "No Python interpreter found on PATH" >&2
  exit 127
fi

bash scripts/run_benchmark_smoke.sh
"${PY[@]}" experiments/benchmark_gym_manip_suite.py
"${PY[@]}" experiments/benchmark_gym_robotics_suite.py
"${PY[@]}" experiments/benchmark_metaworld_suite.py --closed-loop
"${PY[@]}" experiments/benchmark_robosuite_suite.py --closed-loop
"${PY[@]}" experiments/benchmark_maniskill_suite.py --closed-loop
bash scripts/run_benchmark_visual_optional.sh
