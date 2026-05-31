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
if [[ -n "${ROBOCASA_PYTHON:-}" ]]; then
  "$ROBOCASA_PYTHON" experiments/benchmark_robocasa_smoke.py --states 5 --rollouts 16 --horizon 3 --mc-trials 2500 --closed-loop
  "$ROBOCASA_PYTHON" experiments/benchmark_robocasa_learned_wam.py --train-states 5 --train-rollouts 16 --val-states 2 --val-rollouts 16 --eval-states 5 --eval-rollouts 16 --horizon 3 --mc-trials 2500
  "$ROBOCASA_PYTHON" experiments/benchmark_robocasa_multitask_wam.py --train-states 3 --train-rollouts 16 --val-states 2 --val-rollouts 16 --eval-states 5 --eval-rollouts 16 --horizon 3 --mc-trials 1500 --min-eval-pools 15
else
  echo "Skipping optional RoboCasa smoke: set ROBOCASA_PYTHON to a RoboCasa-compatible interpreter to run it."
fi
bash scripts/run_benchmark_visual_optional.sh
