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
  "$ROBOCASA_PYTHON" experiments/benchmark_robocasa_multitask_wam.py --output-tag broad --env-ids robocasa/OpenDrawer robocasa/OpenCabinet robocasa/OpenMicrowave robocasa/TurnOnSinkFaucet --train-states 2 --train-rollouts 8 --val-states 1 --val-rollouts 8 --eval-states 4 --eval-rollouts 8 --horizon 2 --mc-trials 1000 --min-tasks 4 --min-eval-pools 16 --max-exact-mae 0.03
  "$ROBOCASA_PYTHON" experiments/benchmark_robocasa_multitask_wam.py --output-tag family12 --env-ids robocasa/OpenDrawer robocasa/OpenCabinet robocasa/OpenMicrowave robocasa/TurnOnSinkFaucet robocasa/CloseDrawer robocasa/CloseCabinet robocasa/CloseMicrowave robocasa/TurnOffSinkFaucet robocasa/TurnOnStove robocasa/TurnOffStove robocasa/OpenOven robocasa/CloseOven --train-states 1 --train-rollouts 8 --val-states 1 --val-rollouts 8 --eval-states 2 --eval-rollouts 8 --horizon 2 --mc-trials 1000 --min-tasks 12 --min-eval-pools 24 --max-exact-mae 0.03
  "$ROBOCASA_PYTHON" experiments/benchmark_robocasa_multitask_wam.py --output-tag family24 --env-ids robocasa/OpenDrawer robocasa/OpenCabinet robocasa/OpenMicrowave robocasa/TurnOnSinkFaucet robocasa/CloseDrawer robocasa/CloseCabinet robocasa/CloseMicrowave robocasa/TurnOffSinkFaucet robocasa/TurnOnStove robocasa/TurnOffStove robocasa/OpenOven robocasa/CloseOven robocasa/OpenDishwasher robocasa/CloseDishwasher robocasa/OpenFridge robocasa/CloseFridge robocasa/OpenFridgeDrawer robocasa/PickPlaceCounterToCabinet robocasa/PickPlaceCounterToDrawer robocasa/PickPlaceCounterToMicrowave robocasa/PickPlaceCounterToSink robocasa/PickPlaceCounterToStove robocasa/PickPlaceCounterToOven robocasa/PickPlaceCounterToBlender --train-states 1 --train-rollouts 8 --val-states 1 --val-rollouts 8 --eval-states 2 --eval-rollouts 8 --horizon 2 --mc-trials 1000 --min-tasks 24 --min-eval-pools 48 --max-exact-mae 0.03
else
  echo "Skipping optional RoboCasa smoke: set ROBOCASA_PYTHON to a RoboCasa-compatible interpreter to run it."
fi
if [[ -n "${LIBERO_PYTHON:-}" ]]; then
  if [[ -n "${LIBERO_SOURCE_PATH:-}" ]]; then
    export PYTHONPATH="${LIBERO_SOURCE_PATH}:${PYTHONPATH}"
  fi
  "$LIBERO_PYTHON" experiments/benchmark_libero_wam.py --train-states 4 --train-rollouts 16 --val-states 2 --val-rollouts 16 --eval-states 5 --eval-rollouts 16 --horizon 4 --mc-trials 1500 --min-eval-pools 5
  "$LIBERO_PYTHON" experiments/benchmark_libero_scripted_policy.py --suite libero_object --tasks 0 1 2 3 4 5 6 7 8 9 --seeds 100 101 102 103 104 --horizon 512 --bootstrap-samples 3000
  "$LIBERO_PYTHON" experiments/benchmark_libero_learned_action_head.py --tasks 0 1 2 3 4 5 6 7 8 9 --bootstrap-samples 2000 --action-head-model knn --knn-k 3 --knn-temperature 0.05
  "$LIBERO_PYTHON" experiments/benchmark_libero_autonomous_bc_policy.py --tasks 0 1 2 3 4 5 6 7 8 9 --train-seeds 100 101 102 103 104 --eval-seeds 200 201 202 203 204 --eval-steps 350 --bootstrap-samples 3000 --knn-k 3 --knn-temperature 0.05
  "$LIBERO_PYTHON" experiments/benchmark_libero_visual_language_bc_policy.py --tasks 0 1 2 3 4 5 6 7 8 9 --train-seeds 100 101 102 --eval-seeds 200 201 202 --eval-steps 350 --bootstrap-samples 3000
else
  echo "Skipping optional LIBERO WAM run: set LIBERO_PYTHON and, if needed, LIBERO_SOURCE_PATH/LIBERO_CONFIG_PATH."
fi
bash scripts/run_benchmark_visual_optional.sh
