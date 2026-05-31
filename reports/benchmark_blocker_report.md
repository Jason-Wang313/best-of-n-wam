# Benchmark Blocker Report

External benchmark integration was attempted.

## Status
- maniskill: available=True reason=available with state-mode joint-delta control
- gym_manip: available=True reason=Reacher-v5 available
- gym_robotics: available=True reason=FetchPush-v4 available
- metaworld: available=True reason=reach-v3 available
- robosuite: available=True reason=Lift/Panda available
- libero: available=False reason=libero import not found; local pip install failed while building hf-egl-probe/egl_probe on Windows
- robocasa: available=False reason=adapter skeleton only; dependency not installed/validated

## Current Outcome

At least one optional external benchmark path is available. Run `bash scripts/run_benchmark_full.sh` to generate benchmark artifacts.

## Remaining Blockers

Remaining unavailable adapters: libero, robocasa.
