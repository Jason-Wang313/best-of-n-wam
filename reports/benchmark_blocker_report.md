# Benchmark Blocker Report

External benchmark integration was attempted.

## Status
- maniskill: available=True reason=available with state-mode joint-delta control
- gym_manip: available=True reason=Reacher-v5 available
- gym_robotics: available=True reason=FetchPush-v4 available
- metaworld: available=True reason=reach-v3 available
- robosuite: available=True reason=Lift/Panda available
- libero: available=False reason=libero import not found; local pip install failed while building hf-egl-probe/egl_probe on Windows
- robocasa: available=False reason=robocasa import not found

## Current Outcome

At least one optional external benchmark path is available. Run `bash scripts/run_benchmark_full.sh` to generate benchmark artifacts.

## Remaining Blockers

Remaining unavailable adapters: libero, robocasa.

## Separate RoboCasa Smoke Artifact

RoboCasa is unavailable in the active Python environment but has a verified external smoke artifact: `robocasa/PickPlaceCounterToCabinet`, `80` rollouts, exact-law utility MAE `0.0002724664778796843`.
This is single task only, not full multi-task RoboCasa validation.

## Separate RoboCasa Learned-WAM Artifact

A lightweight ridge state/action-sequence WAM-lite was trained on `80` single-task RoboCasa rollouts and evaluated on `80` heldout rollouts.
Validation utility correlation is `0.7639608394479505`; learned-minus-random N8 CI lower bound is `0.05034853210127989`.
This supports only a single-task contact-rich sanity check, not a multi-task RoboCasa benchmark.
