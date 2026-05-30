# Benchmark Blocker Report

External benchmark integration was attempted.

## Status
- maniskill: available=False reason=ManiSkill import not found
- gym_manip: available=True reason=Reacher-v5 available
- libero: available=False reason=adapter skeleton only; dependency not installed/validated
- robocasa: available=False reason=adapter skeleton only; dependency not installed/validated

## Current Outcome

At least one optional external benchmark path is available. Run `bash scripts/run_benchmark_full.sh` to generate benchmark artifacts.

## Remaining Blockers

ManiSkill, LIBERO, and RoboCasa remain unavailable unless their dependencies are installed and their adapters are completed.
