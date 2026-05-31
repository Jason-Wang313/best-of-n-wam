# Benchmark Blocker Report

External benchmark integration was attempted.

## Status
- maniskill: available=True reason=available with state-mode joint-delta control
- gym_manip: available=True reason=Reacher-v5 available
- gym_robotics: available=True reason=FetchPush-v4 available
- metaworld: available=True reason=reach-v3 available
- robosuite: available=True reason=Lift/Panda available
- libero: available=False reason=libero import not found
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

## Separate RoboCasa Three-Task Learned-WAM Artifact

A task conditioned ridge state/action-sequence WAM-lite was trained across `3` RoboCasa task IDs with `144` train rollouts and `240` heldout eval rollouts.
Validation utility correlation is `0.6751791364461345`; promoted scorer `learned_energy_regularized` has learned-minus-random N8 CI lower bound `0.1690501789606415`.
This supports a three-task RoboCasa pick-place family artifact, not full RoboCasa-wide validation.

## Separate RoboCasa Broad Task Family Learned-WAM Artifact

A task conditioned ridge state/action-sequence WAM-lite was trained across `4` non-pick-place RoboCasa task IDs with `64` train rollouts and `128` heldout eval rollouts.
Validation utility correlation is `0.8598503519742565`; promoted scorer `learned_wam` has learned-minus-random N8 CI lower bound `0.211154701538597`.
This supports broad RoboCasa rollout-pool dense-utility validation across atomic kitchen manipulation tasks, not full RoboCasa-wide validation or solved-policy performance.

## Separate RoboCasa 12-Task Family Learned-WAM Artifact

A task conditioned ridge state/action-sequence WAM-lite was trained across `12` RoboCasa open/close/turn task IDs with `96` train rollouts and `192` heldout eval rollouts.
Validation utility correlation is `0.8330260116378324`; promoted scorer `learned_energy_regularized` has learned-minus-random N8 CI lower bound `0.18304368049296985`.
This supports a wider RoboCasa task family rollout-pool dense-utility artifact, not full RoboCasa-wide validation or solved-policy performance.

## Separate LIBERO Three-Task Learned-WAM Artifact

A ridge state/action-sequence WAM-lite was trained across `3` LIBERO Spatial tasks with `192` train rollout samples and `240` heldout eval rollout samples.
Validation utility correlation is `0.3526483541014925`; promoted scorer `learned_energy_regularized` has learned-minus-random N8 CI lower bound `0.2652641899613382`.
This supports LIBERO rollout-pool dense-utility validation, not solved-task LIBERO policy performance.
