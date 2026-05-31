# How Much Should a Robot Imagine?

Exact Test-Time Inference Laws for World-Action Planning.

This repository studies a narrow robotics question: for a fixed rollout generator and a fixed rollout scorer, how much real value should a robot expect from sampling more candidate futures at test time?

## Contribution

The core contribution is an exact, tie-aware best-of-N selection law for rollout pools. Given rollout scores `S` and real utilities `R`, the expected utility of choosing the top-scoring rollout among `N` samples is determined by the empirical score/utility distribution:

```text
sum over score tie groups g:
  mean_R_g * [(r_max_g / m)^N - ((r_min_g - 1) / m)^N]
```

Binary success is the special case `R in {0,1}`. For `N=2`, success obeys `f_2 = p^2 + 2p(1-p)kappa`; for larger `N`, AUC alone is insufficient and higher rank moments matter.

## What Problem This Solves

World-action planners often spend extra test-time compute by sampling many imagined futures, scoring them, and executing the best one. This project separates the inference-time law from the training problem: once the rollout distribution and scorer are fixed, the value of more imagination is exactly computable from the score/utility distribution.

## Inference-Value Audit Framework

The repo now treats the exact curve `N -> selected real utility` as an **inference-value profile**. The audit layer measures whether a rollout/scorer stack is helpful, saturating, unstable, or harmful as `N` grows.

Implemented diagnostics include:

- high-score tail real-utility uplift
- score/real-utility rank alignment
- imagined-vs-real tail gap
- marginal-value stop rules
- conservative deployment gates for blocking harmful high-N selection
- pilot-calibrated scorer repair
- compute-quality frontiers for rollout budget, horizon, and pool size

```bash
bash scripts/run_inference_audit.sh
```

These are artifact-backed audit claims, not claims that a bad WAM becomes good automatically. The audit can recommend “do not use high N” when the top-score tail is anti-aligned with real utility.

## Why This Is Not Just LLM Best-of-N Relabeled

The implementation uses action-sequence rollouts, true and imagined dynamics, real rollout utility, model mismatch, safety penalties, adaptive rollout budgets, and closed-loop receding-horizon execution. The text-response pipeline, logprob scoring, answer parsing, and benchmark grading machinery are absent.

## Verified Theorem Layer

Implemented in `src/wam_inference_value/theorem.py` and documented in `docs/theory.md`:

- finite empirical binary best-of-N law with random tie handling
- continuous binary form and `N=1`, all-success, and no-success edge cases
- exact `N=2` AUC identity
- high-N moment hierarchy
- utility-valued theorem
- receding-horizon conditional corollary
- model-error amplification corollary
- adaptive allocation theorem
- pilot-estimation notes
- safety/risk scoring extension

## Toy Analytic Validation

The canonical CPU toy pipeline validates the exact law, AUC identity, moment hierarchy, pilot-to-heldout prediction, scorer comparison, real-vs-imagined utility gaps, adaptive allocation, closed-loop replanning, and nonstationary conditional-law stress tests.

```bash
bash scripts/run_smoke.sh
```

For the larger analytic-only run:

```bash
bash scripts/run_all.sh
```

## Learned WAM-Lite Validation

The learned toy pipeline trains a CPU NumPy dynamics model on ID BlockPush2D dynamics, evaluates ID and OOD error, and reruns the key learned-backend experiments with five seeds:

```bash
bash scripts/run_learned_wam_toy.sh
```

Current learned BlockPush2D artifacts report validation final-position L2 MAE `0.1117`, validation utility MAE `0.8624`, and OOD utility errors for severe, stuck/slip, and nonstationary regimes.

## Multi-Environment Validation

The max-out multi-env suite adds:

- `BlockPush2D`
- `DrawerPull1D`
- `SlipperyGrasp1D`
- `NonstationaryPhysicalShiftEnv`
- `DeformableToyEnv`

and trains/evaluates:

- analytic nominal WAM
- learned horizon WAM
- learned MLP dynamics WAM
- learned ensemble WAM
- oracle true dynamics

```bash
bash scripts/run_multi_env.sh
```

The current artifact uses 5 environments, 3 learned backbones, 5 seeds, and `N in {1,2,4,8,16,32,64}`. It writes JSON, CSV tables, and figures under `results/`.

## Real-vs-Imagined Utility Gap

The project explicitly tests the failure mode where increasing `N` improves imagined utility while real utility saturates or worsens under mismatch. This is a supported toy and learned-toy claim, not a real-robot claim.

## Adaptive Rollout Allocation

The allocation experiments compare uniform rollout budgets against moment-law/adaptive allocation and oracle allocation under fixed total rollout budget. Supported claims require confidence intervals from generated artifacts.

## Closed-Loop Receding-Horizon Evaluation

Closed-loop experiments execute the first action from the selected rollout, observe the next state, and replan. The theorem is claimed conditionally per visited state; the repo does not claim a global closed-form law for arbitrary nonstationary closed-loop robotics.

## Falsification Tests

`EXP10` verifies that a bad anti-real-utility scorer can make high-N selection worse. This is important: the theorem predicts selection value for the chosen scorer/distribution, but it does not make a bad scorer good.

## Scorer Repair And Compute Frontiers

The audit suite includes a heldout scorer-repair experiment. A small pilot set with real utility labels calibrates a scorer from imagined features, then evaluates high-N selection on heldout rollouts. The scaling experiment varies `N`, rollout horizon, and candidate-pool size to measure the compute-quality frontier rather than assuming more imagination is always worthwhile.

## Benchmark Validation

The repo includes adapters for ManiSkill, Gym-style manipulation, Meta-World, RoboSuite, and optional RoboCasa validation, plus a future-facing LIBERO skeleton under `src/wam_inference_value/benchmarks/`.

Current benchmark artifacts include:

- Gymnasium/MuJoCo `Reacher-v5`
- Gymnasium Robotics `FetchReach-v4`
- Gymnasium Robotics `FetchPush-v4`
- Gymnasium Robotics `FetchPickAndPlace-v4`
- Meta-World ML1 `reach-v3`
- Meta-World ML1 `push-v3`
- Meta-World ML1 `drawer-open-v3`
- RoboSuite Panda `Lift`
- RoboSuite Panda `Stack`
- RoboSuite Panda `Door`
- ManiSkill3 state-mode `PickCube-v1`
- ManiSkill3 state-mode `PushCube-v1`
- ManiSkill3 state-mode `PegInsertionSide-v1`
- RoboCasa three-task pick-place family kitchen artifact

The Gymnasium Robotics artifacts add contact-rich Fetch manipulation tasks with state/action-sequence WAM-lite training, exact-law validation, scorer comparison, closed-loop evaluation, RGB-frame/action-sequence visual WAM-lite validation, and RGB frame artifacts. Meta-World artifacts add a separate multi-task Sawyer manipulation suite with learned state/action-sequence WAM-lite, exact-law validation, scorer comparison, and small closed-loop traces. RoboSuite artifacts add Panda Lift/Stack/Door clone-restored MuJoCo rollout pools with learned state/action-sequence WAM-lite, exact-law validation, open-loop scorer comparison, and small closed-loop learned/reward-versus-random evaluation. The ManiSkill artifact uses CPU state observations and `pd_joint_delta_pos` control. End-effector delta-pose control is not claimed in this environment because the optional Pinocchio dependency was unavailable; the repo also includes a generated ManiSkill visual/EE-control probe so that this limitation is artifact-backed rather than anecdotal. RoboCasa is verified in two optional layers: a single-task `PickPlaceCounterToCabinet` artifact with exact-law MAE `0.00023` and learned-minus-random N8 CI lower `0.0503`, plus a three-task task conditioned artifact over `PickPlaceCounterToCabinet`, `PickPlaceCounterToDrawer`, and `PickPlaceCounterToMicrowave`. The three-task artifact trains a ridge state/action-sequence WAM-lite on 144 rollouts, validates on 96 rollouts with utility correlation `0.675`, evaluates on 240 heldout rollouts, has exact-law MAE `0.00035`, and the promoted learned energy-regularized scorer beats random at N8 with CI lower `0.169`. This is a three-task pick-place family result, not full RoboCasa-wide validation. LIBERO remains future work unless its dependencies and artifacts are added.

```bash
bash scripts/run_benchmark_smoke.sh
bash scripts/run_benchmark_full.sh
```

Current benchmark artifacts include rollout pools, learned benchmark WAM-lite training, exact-law validation, score comparison, real-vs-predicted utility gap, closed-loop evaluation, contact-rich Gymnasium Robotics Fetch validation, Meta-World ML1 validation, RoboSuite Panda validation, optional RoboCasa single-task and three-task learned-WAM validation, and RGB WAM-lite validation for the Gymnasium/MuJoCo and Fetch paths.

The optional RoboCasa runs are not run by default because RoboCasa365 pins a separate MuJoCo stack and requires about 10 GB of kitchen assets. To regenerate them, set `ROBOCASA_PYTHON` to a RoboCasa-compatible interpreter and run:

```bash
ROBOCASA_PYTHON=/path/to/robocasa/python bash scripts/run_benchmark_full.sh
```

## Visual Modes

The toy visual mode renders low-resolution toy states and trains/evaluates a lightweight visual utility predictor. The benchmark visual mode trains RGB-frame/action-sequence WAM-lite models on Gymnasium/MuJoCo `Reacher-v5` and Gymnasium Robotics Fetch frames, then evaluates exact-law and scorer-comparison claims on heldout rollout pools.

```bash
bash scripts/run_visual_optional.sh
bash scripts/run_benchmark_visual_optional.sh
```

Current benchmark visual artifacts use rendered RGB frames plus candidate action sequences. ManiSkill RGB/RGB-D visual WAM validation is not claimed because the local SAPIEN/Vulkan renderer failed with a descriptor-pool error; ManiSkill evidence remains state-mode.

The optional ManiSkill visual probe is:

```bash
python experiments/benchmark_maniskill_visual_probe.py
```

It writes `results/benchmark_maniskill_visual_probe.json`, `results/tables/benchmark_maniskill_visual_probe.csv`, and `reports/maniskill_visual_blocker_report.md`.

The optional ManiSkill dependency probe is:

```bash
python experiments/benchmark_maniskill_dependency_probe.py --attempt-source-install
```

It writes `results/benchmark_maniskill_dependency_probe.json` and `reports/maniskill_dependency_blocker_report.md`. The committed artifact documents that Pinocchio is not importable on this Windows/Python stack and that pip did not expose binary `pin`/`cmeel-boost` wheels, so ManiSkill end-effector-control validation is not claimed.

## Full Max-Out Run

```bash
bash scripts/run_maxout_all.sh
```

This runs tests, smoke, learned WAM toy, multi-env, benchmark/visual attempts, inference audit experiments, report generation, and the claim gate.

## Claims Status

```bash
python scripts/claims_status.py
```

Claims are classified as `VERIFIED`, `PARTIAL`, `UNSUPPORTED`, or `FAILED` from artifacts. README and paper-outline claims are intentionally scoped to artifact-backed results. Unsupported benchmark and universal-training claims belong in future work, not in the results.

## Limitations

This is learned-toy and multi-env toy validation, not real-robot evidence. It does not solve WAM training, robot generalization, or universal train/test compute allocation. Pilot estimates are statistical objects with uncertainty; the exact theorem assumes the relevant score/utility distribution is known. Oracle scorers are diagnostic upper bounds, not deployable controllers.

## Future Work

The natural next step is a Robot Chinchilla-style WAM optimizer: jointly allocate dataset scale, model capacity, rollout horizon, scorer quality, safety constraints, and test-time rollout budget. A second high-value extension is LIBERO or broader RoboCasa validation beyond the current three-task pick-place family artifact.
