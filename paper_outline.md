# Paper Outline

## 1. Introduction

Robotic planners increasingly use world-action models to sample candidate futures at test time. This paper asks a narrower question than WAM training: given a fixed rollout generator and scorer, how much real utility should additional imagined rollouts buy?

## 2. Related Work

Cover best-of-N and test-time compute, world models and world-action models, random shooting and CEM-style MPC, model-based RL, verifier/scorer alignment, and planner exploitation under model error.

## 3. Problem Setup

Define state, rollout/action sequence, imagined dynamics, true dynamics, rollout score `S`, binary success `Y`, real utility `R`, and best-of-N top-score selection. Separate the exact known-distribution law from finite pilot estimation.

## 4. Binary Rollout Inference Theorem

State the finite empirical tie-aware law, the continuous-distribution form, `N=1`, all-success/no-success edge cases, and random tie handling. Prove the `N=2` AUC identity:

```text
f_2 = p^2 + 2p(1-p)kappa
```

Then show the moment hierarchy `f_N = Np E[U^(N-1)]`, where `U = F_mix(S+)`, and explain why AUC is insufficient for `N > 2`.

## 5. Utility-Valued Inference Theorem

Replace binary success with real utility:

```text
V_N(x) = N E[R F_S(S)^(N-1)]
```

Use the finite tie-aware group formula for implementation. Binary success is a special case.

## 6. Receding-Horizon Corollary

At each visited state, the theorem applies conditionally to the rollout distribution sampled at that state. The controller samples `N` rollouts, chooses the top-scoring rollout, executes the first action, observes the next state, and replans. Do not claim a global closed-form law under arbitrary nonstationarity.

## 7. Pilot Estimation And Adaptive Allocation

Discuss finite pilot estimation, bootstrap uncertainty, pilot regret, fixed-budget rollout allocation across states, moment-law allocation, utility-valued marginal gains, and comparison to uniform and oracle allocation.

## 8. Inference-Value Audits

Define the inference-value profile `N -> V_N` as the measurable object exposed by the theorem. Introduce tail alignment, imagined-vs-real tail gaps, marginal stop rules, deployment gates, and pilot-calibrated scorer repair. This section is the bridge from an exact law to a practical pre-deployment audit: decide whether to sample more, stop early, repair the scorer, or block high-N execution.

## 9. Experimental Setup

Analytic and learned toy environments:

- BlockPush2D
- DrawerPull1D
- SlipperyGrasp1D
- NonstationaryPhysicalShiftEnv
- DeformableToyEnv

Backends:

- analytic nominal WAM
- learned horizon WAM
- learned MLP dynamics WAM
- learned ensemble WAM
- oracle true dynamics

Scorers include random, predicted distance, predicted utility, predicted success, uncertainty-penalized utility, safety-penalized utility, learned scorers, oracle utility, and anti-real-utility falsification.

## 10. Results

Report only artifact-backed claims:

1. exact finite law validation
2. `N=2` AUC identity and high-N moment hierarchy
3. pilot-to-heldout prediction
4. score-function comparison
5. real-vs-imagined utility gap
6. adaptive allocation
7. closed-loop receding-horizon evaluation
8. nonstationary dynamics stress test
9. bad-scorer falsification
10. learned-vs-analytic-vs-oracle WAM comparison
11. multi-environment breadth
12. optional visual toy stress test
13. inference-value audit profiles and deployment gates
14. pilot-calibrated scorer repair
15. rollout compute-quality frontier

## 11. Benchmark Status

Gymnasium/MuJoCo Reacher-v5 is integrated as a state-based external benchmark fallback with rollout pools, learned WAM-lite training, exact-law validation, score comparison, real-vs-predicted gap, closed-loop evaluation, and RGB WAM-lite artifacts.

Gymnasium Robotics Fetch is integrated on `FetchReach-v4`, `FetchPush-v4`, and `FetchPickAndPlace-v4`, adding contact-rich MuJoCo manipulation evidence with state/action-sequence WAM-lite training, exact-law validation, score comparison, closed-loop learned-versus-random evaluation, RGB-frame/action-sequence visual WAM-lite validation, and RGB frame artifacts.

Meta-World ML1 is integrated on `reach-v3`, `push-v3`, and `drawer-open-v3`, adding an independent Sawyer manipulation benchmark with learned state/action-sequence WAM-lite training, exact-law validation, scorer comparison, and small closed-loop traces. Artifact-backed Meta-World claims are open-loop; the current closed-loop Meta-World deltas are reported but not promoted because their CIs cross zero.

RoboSuite is integrated on Panda `Lift`, `Stack`, and `Door`, adding an independent MuJoCo manipulation benchmark with clone-restored rollout pools, learned state/action-sequence WAM-lite training, exact-law validation, open-loop scorer comparison, and small closed-loop learned/reward-versus-random evaluation.

ManiSkill3 is integrated in CPU state mode on `PickCube-v1`, `PushCube-v1`, and `PegInsertionSide-v1` with `pd_joint_delta_pos` control. Artifact-backed ManiSkill claims include rollout pools, exact-law validation, score comparison, WAM-lite training, and a small closed-loop learned-versus-random comparison. End-effector delta-pose control is not claimed because Pinocchio was unavailable in this Windows environment; a generated probe records the failed EE-control attempts.

RoboCasa is integrated as an optional contact-rich kitchen artifact in a separate RoboCasa-compatible virtual environment with official kitchen assets. The single-task layer uses `robocasa/PickPlaceCounterToCabinet`, five clone-restored rollout pools with 80 random action-sequence rollouts, exact-law utility MAE `0.00027`, oracle-minus-random N8 CI lower `0.1515`, and a lightweight ridge state/action-sequence WAM-lite with validation utility correlation `0.764` and learned-minus-random N8 CI lower `0.0503`. The pick-place-family layer is a three-task task conditioned WAM-lite over `PickPlaceCounterToCabinet`, `PickPlaceCounterToDrawer`, and `PickPlaceCounterToMicrowave`: 144 train rollouts, 96 validation rollouts, validation utility correlation `0.675`, 240 heldout eval rollouts, exact-law MAE `0.00035`, and learned energy-regularized scorer minus random N8 CI lower `0.169`. The broad atomic-manipulation layer covers `OpenDrawer`, `OpenCabinet`, `OpenMicrowave`, and `TurnOnSinkFaucet`: 64 train rollouts, 32 validation rollouts, validation utility correlation `0.860`, 128 heldout eval rollouts across 16 rollout pools, exact-law MAE `0.00036`, and learned scorer minus random N8 CI lower `0.211`. The 12-task family layer covers open/close drawer, cabinet, microwave, oven plus sink/stove on/off tasks: 96 train rollouts, 96 validation rollouts, validation utility correlation `0.833`, 192 heldout eval rollouts across 24 rollout pools, exact-law MAE `0.00029`, and learned energy-regularized scorer minus random N8 CI lower `0.183`. These support RoboCasa rollout-pool dense-utility results across pick-place and non-pick-place task families, not full RoboCasa-wide validation or solved policies.

LIBERO is integrated as an optional separate-environment artifact over three `libero_spatial` tasks. The artifact trains a ridge state/action-sequence WAM-lite on 192 rollout samples, validates on 96 samples with utility correlation `0.353`, evaluates on 240 heldout rollout samples from 15 rollout pools, has exact-law utility MAE `0.00014`, and the promoted learned energy-regularized scorer beats random at N8 with CI lower `0.265`. This supports LIBERO rollout-pool dense-utility validation, not solved-task LIBERO policy performance.

A separate ManiSkill dependency probe records that Pinocchio is not importable and that pip does not expose binary `pin`/`cmeel-boost` wheels for this Windows/Python stack. This supports the limitation statement; it is not evidence for EE-control validation.

Benchmark RGB WAM-lite validation is integrated on Gymnasium/MuJoCo `Reacher-v5` and Gymnasium Robotics Fetch, where RGB rendering works. The model consumes rendered RGB frame features and action sequences, predicts rollout utility, and is evaluated with exact-law and best-of-N scorer-comparison artifacts. ManiSkill RGB/RGB-D WAM validation is not claimed because the local SAPIEN/Vulkan renderer failed during RGB observation creation; the exact probe matrix and errors are saved in `results/benchmark_maniskill_visual_probe.json` and `reports/maniskill_visual_blocker_report.md`. RoboCasa now includes pick-place-family, broad atomic-manipulation, and 12-task family rollout-pool artifacts; LIBERO remains limited to three Spatial dense rollout-pool tasks until solved-task policy artifacts exist.

## 12. Falsification And Limitations

Emphasize that more imagination helps only when scores align with real utility. Under model mismatch or a bad scorer, best-of-N can amplify hallucinated or unsafe futures. The paper does not claim real robot evidence, DreamZero/UWM-level integration, or a universal WAM training recipe.

## 13. Future Work

Robot Chinchilla: a universal WAM train-inference optimizer that jointly selects data scale, model class, rollout horizon, scorer quality, safety constraints, and test-time sampling budget.
