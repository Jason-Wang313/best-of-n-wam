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

Smoke writes temporary-scale artifacts under `results/smoke/` by default and then checks the canonical claim gate, so a smoke run does not downgrade the committed full-result evidence.

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

The repo includes adapters for ManiSkill, Gym-style manipulation, Meta-World, RoboSuite, optional RoboCasa validation, and optional LIBERO validation under `src/wam_inference_value/benchmarks/`.

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
- RoboCasa broad four-task atomic kitchen artifact
- RoboCasa 12-task open/close/turn kitchen family artifact
- RoboCasa 24-task open/close/turn/pick-place kitchen family artifact
- RoboCasa extra four-task pick-place-direction kitchen artifact
- RoboCasa combined 28-task kitchen family artifact
- RoboCasa combined 32-task kitchen family artifact
- RoboCasa stratified 55-task kitchen artifact
- RoboCasa frontier micro-rollout probe
- RoboCasa stratified 97-task kitchen artifact
- RoboCasa residual 35-task clean/cook kitchen artifact
- RoboCasa residual clean/cook micro-rollout sweep
- LIBERO Spatial three-task rollout-pool WAM-lite artifact
- LIBERO Object sparse-success scripted policy smoke
- LIBERO Object learned action-head sparse-success smoke
- LIBERO Object time-conditioned autonomous low-dimensional BC sparse-success smoke
- LIBERO Object RGB/proprio/language BC sparse-success smoke

The Gymnasium Robotics artifacts add contact-rich Fetch manipulation tasks with state/action-sequence WAM-lite training, exact-law validation, scorer comparison, closed-loop evaluation, RGB-frame/action-sequence visual WAM-lite validation, and RGB frame artifacts. Meta-World artifacts add a separate multi-task Sawyer manipulation suite with learned state/action-sequence WAM-lite, exact-law validation, scorer comparison, and small closed-loop traces. RoboSuite artifacts add Panda Lift/Stack/Door clone-restored MuJoCo rollout pools with learned state/action-sequence WAM-lite training, exact-law validation, open-loop scorer comparison, and small closed-loop learned/reward-versus-random evaluation. The ManiSkill artifact uses CPU state observations and `pd_joint_delta_pos` control. End-effector delta-pose control is not claimed in this environment because the optional Pinocchio dependency was unavailable; the repo also includes a generated ManiSkill visual/EE-control probe so that this limitation is artifact-backed rather than anecdotal.

RoboCasa is verified in eleven optional learned-WAM layers: a single-task `PickPlaceCounterToCabinet` artifact; a three-task pick-place-family artifact; a broad four-task atomic manipulation artifact; 12-task, 24-task, combined 28-task, combined 32-task, stratified 55-task, and stratified 97-task kitchen-family artifacts; an extra four-task pick-place-direction artifact; and a residual 35-task clean/cook artifact. The three-task artifact trains on 144 rollouts, validates on 96 rollouts with utility correlation `0.675`, evaluates on 240 heldout rollouts, has exact-law MAE `0.00035`, and beats random at N8 with CI lower `0.169`. The broad, 12-task, 24-task, extra four-task, combined 28-task, combined 32-task, stratified 55-task, and stratified 97-task artifacts respectively have utility correlations `0.860`, `0.833`, `0.852`, `0.799`, `0.834`, `0.838`, `0.833`, and `0.838`, exact-law MAE at most `0.00036`, and learned-minus-random CI lower bounds `0.211`, `0.183`, `0.239`, `0.247`, `0.235`, `0.228`, `0.274`, and `0.315`. The stratified 97-task artifact evaluates 1,552 heldout rollouts across 194 rollout pools and keeps an oracle-minus-learned N8 CI lower bound of `0.0369`.

The residual 35-task clean/cook artifact trains on 140 rollouts, validates on 140 rollouts with utility correlation `0.835`, evaluates on 280 heldout rollouts across 35 rollout pools, uses horizon `1` and Nmax `4`, has exact-law utility MAE `0.00025`, the learned scorer beats random at N4 with CI lower `0.197`, and the oracle-minus-learned N4 CI lower remains `0.0137`. A separate residual clean/cook micro-rollout sweep attempted 43 task IDs, completed 41 chunks, timed out 2 chunks, found 39 runnable IDs, and found 35 nondegenerate IDs. A catalog audit finds 396 registered local task IDs, 132 task IDs covered by verified rollout-pool artifacts, 106 task IDs covered by micro-rollout probes, and 134 task IDs covered by any committed artifact. These are not full RoboCasa-wide validation or solved policies.

LIBERO is verified in five optional layers: a three-task `libero_spatial` rollout-pool WAM-lite artifact with 192 train samples, 96 validation samples, 240 heldout eval samples, exact-law utility MAE `0.00014`, validation utility correlation `0.353`, and learned energy-regularized scorer minus random N8 CI lower `0.265`; a separate all-ten `libero_object` sparse-success scripted controller smoke with 50/50 successes across 10 tasks and 5 seeds, success-rate CI `[1.0, 1.0]`; a kNN action head trained on 5,014 scripted action examples that achieves 30/30 heldout sparse successes across all 10 Object tasks, success-rate CI `[1.0, 1.0]`; a time-conditioned low-dimensional kNN behavior-cloned policy trained on 12,535 scripted action examples that achieves 50/50 heldout sparse successes across all 10 Object tasks, success-rate CI `[1.0, 1.0]`, without scripted phase labels or target-point commands at evaluation time; and an RGB/proprio/language feature-kNN behavior-cloned policy trained on 7,521 scripted action examples that achieves 30/30 heldout sparse successes across all 10 Object tasks, success-rate CI `[1.0, 1.0]`, without simulator object state, task IDs, phase labels, or target-point commands at evaluation time. These are not modern VLA policy performance, full LIBERO validation, or real-robot evidence. VERIFIED CLAIM 86. VERIFIED CLAIM 87. VERIFIED CLAIM 88. VERIFIED CLAIM 89. VERIFIED CLAIM 91. VERIFIED CLAIM 92. VERIFIED CLAIM 93. VERIFIED CLAIM 94. VERIFIED CLAIM 95. VERIFIED CLAIM 96. VERIFIED CLAIM 97. VERIFIED CLAIM 98.

```bash
bash scripts/run_benchmark_smoke.sh
bash scripts/run_benchmark_full.sh
```

Current benchmark artifacts include rollout pools, learned benchmark WAM-lite training, exact-law validation, score comparison, real-vs-predicted utility gap, closed-loop evaluation, contact-rich Gymnasium Robotics Fetch validation, Meta-World ML1 validation, RoboSuite Panda validation, optional RoboCasa single-task, three-task pick-place-family, broad four-task, 12-task family, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned-WAM validation, micro-rollout viability, and registry coverage audit, optional LIBERO three-task rollout-pool learned-WAM validation, optional LIBERO Object sparse-success scripted smoke, optional LIBERO learned action-head smoke, optional LIBERO time-conditioned low-dimensional BC smoke, optional LIBERO RGB/proprio/language BC smoke, and RGB WAM-lite validation for the Gymnasium/MuJoCo and Fetch paths.

The optional RoboCasa runs are not run by default because RoboCasa365 pins a separate MuJoCo stack and requires about 10 GB of kitchen assets. To regenerate them, set `ROBOCASA_PYTHON` to a RoboCasa-compatible interpreter and run:

```bash
ROBOCASA_PYTHON=/path/to/robocasa/python bash scripts/run_benchmark_full.sh
```

The optional LIBERO run is not run by default because LIBERO expects a separate RoboSuite 1.4 era runtime. To regenerate the committed LIBERO artifact, set `LIBERO_PYTHON` to a compatible interpreter, point `LIBERO_SOURCE_PATH` at the LIBERO source checkout if needed, set `LIBERO_CONFIG_PATH`, and run:

```bash
LIBERO_PYTHON=/path/to/libero/python LIBERO_SOURCE_PATH=/path/to/LIBERO bash scripts/run_benchmark_full.sh
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

Claims are classified as `VERIFIED`, `PARTIAL`, `UNSUPPORTED`, or `FAILED` from artifacts. The claim gate is surrounded by nineteen stricter consistency checks:

```bash
python scripts/test_inventory.py --fail-on-error
python scripts/artifact_integrity.py --fail-on-error
python scripts/result_consistency.py --fail-on-error
python scripts/raw_result_recompute.py --fail-on-error
python scripts/table_schema.py --fail-on-error
python scripts/source_manifest.py --fail-on-error
python scripts/runtime_environment.py --fail-on-error
python scripts/experiment_registry.py --fail-on-error
python scripts/artifact_manifest.py --fail-on-error
python scripts/model_artifact_integrity.py --fail-on-error
python scripts/figure_quality.py --fail-on-error
python scripts/narrative_consistency.py --fail-on-error
python scripts/script_contracts.py --fail-on-error
python scripts/claims_status.py
python scripts/claim_semantics.py --fail-on-error
python scripts/claims_status.py
python scripts/claim_evidence_quality.py --fail-on-error
python scripts/claims_status.py
python scripts/claim_ledger_integrity.py --fail-on-error
python scripts/claim_generation_consistency.py --fail-on-error
python scripts/report_generation_consistency.py --fail-on-error
python scripts/command_result_consistency.py --fail-on-error
```

The test-inventory gate collects pytest node IDs and verifies the reported test count comes from an artifact rather than hard-coded stale text. The artifact-integrity gate checks that referenced result files exist, parse, and are nonempty. The result-consistency gate checks that summary JSONs agree with canonical tables for confidence-interval sanity, row counts, seed coverage, task/environment coverage, rollout-pool counts, promoted-scorer CIs, and LIBERO success counts. The raw-recompute gate independently recomputes aggregate means, exact-law MAEs, and seed-metric confidence intervals from raw CSV artifacts. The table-schema gate checks canonical CSV headers, required family columns, finite numeric values, nonblank key cells, valid success-rate ranges, and explicit optional blanks. The source-manifest gate writes deterministic SHA-256 hashes for source, experiment, script, test, README/paper, requirements, and theory files. The runtime-environment gate records and verifies Python/platform metadata, requirement-file hashes, core package versions, optional package availability, module probes, and command probes. The experiment-registry gate verifies canonical experiment-family scripts, JSON summaries, wrapper coverage, table artifacts, and figures where expected. The artifact-manifest gate writes deterministic SHA-256 hashes for canonical scientific result JSONs, CSV tables, figures, and model files. The model-artifact gate loads committed `.npz` and `.joblib` models and verifies nonempty finite numeric arrays plus predictor availability where applicable. The figure-quality gate checks that canonical PNG figures are present, readable, nonblank, nonflat, and large enough for publication-style inspection. The narrative-consistency gate checks high-impact README and final-report numbers against the current JSON artifacts. The script-contract gate checks canonical shell scripts for required experiment steps, strict Bash mode, optional benchmark guards, and ordered verification gates. The claim-semantics gate checks that verified claim wording is backed by matching threshold semantics, such as positive CI lower bounds for "beats" claims and small errors for exact-law claims. The claim-evidence gate maps every current claim ID to source artifacts and rejects missing sources, unstructured evidence, unresolved placeholders, and CI claims without CI evidence. The claim-ledger gate checks sorted contiguous claim IDs, valid statuses, JSON/Markdown count agreement, structured evidence strings, evidence-path references, empty overclaim arrays, and no non-verified final claims. The claim-generation gate reruns `claims_status.py` and verifies the JSON and Markdown claim ledgers are byte-stable and overclaim-free. The report-generation gate reruns `write_maxout_reports.py` and verifies the generated narrative reports are byte-stable. The command-result gate checks that published final-report command lines match the current verification artifacts rather than stale run text. README and paper-outline claims are intentionally scoped to artifact-backed results. Unsupported benchmark and universal-training claims belong in future work, not in the results.

## Limitations

This is learned-toy, multi-env toy, benchmark rollout-pool validation, all-ten LIBERO Object sparse-success scripted/action-head/time-conditioned/RGB-language BC smokes, and visual WAM-lite validation on Gymnasium/MuJoCo and Fetch RGB frames, not real-robot evidence. The LIBERO WAM result is dense rollout-pool utility evidence; the LIBERO Object controller is hand scripted with object-conditioned grasp heights; the learned action-head result still uses scripted phase ordering and target points; the time-conditioned BC result uses low-dimensional simulator state plus a finite-horizon step clock; and the RGB/proprio/language BC result is a lightweight feature-kNN behavior clone, not a modern VLA policy. The repo does not solve WAM training, robot generalization, or universal train/test compute allocation. Pilot estimates are statistical objects with uncertainty; the exact theorem assumes the relevant score/utility distribution is known. Oracle scorers are diagnostic upper bounds, not deployable controllers.

## Future Work

The natural next step is a Robot Chinchilla-style WAM optimizer: jointly allocate dataset scale, model capacity, rollout horizon, scorer quality, safety constraints, and test-time rollout budget. A second high-value extension is modern VLA-style sparse-success LIBERO policy evaluation or full RoboCasa-wide validation beyond the current pick-place-family, broad atomic-manipulation, 12-task family, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook rollout-pool artifacts.
