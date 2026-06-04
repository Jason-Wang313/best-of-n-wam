# Ideal Claim Boundary Report

- Verified boundary: True
- Ideal claims audited: 9
- Promotable artifact-backed claims: 4
- Future-only non-promotable claims: 5
- Endpoint-supported claims: 4
- Unsupported future-only endpoints: 5
- All ideal claims promotable: False
- Goal completion status: incomplete_future_only_gaps_remain
- Future-only rows with promotion requirements: 5
- Future-only rows with missing-evidence classes: 5
- Future-only rows with gap evidence files: 5
- Checks: 64
- Issues: 0

## Boundary Matrix

- `exact_math_core`: status=`promotable_result`, endpoint_supported=True, boundary_evidence_present=True, promotable=True, future_only=False
- `learned_multi_env_core`: status=`promotable_result`, endpoint_supported=True, boundary_evidence_present=True, promotable=True, future_only=False
  Limitation: Toy CPU environments, not real-robot evidence.
- `contact_benchmark_state_mode`: status=`promotable_limited_scope`, endpoint_supported=True, boundary_evidence_present=True, promotable=True, future_only=False
  Limitation: State-mode and rollout-pool/short-horizon benchmark artifacts; not full benchmark-wide validation.
- `visual_observation_limited`: status=`promotable_limited_scope`, endpoint_supported=True, boundary_evidence_present=True, promotable=True, future_only=False
  Limitation: Toy/Gymnasium/Fetch RGB artifacts only; ManiSkill RGB/RGB-D is blocker-documented, not claimed.
- `real_robot_hil`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: No real-robot or hardware-in-the-loop artifact exists in this repository.
  Promotion requirements:
  - Future-only promotion requirement, not current evidence: Committed real-robot or hardware-in-the-loop rollout/control artifacts with task definitions, seeds, and success or utility metrics.
  - Future-only promotion requirement, not current evidence: A claims_status entry whose evidence points to those artifacts rather than simulator-only benchmark results.
  Missing evidence classes:
  - Missing future-only evidence class, not current evidence: Physical robot or hardware-in-the-loop execution logs.
  - Missing future-only evidence class, not current evidence: Task-level real-world success/utility metrics with seeds or trial IDs.
  - Missing future-only evidence class, not current evidence: A verified claim ledger entry sourced from real-world artifacts.
  Gap evidence files: `results/ideal_frontier_readiness.json`, `reports/ideal_frontier_readiness_report.md`, `reports/final_decision_report.md`, `reports/reviewer_risk_assessment.md`
- `modern_vla_libero`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: LIBERO artifacts are scripted/BC smokes and dense rollout-pool WAM evidence, not modern VLA performance.
  Promotion requirements:
  - Future-only promotion requirement, not current evidence: A modern VLA-style policy or policy-compatible controller evaluated on LIBERO sparse-success tasks.
  - Future-only promotion requirement, not current evidence: Heldout success metrics with confidence intervals that do not rely on scripted phase labels, target-point commands, or simulator object-state shortcuts unless explicitly scoped.
  Missing evidence classes:
  - Missing future-only evidence class, not current evidence: Modern VLA-style policy artifact evaluated as a policy, not a rollout-pool scorer or scripted/BC smoke.
  - Missing future-only evidence class, not current evidence: Sparse-success heldout LIBERO metrics with confidence intervals under the promoted observation/action interface.
  - Missing future-only evidence class, not current evidence: Evidence that evaluation-time inputs do not use shortcuts beyond the stated policy scope.
  Gap evidence files: `results/ideal_frontier_readiness.json`, `reports/ideal_frontier_readiness_report.md`, `results/external_benchmark_runtime_probe.json`, `reports/external_benchmark_runtime_probe_report.md`, `results/benchmark_libero_wam.json`, `results/benchmark_libero_visual_language_bc_policy.json`, `reports/final_decision_report.md`
- `full_robocasa_wide`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: RoboCasa has broad committed coverage, but not full RoboCasa-wide validation.
  Promotion requirements:
  - Future-only promotion requirement, not current evidence: Rollout-pool or policy artifacts covering the full declared RoboCasa task distribution, not only sampled or stratified subsets.
  - Future-only promotion requirement, not current evidence: Registry coverage evidence showing the promoted task set matches the full benchmark scope claimed in README and paper text.
  Missing evidence classes:
  - Missing future-only evidence class, not current evidence: Full declared RoboCasa task-distribution rollout-pool or policy artifacts.
  - Missing future-only evidence class, not current evidence: Coverage proof that promoted task IDs match the full local benchmark registry scope.
  - Missing future-only evidence class, not current evidence: Claim evidence that distinguishes full-suite validation from sampled or stratified-family validation.
  Gap evidence files: `results/ideal_frontier_readiness.json`, `reports/ideal_frontier_readiness_report.md`, `results/benchmark_robocasa_catalog_probe.json`, `reports/benchmark_blocker_report.md`, `reports/final_decision_report.md`
- `maniskill_visual_ee`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: ManiSkill evidence is state-mode; visual and EE-control blockers are artifact-documented.
  Promotion requirements:
  - Future-only promotion requirement, not current evidence: Successful ManiSkill RGB/RGB-D rollout or WAM artifacts generated from rendered observations without the current renderer blocker.
  - Future-only promotion requirement, not current evidence: Successful ManiSkill end-effector-control artifacts or a scoped statement that no EE-control claim is being made.
  Missing evidence classes:
  - Missing future-only evidence class, not current evidence: Successful ManiSkill RGB/RGB-D rendered-observation rollout or WAM artifact.
  - Missing future-only evidence class, not current evidence: Successful ManiSkill end-effector-control artifact or an explicitly narrower promoted control scope.
  - Missing future-only evidence class, not current evidence: Closed-loop or rollout-pool metrics generated after the renderer/control blockers are cleared.
  Gap evidence files: `results/ideal_frontier_readiness.json`, `reports/ideal_frontier_readiness_report.md`, `results/benchmark_maniskill_visual_probe.json`, `results/benchmark_maniskill_dependency_probe.json`, `reports/maniskill_visual_blocker_report.md`, `reports/maniskill_dependency_blocker_report.md`
- `universal_wam_training_recipe`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: Universal WAM training optimization is framed as future work, not a current result.
  Promotion requirements:
  - Future-only promotion requirement, not current evidence: A tested train/inference optimizer that chooses data scale, model capacity, rollout horizon, scorer quality, safety constraints, and sampling budget.
  - Future-only promotion requirement, not current evidence: Evidence that the optimizer generalizes beyond the current artifact-specific WAM-lite and benchmark recipes.
  Missing evidence classes:
  - Missing future-only evidence class, not current evidence: Executable universal train/inference optimizer artifact.
  - Missing future-only evidence class, not current evidence: Cross-environment evidence that the optimizer chooses data, model, scorer, horizon, safety, and sampling budgets.
  - Missing future-only evidence class, not current evidence: Claim evidence separating this future recipe from the current exact test-time inference law.
  Gap evidence files: `results/ideal_frontier_readiness.json`, `reports/ideal_frontier_readiness_report.md`, `results/universal_wam_train_inference_optimizer.json`, `reports/universal_wam_train_inference_optimizer_report.md`, `results/publication_scope.json`, `reports/final_decision_report.md`, `reports/reviewer_risk_assessment.md`

The boundary is clean: artifact-backed rows may be promoted with their stated scope, while future-only ideal endpoints remain non-promotable.
