# Ideal Frontier Blocker Audit

- verified: `True`
- scope: `documents why ideal-frontier claims remain unpromoted; not evidence for those claims`
- ready to promote: `0`
- issues: `0`

This report is blocker evidence, not validation evidence for the ideal endpoints.

## real_robot_hil

- blocker class: `hardware_or_hil_trial_absent`
- resolution class: `external_physical_evidence_required`
- local progress status: No local code or simulator artifact can promote this endpoint; it needs physical or HIL trial metrics.
- ready to promote: `False`
- missing signals: `['real_robot_or_hil_artifact_present', 'real_world_success_metrics_present']`
- next action: Collect real robot or hardware-in-the-loop rollout/control logs with trial IDs and success or utility metrics.

### Evidence

- `probe_verified`: `True`
- `possible_hardware_device_count`: `0`
- `trial_metric_artifact_count`: `0`
- `claim_ready`: `False`

## modern_vla_libero

- blocker class: `modern_vla_execution_or_eval_absent`
- resolution class: `runtime_policy_eval_required`
- local progress status: Runtime access exists, but the promoted endpoint needs heldout sparse-success modern-VLA episodes with nonzero successes and CIs.
- ready to promote: `False`
- missing signals: `['heldout_sparse_success_modern_vla_eval']`
- next action: Extend the verified one-step LIBERO-tuned SmolVLA execution into heldout sparse-success episodes with confidence intervals.

### Evidence

- `probe_verified`: `True`
- `vla_package_importable`: `False`
- `local_vla_like_count`: `0`
- `hf_reachable_count`: `5`
- `vla_libero_joint_runtime_available`: `True`
- `pretrained_vla_loaded`: `True`
- `pretrained_vla_parameter_count`: `604934220`
- `execution_attempted`: `True`
- `execution_verified`: `True`
- `execution_failure_stage`: `None`
- `execution_error_type`: `None`
- `ready_for_policy_eval`: `True`
- `policy_eval_verified`: `True`
- `policy_eval_episodes`: `1`
- `policy_eval_successes`: `0`
- `policy_eval_success_ci`: `{'n': 1, 'mean': 0.0, 'lo': 0.0, 'hi': 0.7934567085261071, 'method': 'wilson'}`
- `last_attempt_recorded`: `True`
- `last_attempt_verified`: `False`
- `last_attempt_horizon`: `2`
- `last_attempt_max_steps`: `2`
- `last_attempt_requested_eval_seeds`: `[305]`
- `last_attempt_failure_stage`: `timeout`
- `last_attempt_error_type`: `TimeoutExpired`
- `last_attempt_child_returncode`: `None`
- `attempt_history_count`: `2`
- `attempt_history_error_types`: `['TimeoutExpired', 'WindowsAccessViolation']`

## full_robocasa_wide

- blocker class: `registry_coverage_incomplete`
- resolution class: `benchmark_coverage_required`
- local progress status: Further local RoboCasa sweeps can improve coverage, but full-suite promotion requires coverage of every declared registry task or a narrower claim.
- ready to promote: `False`
- missing signals: `['full_registry_rollout_pool_coverage', 'full_registry_any_artifact_coverage', 'all_categories_fully_covered']`
- next action: Extend RoboCasa artifacts from sampled/stratified coverage to every declared registry task or state a narrower benchmark scope.

### Evidence

- `catalog_verified`: `True`
- `registry_count`: `396`
- `any_artifact_task_count`: `136`
- `any_artifact_coverage_fraction`: `0.3434343434343434`
- `triage_verified`: `True`
- `unattempted`: `222`
- `attempted_not_covered`: `38`

## maniskill_visual_ee

- blocker class: `renderer_or_ee_dependency_blocked`
- resolution class: `local_runtime_dependency_required`
- local progress status: State-mode evidence is complete; RGB/RGB-D and EE-control promotion needs a working Vulkan/SAPIEN renderer and robotics Pinocchio API.
- ready to promote: `False`
- missing signals: `['rgb_or_rgbd_success', 'ee_control_success', 'pinocchio_available_for_ee']`
- next action: Clear the Vulkan descriptor-pool renderer failure and install a compatible Pinocchio stack for EE controllers.

### Evidence

- `visual_attempted`: `True`
- `any_visual_success`: `False`
- `visual_attempt_count`: `13`
- `any_ee_control_success`: `False`
- `pinocchio_api_available`: `False`
- `pinocchio_missing_symbols`: `['Model', 'GeometryModel', 'buildModelFromUrdf']`
- `external_env_python_count`: `3`
- `external_env_pinocchio_api_any_available`: `False`
- `external_env_pinocchio_api_available_pythons`: `[]`

## universal_wam_training_recipe

- blocker class: `unrestricted_universal_recipe_no_free_lunch`
- resolution class: `mathematical_scope_boundary`
- local progress status: This unrestricted universal-recipe endpoint is intentionally future-only; current artifacts support scoped optimization, not a universal proof.
- ready to promote: `False`
- missing signals: `['universal_generalization_proof_present']`
- next action: Build and evaluate a train/inference optimizer across data scale, model class, horizon, scorer, safety, and rollout budget.

### Evidence

- `optimizer_verified`: `True`
- `not_a_universal_proof`: `True`
- `boundary_verified`: `True`
- `boundary_type`: `no_free_lunch_boundary`
- `boundary_regret_lb`: `0.5`
