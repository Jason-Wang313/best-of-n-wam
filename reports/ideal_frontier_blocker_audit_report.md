# Ideal Frontier Blocker Audit

- verified: `True`
- scope: `documents why ideal-frontier claims remain unpromoted; not evidence for those claims`
- ready to promote: `0`
- issues: `0`

This report is blocker evidence, not validation evidence for the ideal endpoints.

## real_robot_hil

- blocker class: `hardware_or_hil_trial_absent`
- ready to promote: `False`
- missing signals: `['real_robot_or_hil_artifact_present', 'real_world_success_metrics_present']`
- next action: Collect real robot or hardware-in-the-loop rollout/control logs with trial IDs and success or utility metrics.

### Evidence

- `probe_verified`: `True`
- `possible_hardware_device_count`: `0`
- `trial_metric_artifact_count`: `0`
- `claim_ready`: `False`

## modern_vla_libero

- blocker class: `modern_vla_runtime_or_checkpoint_absent`
- ready to promote: `False`
- missing signals: `['modern_vla_scale_or_pretrained_model']`
- next action: Run a pretrained or VLA-scale neural RGB/proprio/language policy under a compatible LIBERO runtime and report heldout sparse-success CIs.

### Evidence

- `probe_verified`: `True`
- `vla_package_importable`: `False`
- `local_vla_like_count`: `0`
- `hf_reachable_count`: `5`
- `vla_libero_joint_runtime_available`: `True`
- `pretrained_vla_loaded`: `True`
- `pretrained_vla_parameter_count`: `450046212`
- `ready_for_policy_eval`: `True`

## full_robocasa_wide

- blocker class: `registry_coverage_incomplete`
- ready to promote: `False`
- missing signals: `['full_registry_rollout_pool_coverage', 'full_registry_any_artifact_coverage', 'all_categories_fully_covered']`
- next action: Extend RoboCasa artifacts from sampled/stratified coverage to every declared registry task or state a narrower benchmark scope.

### Evidence

- `catalog_verified`: `True`
- `registry_count`: `396`
- `any_artifact_task_count`: `136`
- `any_artifact_coverage_fraction`: `0.3434343434343434`
- `triage_verified`: `True`
- `unattempted`: `230`
- `attempted_not_covered`: `30`

## maniskill_visual_ee

- blocker class: `renderer_or_ee_dependency_blocked`
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

## universal_wam_training_recipe

- blocker class: `unrestricted_universal_recipe_no_free_lunch`
- ready to promote: `False`
- missing signals: `['universal_generalization_proof_present']`
- next action: Build and evaluate a train/inference optimizer across data scale, model class, horizon, scorer, safety, and rollout budget.

### Evidence

- `optimizer_verified`: `True`
- `not_a_universal_proof`: `True`
- `boundary_verified`: `True`
- `boundary_type`: `no_free_lunch_boundary`
- `boundary_regret_lb`: `0.5`
