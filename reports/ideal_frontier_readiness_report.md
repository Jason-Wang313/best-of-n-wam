# Ideal Frontier Readiness

- Verified audit: `True`
- Frontiers audited: `5`
- Ready to promote: `0`
- Not ready to promote: `5`

This report is a gap audit, not a result promotion mechanism.

## Frontier Matrix

- `real_robot_hil`: ready_to_promote=`False`; missing=['real_robot_or_hil_artifact_present', 'real_world_success_metrics_present']
  - real_robot_or_hil_artifact_present: `False` (files=0)
  - real_world_success_metrics_present: `False` (no physical trial metric artifact is declared)
  Next action: Collect real robot or hardware-in-the-loop rollout/control logs with trial IDs and success or utility metrics.
- `modern_vla_libero`: ready_to_promote=`False`; missing=['neural_visual_language_model_class', 'modern_vla_scale_or_pretrained_model']
  - libero_rgb_language_policy_evaluated: `True` (verified=True, eval_episodes=30)
  - no_shortcut_eval_interface: `True` (policy={'type': 'rgb_proprio_language_knn_behavior_cloning', 'uses_rgb': True, 'uses_language': True, 'uses_robot_proprio': True, 'uses_simulator_object_state': False, 'uses_task_id': False, 'uses_phase_index': False, 'uses_target_point_command': False, 'uses_previous_action': True, 'uses_step_clock': True, 'uses_language_candidate_filter': True, 'knn_k': 3, 'knn_temperature': 0.05, 'image_grid': 8, 'language_hash_dim': 64})
  - sparse_success_ci_reported: `True` (ci={'n': 30, 'mean': 1.0, 'lo': 1.0, 'hi': 1.0, 'std': 0.0})
  - model_artifact_present: `True` (model_path=results\models\benchmark_libero_visual_language_bc_policy.npz)
  - neural_visual_language_model_class: `False` (canonical_type='rgb_proprio_language_knn_behavior_cloning', canonical_is_neural=None, canonical_eval_successes=30, canonical_heldout=True, aux_type='distilled_tiny_neural_vla_behavior_cloning', aux_is_neural=True, aux_verified=False, aux_eval=1, aux_eval_successes=0, aux_heldout=True)
  - modern_vla_scale_or_pretrained_model: `False` (canonical_pretrained=None, canonical_params=None, aux_pretrained=False, aux_params=607751)
  - current_runtime_can_rerun_libero: `True` (runtime_probe_verified=True, libero=True)
  Next action: Run a pretrained or VLA-scale neural RGB/proprio/language policy under a compatible LIBERO runtime and report heldout sparse-success CIs.
- `full_robocasa_wide`: ready_to_promote=`False`; missing=['full_registry_rollout_pool_coverage', 'full_registry_any_artifact_coverage', 'all_categories_fully_covered']
  - catalog_probe_verified: `True` (verified=True)
  - full_registry_rollout_pool_coverage: `False` (rollout_pool_covered=132/396)
  - full_registry_any_artifact_coverage: `False` (any_artifact_covered=136/396)
  - all_categories_fully_covered: `False` (missing=['cleaning', 'close', 'cooking', 'long_horizon_or_compositional', 'manipulate', 'open', 'pick_place', 'turn'])
  Next action: Extend RoboCasa artifacts from sampled/stratified coverage to every declared registry task or state a narrower benchmark scope.
- `maniskill_visual_ee`: ready_to_promote=`False`; missing=['rgb_or_rgbd_success', 'ee_control_success', 'pinocchio_available_for_ee']
  - visual_probe_attempted: `True` (attempted=True)
  - rgb_or_rgbd_success: `False` (visual_success=False, attempts=10)
  - ee_control_success: `False` (ee_success=False, attempts=2)
  - pinocchio_available_for_ee: `False` (pinocchio_import=False, pinocchio_api=False, missing_symbols=['Model', 'GeometryModel', 'buildModelFromUrdf'])
  Next action: Clear the Vulkan descriptor-pool renderer failure and install a compatible Pinocchio stack for EE controllers.
- `universal_wam_training_recipe`: ready_to_promote=`False`; missing=['universal_generalization_proof_present']
  - future_scope_guard_present: `True` (publication_scope_verified=True)
  - optimizer_artifact_present: `True` (checked=['src\\wam_inference_value\\train_inference_optimizer.py', 'experiments\\universal_wam_train_inference_optimizer.py'])
  - optimizer_result_verified: `True` (verified=True, not_a_universal_proof=True)
  - optimizer_choice_dimensions_covered: `True` (dimensions={'data_scale': True, 'model_class': True, 'model_capacity': True, 'rollout_horizon': True, 'scorer': True, 'safety_policy': True, 'rollout_budget': True})
  - cross_environment_optimizer_evidence_present: `True` (selected_envs=9, families=3)
  - universal_generalization_proof_present: `False` (optimizer is evidence-bound to committed artifacts and explicitly not a universal proof)
  Next action: Build and evaluate a train/inference optimizer across data scale, model class, horizon, scorer, safety, and rollout budget.
