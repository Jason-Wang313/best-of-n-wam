# Ideal Frontier Readiness

- Verified audit: `True`
- Frontiers audited: `5`
- Ready to promote: `0`
- Not ready to promote: `5`

This report is a gap audit, not a result promotion mechanism.

## Frontier Matrix

- `real_robot_hil`: ready_to_promote=`False`; missing=['real_robot_or_hil_artifact_present', 'real_world_success_metrics_present']
  - real_robot_or_hil_artifact_present: `False` (physical_trial_metric_files=0, availability_probe_verified=True, possible_hardware=0, probe_trial_metrics=0)
  - real_world_success_metrics_present: `False` (physical trial success/utility metric artifacts=0)
  Next action: Collect real robot or hardware-in-the-loop rollout/control logs with trial IDs and success or utility metrics.
- `modern_vla_libero`: ready_to_promote=`False`; missing=['heldout_sparse_success_modern_vla_eval']
  - libero_rgb_language_policy_evaluated: `True` (verified=True, eval_episodes=30)
  - no_shortcut_eval_interface: `True` (policy={'type': 'rgb_proprio_language_knn_behavior_cloning', 'uses_rgb': True, 'uses_language': True, 'uses_robot_proprio': True, 'uses_simulator_object_state': False, 'uses_task_id': False, 'uses_phase_index': False, 'uses_target_point_command': False, 'uses_previous_action': True, 'uses_step_clock': True, 'uses_language_candidate_filter': True, 'knn_k': 3, 'knn_temperature': 0.05, 'image_grid': 8, 'language_hash_dim': 64})
  - sparse_success_ci_reported: `True` (ci={'n': 30, 'mean': 1.0, 'lo': 1.0, 'hi': 1.0, 'std': 0.0})
  - model_artifact_present: `True` (model_path=results\models\benchmark_libero_visual_language_bc_policy.npz)
  - neural_visual_language_model_class: `True` (canonical_type='rgb_proprio_language_knn_behavior_cloning', canonical_is_neural=None, canonical_eval_successes=30, canonical_heldout=True, aux_candidates=4, aux_artifact='benchmark_libero_tiny_neural_rbf_vla_policy.json', aux_type='rbf_neural_vla_behavior_cloning', aux_is_neural=True, aux_verified=True, aux_eval=1, aux_eval_successes=1, aux_heldout=True)
  - modern_vla_scale_or_pretrained_model: `True` (canonical_pretrained=None, canonical_params=None, aux_pretrained=False, aux_params=846286; availability_probe_verified=True, vla_package_importable=False, local_vla_like=0, hf_reachable=5, joint_runtime=True, pretrained_loaded=True, pretrained_params=604934220, ready_for_policy_eval=True, execution_attempted=True, execution_verified=True, execution_model=HuggingFaceVLA/smolvla_libero, execution_params=604934220, execution_action=True, execution_step=True, execution_stage=None, execution_error=None)
  - heldout_sparse_success_modern_vla_eval: `False` (eval_verified=True, heldout_eval=True, eval_episodes=1, n_eval_seeds=1, max_steps=1, eval_successes=0, success_rate=0.0, success_ci={'n': 1, 'mean': 0.0, 'lo': 0.0, 'hi': 0.7934567085261071, 'method': 'wilson'}; last_attempt_present=True, last_attempt_verified=False, last_attempt_horizon=2, last_attempt_max_steps=2, last_attempt_requested_seeds=[305], last_attempt_failure_stage=timeout, last_attempt_error_type=TimeoutExpired; attempt_history_count=2, attempt_history_error_types=['TimeoutExpired', 'WindowsAccessViolation']; one_step_verified=True, one_step_success=False)
  - current_runtime_can_rerun_libero: `True` (runtime_probe_verified=True, libero=True)
  Next action: Extend the verified one-step LIBERO-tuned SmolVLA execution into heldout sparse-success episodes with confidence intervals.
- `full_robocasa_wide`: ready_to_promote=`False`; missing=['full_registry_rollout_pool_coverage', 'full_registry_any_artifact_coverage', 'all_categories_fully_covered']
  - catalog_probe_verified: `True` (verified=True)
  - full_registry_rollout_pool_coverage: `False` (rollout_pool_covered=132/396)
  - full_registry_any_artifact_coverage: `False` (any_artifact_covered=136/396; residual_attempts=7, residual_candidates=14, residual_timeouts=13, residual_nondegenerate=0, residual_categories=['cleaning', 'close', 'long_horizon_or_compositional', 'manipulate', 'turn'], residual_artifacts=['benchmark_robocasa_residual_frontier_sweep_long_unattempted_probe.json', 'benchmark_robocasa_residual_frontier_sweep_longhorizon_gap_probe.json', 'benchmark_robocasa_residual_frontier_sweep_quick_close.json', 'benchmark_robocasa_residual_frontier_sweep_quick_gap.json', 'benchmark_robocasa_residual_frontier_sweep_short_unattempted_probe.json']; triage_attempted_not_covered=36, triage_unattempted=224, triage_status_counts={'constructor_signature_failure': 11, 'micro_nondegenerate_covered': 4, 'not_implemented': 1, 'rollout_pool_covered': 132, 'timed_out': 13, 'unattempted': 224, 'value_error': 2, 'zero_distance_no_progress': 9})
  - all_categories_fully_covered: `False` (missing=['cleaning', 'close', 'cooking', 'long_horizon_or_compositional', 'manipulate', 'open', 'pick_place', 'turn'])
  Next action: Extend RoboCasa artifacts from sampled/stratified coverage to every declared registry task or state a narrower benchmark scope.
- `maniskill_visual_ee`: ready_to_promote=`False`; missing=['rgb_or_rgbd_success', 'ee_control_success', 'pinocchio_available_for_ee']
  - visual_probe_attempted: `True` (attempted=True)
  - rgb_or_rgbd_success: `False` (visual_success=False, attempts=13)
  - ee_control_success: `False` (ee_success=False, attempts=2)
  - pinocchio_available_for_ee: `False` (pinocchio_import=False, pinocchio_api=False, pypi_pinocchio_api=False, external_env_pinocchio_api_any=False, external_env_python_count=3, pypi_missing_symbols=['Model', 'GeometryModel', 'buildModelFromUrdf'], missing_symbols=['Model', 'GeometryModel', 'buildModelFromUrdf'])
  Next action: Clear the Vulkan descriptor-pool renderer failure and install a compatible Pinocchio stack for EE controllers.
- `universal_wam_training_recipe`: ready_to_promote=`False`; missing=['universal_generalization_proof_present']
  - future_scope_guard_present: `True` (publication_scope_verified=True)
  - optimizer_artifact_present: `True` (checked=['src\\wam_inference_value\\train_inference_optimizer.py', 'experiments\\universal_wam_train_inference_optimizer.py'])
  - optimizer_result_verified: `True` (verified=True, not_a_universal_proof=True)
  - optimizer_choice_dimensions_covered: `True` (dimensions={'data_scale': True, 'model_class': True, 'model_capacity': True, 'rollout_horizon': True, 'scorer': True, 'safety_policy': True, 'rollout_budget': True})
  - cross_environment_optimizer_evidence_present: `True` (selected_envs=9, families=3)
  - universal_generalization_proof_present: `False` (optimizer is evidence-bound to committed artifacts and explicitly not a universal proof; boundary_verified=True, boundary_type=no_free_lunch_boundary, boundary_regret_lb=0.5)
  Next action: Build and evaluate a train/inference optimizer across data scale, model class, horizon, scorer, safety, and rollout budget.
