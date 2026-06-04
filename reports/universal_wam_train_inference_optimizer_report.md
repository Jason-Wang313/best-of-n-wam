# Train/Inference Optimizer

- Verified: `True`
- Scope: `evidence_bound_empirical_optimizer_over_existing_artifacts`
- Not a universal proof: `True`
- Candidates: `325`
- Selected environments: `9`
- Environment families: `['LIBERO', 'RoboCasa', 'ToyCPU']`

## Choice Dimensions

- `data_scale`: `True`
- `model_class`: `True`
- `model_capacity`: `True`
- `rollout_horizon`: `True`
- `scorer`: `True`
- `safety_policy`: `True`
- `rollout_budget`: `True`

## Global Conservative Recommendation

- Environment: `slippery_grasp`
- Family: `ToyCPU`
- Model class: `mlp_dynamics_wam`
- Data scale: `288`
- Horizon: `10`
- Scorer: `safety_penalized`
- Safety policy: `safety_penalized`
- Rollout budget: `8`
- CI lower delta vs random: `0.3363115905071804`
- Evidence score: `0.3013115905071804`

## Limitations

- This optimizer chooses among configurations already represented in committed artifacts; it is not a new universal training law.
- The objective is conservative CI-lower-bound improvement over random with a rollout-budget penalty, not a task-agnostic proof of optimality.
- Real robot, modern VLA LIBERO, full RoboCasa-wide, and ManiSkill RGB/EE blockers remain separate future-only frontiers.
