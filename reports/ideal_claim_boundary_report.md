# Ideal Claim Boundary Report

- Verified boundary: True
- Ideal claims audited: 9
- Promotable artifact-backed claims: 4
- Future-only non-promotable claims: 5
- All ideal claims promotable: False
- Checks: 45
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
- `modern_vla_libero`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: LIBERO artifacts are scripted/BC smokes and dense rollout-pool WAM evidence, not modern VLA performance.
- `full_robocasa_wide`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: RoboCasa has broad committed coverage, but not full RoboCasa-wide validation.
- `maniskill_visual_ee`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: ManiSkill evidence is state-mode; visual and EE-control blockers are artifact-documented.
- `universal_wam_training_recipe`: status=`future_only_not_promotable`, endpoint_supported=False, boundary_evidence_present=True, promotable=False, future_only=True
  Limitation: Universal WAM training optimization is framed as future work, not a current result.

The boundary is clean: artifact-backed rows may be promoted with their stated scope, while future-only ideal endpoints remain non-promotable.
