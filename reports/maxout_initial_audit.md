# Max-Out Initial Audit

Audit date: 2026-05-30.

## 1. Currently Verified

- Exact finite best-of-N theorem code and tie-aware implementation exist.
- Unit tests cover binary, utility, AUC, ties, adaptive allocation math, and toy environments.
- Analytic BlockPush2D artifacts exist for EXP1-EXP8.
- Learned BlockPush2D WAM-lite artifacts exist for EXP1, EXP4, EXP5, EXP6, EXP7, and learned-vs-analytic-vs-oracle.
- `claims_status.py` gates README and paper-outline overclaims.

## 2. Toy-Only

- The main controlled environments are CPU toy environments.
- Gymnasium/MuJoCo Reacher-v5, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, ManiSkill3 state-mode tasks, RoboCasa kitchen smoke plus single-task and three-task learned-WAM artifacts, and LIBERO Spatial rollout-pool WAM artifacts now have external benchmark artifacts.
- No real robot, DreamZero, UWM, or solved-task LIBERO policy-performance result is claimed; RoboCasa is verified only for a three-task pick-place family, and LIBERO is verified only as a three-task rollout-pool dense-utility artifact.

## 3. Learned-Model Evidence

- Learned ridge WAM-lite validation final-position L2 MAE: `0.1117`.
- Learned ridge WAM-lite validation utility MAE: `0.8624`.
- OOD splits reported: `3`.
- Multi-env learned backbones trained: `horizon_wam, mlp_dynamics_wam, ensemble_wam`.

## 4. Missing For Robotics Reviewers

- LIBERO solved-task policy-performance artifacts are still missing; current LIBERO evidence is a three-task Spatial rollout-pool WAM-lite artifact with dense progress utility.
- Full RoboCasa-wide learned-WAM benchmark artifacts are still missing; current RoboCasa evidence is a single-task smoke rollout pool, a single-task learned-WAM artifact, and a three-task pick-place family learned-WAM artifact.
- ManiSkill evidence is state-mode and joint-delta controlled; end-effector delta-pose control is not claimed because Pinocchio was unavailable.
- Meta-World ML1 evidence covers `reach-v3`, `push-v3`, and `drawer-open-v3` with state/action-sequence WAM-lite artifacts.
- RoboSuite evidence covers Panda `Lift`, `Stack`, and `Door` with clone-restored MuJoCo rollout pools, state/action-sequence WAM-lite artifacts, and small closed-loop learned/reward-versus-random evaluation.
- No real robot data.
- No high-dimensional policy or vision-language WAM evidence.

## 5. Exact-Law Tautologies Versus Heldout Predictions

- Exact-law claims are conditional identities for a fixed known rollout score/utility distribution.
- Monte Carlo agreement checks implementation, not generalization.
- Pilot-to-heldout curves are statistical predictions and can fail under small pilots or shift.
- Learned WAM claims are heldout toy predictions, not theorem consequences.

## 6. README Claim Guarding

- README must state LIBERO as optional/separate-environment and limited to three Spatial rollout-pool dense-utility validation; RoboCasa remains three-task pick-place family validation unless broader task family artifacts exist.
- README must state ManiSkill as state-mode joint-delta evidence only, not EE-control or real-robot evidence.
- README must call current evidence learned-toy and multi-env toy validation.
- README must not claim real robot evidence or universal WAM training laws.

## 7. Canonical Versus Legacy Scripts

- Canonical smoke: `scripts/run_smoke.sh`.
- Canonical learned toy: `scripts/run_learned_wam_toy.sh`.
- Canonical multi-env: `scripts/run_multi_env.sh`.
- Optional benchmark: `scripts/run_benchmark_smoke.sh`, `scripts/run_benchmark_full.sh`.
- Optional visual: `scripts/run_visual_optional.sh`.
- Canonical audit layer: `scripts/run_inference_audit.sh`.
- Max-out orchestration: `scripts/run_maxout_all.sh`.

## 8. Utility Normalization

- Main experiment tables include raw utility.
- Multi-env curves include `normalized_real_utility`.
- Canonical older analytic artifacts still mix raw task utilities, so cross-env comparisons should use normalized metrics or within-env deltas.

## 9. Confidence Intervals

- Learned and multi-env main claims use five seeds with CIs.
- Some analytic smoke artifacts remain single-seed smoke checks by design.
- Claim status should downgrade any claim whose CI is absent or non-supportive.

## 10. Readiness Tier

The project has learned-toy, multi-env toy, Gymnasium/MuJoCo, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, and ManiSkill3 state-mode benchmark validation paths. It is much closer to a serious ML submission artifact, but still not real-robot validated.
