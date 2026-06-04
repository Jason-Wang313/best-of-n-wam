# Max-Out Initial Audit

Audit date: 2026-05-30.

## 1. Currently Verified

- Exact finite best-of-N theorem code and tie-aware implementation exist.
- Unit tests cover binary, utility, AUC, ties, adaptive allocation math, and toy environments.
- Analytic BlockPush2D artifacts exist for EXP1-EXP8.
- Learned BlockPush2D WAM-lite artifacts exist for EXP1, EXP4, EXP5, EXP6, EXP7, and learned-vs-analytic-vs-oracle.
- `claims_status.py` gates README and paper-outline overclaims.
- `artifact_integrity.py` verifies that referenced result artifacts exist, parse, and are nonempty.
- `artifact_manifest.py` writes deterministic SHA-256 hashes for canonical scientific result artifacts.
- `figure_quality.py` verifies that canonical PNG figures are present, readable, nonblank, nonflat, and large enough for publication-style inspection.
- `result_consistency.py` verifies that summary JSONs agree with canonical tables for row counts, coverage, CI sanity, and success counts.
- `raw_result_recompute.py` independently recomputes aggregate means, exact-law MAEs, and seed-metric CIs from raw CSV artifacts.
- `table_schema.py` verifies canonical CSV table headers, required family columns, finite numeric values, key-cell completeness, success-rate ranges, and explicit optional blanks.
- `source_manifest.py` verifies deterministic hashes for source, experiment, script, test, README/paper, requirements, and theory files.
- `runtime_environment.py` records and verifies Python/platform metadata, requirement-file hashes, core package versions, optional package availability, module probes, and command probes.
- `experiment_registry.py` verifies canonical experiment-family scripts, JSON summaries, wrapper coverage, table artifacts, and figures where expected.
- `model_artifact_integrity.py` loads committed `.npz` and `.joblib` model artifacts and verifies nonempty finite numeric arrays and predictor availability.
- `test_inventory.py` records the collected pytest node IDs and verifies the final report's pytest count is not hard-coded stale text.
- `narrative_consistency.py` verifies that high-impact README and final-report numbers match the current JSON artifacts.
- `script_contracts.py` verifies that canonical shell scripts preserve required experiment steps, optional benchmark guards, and ordered verification gates.
- `claim_semantics.py` verifies that verified-claim wording is backed by matching semantic thresholds.
- `claim_evidence_quality.py` verifies that each current claim ID is mapped to source artifacts and has structured, non-placeholder evidence.
- `tracked_artifact_provenance.py` verifies that each current claim source and published artifact reference is represented in the git index.
- `repo_bound_artifact_audit.py` verifies that every current claim source and published artifact reference resolves inside the repository and spans the expected artifact classes.
- `claim_ledger_integrity.py` verifies sorted contiguous claim IDs, JSON/Markdown count agreement, structured claim evidence, evidence-path references, empty overclaim arrays, and no non-verified final claims.
- `claim_generation_consistency.py` reruns the claim generator and verifies the JSON and Markdown claim ledgers are byte-stable and overclaim-free.
- `report_generation_consistency.py` reruns the report generator and verifies the generated narrative reports are byte-stable.
- `command_result_consistency.py` verifies that the final decision report's command-result lines match current verification artifacts.
- `evidence_hash_coverage.py` verifies that non-self current claim sources and non-self published artifact references have deterministic SHA-256 hash coverage after the other audit outputs have settled.

## 2. Toy-Only

- The main controlled environments are CPU toy environments.
- Gymnasium/MuJoCo Reacher-v5, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, ManiSkill3 state-mode tasks, RoboCasa kitchen smoke plus single-task, three-task pick-place-family, broad four-task, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned-WAM artifacts, LIBERO Spatial rollout-pool WAM artifacts, a LIBERO Object sparse-success scripted smoke, a LIBERO learned action-head smoke, a LIBERO time-conditioned autonomous low-dimensional BC sparse-success smoke, and a LIBERO RGB/proprio/language BC sparse-success smoke now have external benchmark artifacts.
- No real robot, DreamZero, UWM, modern VLA, or full LIBERO policy result is claimed; RoboCasa is verified for pick-place-family, broad atomic kitchen, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook rollout-pool artifacts, and LIBERO is verified as a three-task rollout-pool dense-utility artifact plus narrow scripted, learned action-head, time-conditioned low-dimensional BC, and RGB/proprio/language BC sparse-success smokes.

## 3. Learned-Model Evidence

- Learned ridge WAM-lite validation final-position L2 MAE: `0.1117`.
- Learned ridge WAM-lite validation utility MAE: `0.8624`.
- OOD splits reported: `3`.
- Multi-env learned backbones trained: `horizon_wam, mlp_dynamics_wam, ensemble_wam`.

## 4. Missing For Robotics Reviewers

- Modern VLA-style or full-suite LIBERO policy artifacts are still missing; current LIBERO evidence is a three-task Spatial rollout-pool WAM-lite artifact with dense progress utility, a hand scripted Object sparse-success smoke, a learned action-head smoke with scripted phases and target points, a time-conditioned low-dimensional BC smoke without phase labels or target commands at evaluation time, and an RGB/proprio/language feature-kNN BC smoke without object state or task IDs.
- Full RoboCasa-wide learned-WAM benchmark artifacts are still missing; current RoboCasa evidence is a single-task smoke rollout pool, a single-task learned-WAM artifact, a three-task pick-place family learned-WAM artifact, a broad four-task atomic-manipulation artifact, a 12-task open/close/turn family artifact, a 24-task open/close/turn/pick-place family artifact, an extra four-task pick-place-direction artifact, combined 28-task and 32-task family artifacts, stratified 55-task and 97-task kitchen artifacts, and a separate residual 35-task clean/cook horizon-1/N4 artifact.
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

- README must state LIBERO as optional/separate-environment and limited to three Spatial rollout-pool dense-utility validation plus Object sparse-success scripted, learned action-head, time-conditioned low-dimensional BC, and RGB/proprio/language BC smokes; RoboCasa is optional/separate-environment and includes pick-place-family, broad atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook rollout-pool validation, not full RoboCasa-wide validation.
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
