# Final Decision Report

## 1. Tier

Benchmark-full plus Fetch, Meta-World, RoboSuite, ManiSkill state-mode, RoboCasa single-task, three-task pick-place-family, broad four-task, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned WAM-lite, LIBERO Spatial three-task rollout-pool WAM-lite, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned autonomous low-dimensional BC smoke, LIBERO RGB/proprio/language BC smoke, visual-, blocker-probe-, and audit-validated: learned-toy, multi-env toy validation, Gymnasium/MuJoCo Reacher-v5 benchmark validation, Gymnasium Robotics Fetch validation, Meta-World ML1 manipulation validation, RoboSuite Panda manipulation validation, ManiSkill3 state-mode manipulation validation, RoboCasa kitchen smoke plus single-task, three-task, broad task family, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual clean/cook learned-WAM validation, LIBERO rollout-pool learned-WAM validation, LIBERO scripted sparse-success smoke, LIBERO learned action-head sparse-success smoke, LIBERO time-conditioned low-dimensional BC sparse-success smoke, LIBERO RGB/proprio/language sparse-success smoke, toy visual mode, Reacher RGB WAM-lite, Fetch RGB WAM-lite, ManiSkill visual/EE-control blocker probing, and inference-value audit framework artifacts.

## 2. Strongest Verified Claims

- 1. Exact finite binary law verified. Evidence: success MAE=0.002104612037108505
- 2. Utility-valued finite law verified. Evidence: utility MAE=0.013762680427723635
- 3. N=2 AUC identity verified. Evidence: max identity error=0.0
- 4. High-N moment hierarchy verified. Evidence: same-p/kappa gap=0.9988209815422198
- 5. Pilot-to-heldout improves with K. Evidence: relative MAE reduction=0.7280063038060411; reduction CI={'n': 1200, 'mean': 0.8233589071075791, 'std': 0.7110080730202641, 'stderr': 0.020525035117712326, 'ci95': 0.04022906883071616, 'lo': 0.7831298382768629, 'hi': 0.8635879759382953}
- 6. Pilot uncertainty is reported. Evidence: pilot improvement CI={'n': 1200, 'mean': 0.8233589071075791, 'std': 0.7110080730202641, 'stderr': 0.020525035117712326, 'ci95': 0.04022906883071616, 'lo': 0.7831298382768629, 'hi': 0.8635879759382953}
- 7. Score function controls inference value. Evidence: oracle-random N64=7.084333950224235
- 8. Best non-oracle beats random with CI. Evidence: learned CI={'n': 5, 'mean': 5.97838149808149, 'std': 1.2959235541324863, 'stderr': 0.5795546321366736, 'ci95': 1.1359270789878801, 'lo': 4.8424544190936105, 'hi': 7.11430857706937}
- 9. Oracle remains above learned/non-oracle. Evidence: oracle-learned CI={'n': 5, 'mean': 1.9951266518211337, 'std': 0.18826614525085428, 'stderr': 0.08419517972855187, 'ci95': 0.16502255226796164, 'lo': 1.830104099553172, 'hi': 2.1601492040890955}
- 10. Real-vs-imagined utility gap verified. Evidence: severe-none=16.725668702334993
- 11. Mismatch gap grows with N. Evidence: learned severe gap CI={'n': 5, 'mean': 13.880775173817053, 'std': 0.7005630946072381, 'stderr': 0.31330134041388014, 'ci95': 0.6140706272112051, 'lo': 13.266704546605848, 'hi': 14.494845801028259}
- 12. Bad scorer falsification verified. Evidence: anti_scorer utility N64=-26.565706083129044, utility N1=-14.101561337022474

## 3. Weakest Claims

- No real-robot or hardware-in-the-loop evidence; every promoted robotics result is simulator or benchmark evidence.
- LIBERO evidence is limited to three Spatial dense rollout-pool WAM-lite tasks plus Object sparse-success scripted/action-head/time-conditioned/RGB-proprio-language feature-kNN smokes, not modern VLA policy performance or full LIBERO validation.
- RoboCasa evidence is broad but not full RoboCasa-wide validation: committed rollout-pool artifacts cover 132 of 396 local registry task IDs, with micro-rollout probes covering 106 task IDs.
- ManiSkill evidence is CPU state-mode joint-delta control; RGB/RGB-D and end-effector-control validation are blocker-documented, not verified.
- Learned WAMs are intentionally lightweight ridge/kNN/CPU models; the repo does not prove a universal WAM training recipe.

## 4. Abstract Claims

- Exact best-of-N inference laws for rollout selection.
- The score/utility distribution determines the value of additional rollouts.
- Model/scorer mismatch can make best-of-N amplify imagined futures rather than real utility.
- Learned and multi-env toy artifacts validate the theory and failure modes.

## 5. Discussion-Only Claims

- Modern VLA-style sparse-success LIBERO policy performance and full RoboCasa-wide learned-WAM validation.
- ManiSkill RGB/RGB-D or EE-control validation.
- ManiSkill RGB/RGB-D benchmark WAM validation.
- Universal WAM training and train-inference scaling.

## 6. Skeptical Reviewer Attack

The project still lacks real robot artifacts, modern VLA-style sparse-success LIBERO policy validation, full RoboCasa-wide learned-WAM validation, and ManiSkill visual or EE-control validation.

## 7. Current Answer

The repo answers the mathematical and controlled toy-science objections with tests, multi-env artifacts, learned WAM-lite backbones, falsification, an anti-overclaim system, a state-based Gymnasium/MuJoCo benchmark, a three-task Gymnasium Robotics Fetch benchmark, a three-task Meta-World ML1 benchmark, a three-task RoboSuite Panda benchmark, a three-task ManiSkill3 state-mode benchmark, RoboCasa kitchen pick-place, broad atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned-WAM artifacts, a three-task LIBERO Spatial rollout-pool learned-WAM artifact, a LIBERO Object sparse-success scripted smoke, a LIBERO learned action-head smoke, a LIBERO time-conditioned low-dimensional BC sparse-success smoke, and a LIBERO RGB/proprio/language sparse-success smoke. It does not yet answer real-robot realism.

## 8. Unresolved

- Modern VLA-style sparse-success LIBERO policy evaluation beyond the current hand scripted/action-head/time-conditioned low-dimensional/RGB-proprio-language feature-kNN smokes and dense rollout-pool utility.
- Full RoboCasa-wide learned-WAM rollout collection beyond the current pick-place-family, broad atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook artifacts.
- ManiSkill RGB/RGB-D or end-effector-control validation.
- Real robot validation.
- Strong ManiSkill RGB/RGB-D WAM evidence; current repo has only a local failure probe with exact renderer/control blockers.

## 9. Workshop Readiness

Yes, as a theory-plus-controlled-learned-toy paper artifact.

## 10. Main-Conference Readiness

Substantially stronger after Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, ManiSkill state-mode validation, pick-place, broad, 12-task, 24-task, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook RoboCasa learned-WAM validation, three-task LIBERO rollout-pool validation, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned low-dimensional BC smoke, and LIBERO RGB/proprio/language BC smoke. Still not a real-robot paper and still weaker than a benchmark-heavy robotics paper with modern VLA-style sparse-success LIBERO policy performance, full RoboCasa-wide validation, or RGB-D manipulation.

## 11. Single Highest-Value Next Step

Add modern VLA-style sparse-success LIBERO policy evaluation or full RoboCasa-wide task coverage next; ManiSkill RGB/RGB-D WAM validation remains the other high-value path if the local SAPIEN/Vulkan blocker is cleared.

## Command Results

- `python -m pytest -q`: passed with `183 passed`.
- `python scripts/test_inventory.py --fail-on-error`: passed with `183` collected tests, `6` inventory checks, and `0` issues.
- `bash scripts/run_all.sh`: passed; full analytic EXP1-EXP8 sweep completed with EXP1 success MAE `0.00210`, EXP3 relative MAE reduction `0.728`, EXP6 moment-uniform delta `0.0767`, EXP7 useful N64-N1 success delta `0.217`, and EXP8 conditional-law MAE `0.0027`.
- `bash scripts/run_smoke.sh`: passed; EXP1 success MAE `0.00696`, utility MAE `0.04511`; EXP8 smoke conditional-law MAE `0.0055`.
- `bash scripts/run_learned_wam_toy.sh`: passed; learned validation utility MAE `0.8624`, final-position L2 MAE `0.1117`; learned-vs-analytic N64 real-utility delta `1.170 +/- 0.219`.
- `bash scripts/run_multi_env.sh`: passed with `envs=5`, `backbones=3`, `seeds=5`.
- robust EXP8 rerun: passed; stale post-pre CI lower bound `0.0255`, stale-adaptive post CI lower bound `0.0613`.
- `bash scripts/run_benchmark_full.sh`: passed with Gymnasium/MuJoCo `Reacher-v5`, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, and ManiSkill3 state-mode tasks; optional RoboCasa smoke, single-task learned-WAM, three-task learned-WAM, broad four-task learned-WAM, 12-task family learned-WAM, 24-task family learned-WAM, extra four-task learned-WAM, combined 28-task learned-WAM, combined 32-task learned-WAM, stratified micro probe, frontier micro probe, stratified 55-task learned-WAM, and stratified 97-task learned-WAM runs were generated separately with `ROBOCASA_PYTHON`; optional LIBERO three-task Spatial rollout-pool WAM, Object sparse-success scripted smoke, learned action-head smoke, time-conditioned autonomous low-dimensional BC smoke, and RGB/proprio/language BC smoke were generated separately with `LIBERO_PYTHON`; Reacher exact-law utility MAE `0.01875`; Reacher closed-loop learned-random CI lower bound `0.4102`; Fetch exact-law utility MAE `0.0126`; Meta-World exact-law utility MAE `0.0298`; Meta-World learned-random N32 CI lower `0.0975`; RoboSuite exact-law utility MAE `0.0024`; RoboSuite learned-random N32 CI lower `0.2447`; RoboSuite closed-loop learned-random N8 CI lower `0.0798`; ManiSkill exact-law utility MAE `0.0034`; ManiSkill closed-loop learned-random CI lower bound `0.0102`; RoboCasa smoke exact-law utility MAE `0.0003`; RoboCasa learned utility corr `0.7640`; RoboCasa learned-random N8 CI lower `0.0503`; RoboCasa three-task utility corr `0.6752`; RoboCasa three-task learned-random N8 CI lower `0.1691`; RoboCasa broad utility corr `0.8599`; RoboCasa broad learned-random N8 CI lower `0.2112`; RoboCasa 12-task family utility corr `0.8330`; RoboCasa 12-task family learned-random N8 CI lower `0.1830`; RoboCasa 24-task family utility corr `0.8523`; RoboCasa 24-task family learned-random N8 CI lower `0.2393`; RoboCasa extra four-task utility corr `0.7993`; RoboCasa extra four-task learned-random N8 CI lower `0.2469`; RoboCasa combined 28-task utility corr `0.8335`; RoboCasa combined 28-task learned-random N8 CI lower `0.2353`; RoboCasa combined 32-task utility corr `0.8384`; RoboCasa combined 32-task learned-random N8 CI lower `0.2276`; RoboCasa stratified 55-task utility corr `0.8326`; RoboCasa stratified 55-task learned-random N8 CI lower `0.2743`; RoboCasa stratified 55-task exact-law utility MAE `0.0003`; RoboCasa stratified 97-task utility corr `0.8380`; RoboCasa stratified 97-task learned-random N8 CI lower `0.3149`; RoboCasa stratified 97-task oracle-learned N8 CI lower `0.0369`; RoboCasa stratified 97-task exact-law utility MAE `0.0003`; RoboCasa frontier micro nondegenerate tasks `42`; LIBERO utility corr `0.3526`; LIBERO learned-random N8 CI lower `0.2653`; LIBERO scripted success CI lower `1.0000`; LIBERO action-head success CI lower `1.0000`; LIBERO autonomous BC success CI lower `1.0000`; LIBERO RGB/proprio/language BC success CI lower `1.0000`.
- `bash scripts/run_robocasa_residual_probes.sh`: generated separately with `ROBOCASA_PYTHON`; residual clean/cook sweep verified `True` with `35` / `43` nondegenerate task IDs and `2` timeout chunks; residual 35-task learned-WAM verified `True` with train/validation/eval `140`/`140`/`280`, exact-law utility MAE `0.0002`, utility corr `0.8345`, learned-random N4 CI lower `0.1974`, and oracle-learned N4 CI lower `0.0137`.
- `ROBOCASA_PYTHON experiments/benchmark_robocasa_residual_frontier_sweep.py --summary-tag atomic_gap_probe ...`: attempted `4` additional uncovered atomic/control task IDs; completed chunks `1`; timed-out chunks `3`; nondegenerate task IDs `0`; not promoted when verified is `False`.
- `python experiments/benchmark_gym_robotics_suite.py`: passed with `['FetchReach-v4', 'FetchPush-v4', 'FetchPickAndPlace-v4']`; exact-law utility MAE `0.0126`; learned-random N32 CI lower `0.4409`; closed-loop learned-random N32 CI lower `0.2572`.
- `bash scripts/run_visual_optional.sh`: passed; toy visual MAE `0.0185`; Reacher RGB WAM utility corr `0.2199`, utility MAE `0.5208`, visual-random N32 CI lower `0.1998`; Fetch RGB WAM mean corr `0.7325`, visual-random N32 CI lower `0.3475`; ManiSkill visual probe any visual success `False` with blocker `vk::Device::allocateDescriptorSetsUnique: ErrorOutOfPoolMemory`.
- `python experiments/benchmark_maniskill_dependency_probe.py --quick-timeout-s 60`: passed as a blocker probe; Pinocchio import `False`, robotics API `False`, binary PyPI `pinocchio` wheel `True`, binary `pin` wheel `False`, source install attempted `False`.
- `bash scripts/run_inference_audit.sh`: passed; audit tail-gain correlation `0.9864`, repair-predicted N64 CI mean `0.3489`, predicted N128-N1 scaling gain `0.0255`.
- `python scripts/artifact_integrity.py --fail-on-error`: passed with `726` artifact references checked and `0` issues.
- `python scripts/artifact_manifest.py --fail-on-error`: passed with `443` scientific artifacts, `77209558` bytes, `15` manifest checks, and `0` issues.
- `python scripts/figure_quality.py --fail-on-error`: passed with `36` figures, `8` image-quality checks, and `0` issues.
- `python scripts/result_consistency.py --fail-on-error`: passed with `157` consistency checks and `0` issues.
- `python scripts/raw_result_recompute.py --fail-on-error`: passed with `10710` aggregate metrics, `20` exact-law files, `125` seed CI columns, and `0` issues.
- `python scripts/table_schema.py --fail-on-error`: passed with `227` tables, `213298` rows, `17` schema checks, and `0` issues.
- `python scripts/source_manifest.py --fail-on-error`: passed with `243` source files, `1757562` bytes, `14` source-manifest checks, and `0` issues.
- `python scripts/runtime_environment.py --fail-on-error`: passed with Python `3.10.11`, `4` core requirements, `5` / `5` optional requirements available, `15` runtime checks, and `0` issues.
- `python scripts/experiment_registry.py --fail-on-error`: passed with `57` experiment-family entries, `69` wrapper links, `309` table artifacts, `40` figures, `10` registry checks, and `0` issues.
- `python scripts/model_artifact_integrity.py --fail-on-error`: passed with `53` model artifacts, `398` NPZ arrays, `13` joblib predictors, `10` model-artifact checks, and `0` issues.
- `python scripts/narrative_consistency.py --fail-on-error`: passed with `52` narrative checks and `0` issues.
- `python scripts/script_contracts.py --fail-on-error`: passed with `7` scripts, `133` contract checks, and `0` issues.
- `python scripts/abstract_claim_support.py --fail-on-error`: passed with `4` abstract claims, `23` backing claim links, `0` forbidden headline hits, and `0` issues.
- `python scripts/publication_scope.py --fail-on-error`: passed with `5` publication surfaces, `80` risky mentions, `0` unguarded mentions, and `0` issues.
- `python scripts/frontier_integrity.py --fail-on-error`: passed with `4` frontier items, `85` guarded mentions, `0` promoted frontier claims, and `0` issues.
- `python scripts/ideal_claim_boundary.py --fail-on-error`: passed with `9` ideal claims, `4` promotable claims, `5` future-only claims, `False` all-promotable flag, and `0` issues.
- `python scripts/claim_semantics.py --fail-on-error`: passed with `127` claims, `177` semantic checks, `53` CI-backed claims, and `0` issues.
- `python scripts/claim_scope_audit.py --fail-on-error`: passed with `127` claims, `155` scoped broad-claim mentions, `160` checks, and `0` issues.
- `python scripts/claim_reference_integrity.py --fail-on-error`: passed with `24` explicit verified-claim references, `12` unique referenced claims, and `0` issues.
- `python scripts/claim_evidence_quality.py --fail-on-error`: passed with `127` claims, `175` source links, `7` evidence checks, and `0` issues.
- `python scripts/tracked_artifact_provenance.py --fail-on-error`: passed with `119` claim sources, `470` artifact references, and `0` issues.
- `python scripts/repo_bound_artifact_audit.py --fail-on-error`: passed with `901` records, `175` claim sources, `726` artifact references, `0` outside-repo records, `0` missing records, and `0` issues.
- `python scripts/claim_ledger_integrity.py --fail-on-error`: passed with `127` claims, `31` ledger checks, and `0` issues.
- `python scripts/claim_generation_consistency.py --fail-on-error`: passed with `127` claims, `10` generation checks, and `0` issues.
- `python scripts/report_generation_consistency.py --fail-on-error`: passed with `8` reports, `9` generation checks, and `0` issues.
- `python scripts/evidence_hash_coverage.py --fail-on-error`: passed with `117` claim sources, `468` artifact references, `585` hashed records, and `0` issues.
- `python scripts/claims_status.py`: passed with `127` verified, `0` partial, `0` unsupported, `0` failed, and `0` README/paper overclaims.
