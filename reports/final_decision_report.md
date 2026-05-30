# Final Decision Report

## 1. Tier

Benchmark-full plus ManiSkill state-mode, visual-, and audit-validated: learned-toy, multi-env toy validation, Gymnasium/MuJoCo Reacher-v5 benchmark validation, ManiSkill3 state-mode manipulation validation, toy visual mode, benchmark RGB WAM-lite on Reacher-v5 frames, and inference-value audit framework artifacts.

## 2. Strongest Verified Claims

- 1. Exact finite binary law verified. Evidence: success MAE=0.002104612037108505
- 2. Utility-valued finite law verified. Evidence: utility MAE=0.013762680427723635
- 3. N=2 AUC identity verified. Evidence: max identity error=0.0
- 4. High-N moment hierarchy verified. Evidence: same-p/kappa gap=0.9988209815422198
- 5. Pilot-to-heldout improves with K. Evidence: relative MAE reduction=0.43698115249631736
- 6. Pilot uncertainty is reported. Evidence: pilot improvement CI={'n': 16, 'mean': 0.44172358371190434, 'std': 0.6445133007518349, 'stderr': 0.16112832518795872, 'ci95': 0.3158115173683991, 'lo': 0.12591206634350527, 'hi': 0.7575351010803034}
- 7. Score function controls inference value. Evidence: oracle-random N64=6.976857029097962
- 8. Best non-oracle beats random with CI. Evidence: learned CI={'n': 5, 'mean': 5.97838149808149, 'std': 1.2959235541324863, 'stderr': 0.5795546321366736, 'ci95': 1.1359270789878801, 'lo': 4.8424544190936105, 'hi': 7.11430857706937}
- 9. Oracle remains above learned/non-oracle. Evidence: oracle-learned CI={'n': 5, 'mean': 1.9951266518211337, 'std': 0.18826614525085428, 'stderr': 0.08419517972855187, 'ci95': 0.16502255226796164, 'lo': 1.830104099553172, 'hi': 2.1601492040890955}
- 10. Real-vs-imagined utility gap verified. Evidence: severe-none=16.435472824968038
- 11. Mismatch gap grows with N. Evidence: learned severe gap CI={'n': 5, 'mean': 13.8807751738171, 'std': 0.7005630946072382, 'stderr': 0.3133013404138802, 'ci95': 0.6140706272112052, 'lo': 13.266704546605894, 'hi': 14.494845801028305}
- 12. Bad scorer falsification verified. Evidence: anti N64=-26.565706083129044, N1=-14.101561337022474

## 3. Weakest Claims

- none

## 4. Abstract Claims

- Exact best-of-N inference laws for rollout selection.
- The score/utility distribution determines the value of additional rollouts.
- Model/scorer mismatch can make best-of-N amplify imagined futures rather than real utility.
- Learned and multi-env toy artifacts validate the theory and failure modes.

## 5. Discussion-Only Claims

- LIBERO/RoboCasa integration.
- ManiSkill RGB/RGB-D or EE-control validation.
- ManiSkill RGB/RGB-D benchmark WAM validation.
- Universal WAM training and train-inference scaling.

## 6. Skeptical Reviewer Attack

The project still lacks real robot artifacts, LIBERO/RoboCasa, and ManiSkill visual or EE-control validation.

## 7. Current Answer

The repo answers the mathematical and controlled toy-science objections with tests, multi-env artifacts, learned WAM-lite backbones, falsification, an anti-overclaim system, a state-based Gymnasium/MuJoCo benchmark, and a three-task ManiSkill3 state-mode benchmark. It does not yet answer real-robot realism.

## 8. Unresolved

- LIBERO/RoboCasa rollout collection.
- ManiSkill RGB/RGB-D or end-effector-control validation.
- Real robot validation.
- Strong ManiSkill RGB/RGB-D WAM evidence.

## 9. Workshop Readiness

Yes, as a theory-plus-controlled-learned-toy paper artifact.

## 10. Main-Conference Readiness

Substantially stronger after ManiSkill state-mode validation. Still not a real-robot paper and still weaker than a benchmark-heavy robotics paper with LIBERO/RoboCasa or RGB-D manipulation.

## 11. Single Highest-Value Next Step

Add LIBERO or ManiSkill RGB/RGB-D WAM validation next; that is now higher value than another state-only toy run.

## Command Results

- `python -m pytest -q`: passed with `32 passed`.
- Large analytic `scripts/run_all.sh`: attempted; the tool timeout was reached during the heavy EXP6 allocation sweep after EXP1-EXP5 refreshed. The spawned allocation process was stopped, robust EXP8 was regenerated separately, and the final claim gate remained fully verified.
- `bash scripts/run_smoke.sh`: passed; EXP1 success MAE `0.00696`, utility MAE `0.04511`; EXP8 smoke conditional-law MAE `0.0055`.
- `bash scripts/run_learned_wam_toy.sh`: passed; learned validation utility MAE `0.8624`, final-position L2 MAE `0.1117`; learned-vs-analytic N64 real-utility delta `1.170 +/- 0.219`.
- `bash scripts/run_multi_env.sh`: passed with `envs=5`, `backbones=3`, `seeds=5`.
- robust EXP8 rerun: passed; stale post-pre CI lower bound `0.0255`, stale-adaptive post CI lower bound `0.0613`.
- `bash scripts/run_benchmark_full.sh`: passed with Gymnasium/MuJoCo `Reacher-v5` and ManiSkill3 state-mode tasks; Reacher exact-law utility MAE `0.01875`; Reacher closed-loop learned-random CI lower bound `0.4102`; ManiSkill exact-law utility MAE `0.0034`; ManiSkill closed-loop learned-random CI lower bound `0.0102`.
- `bash scripts/run_visual_optional.sh`: passed; toy visual MAE `0.0185`; benchmark RGB WAM utility corr `0.2199`, utility MAE `0.5208`, visual-random N32 CI lower `0.1998`.
- `bash scripts/run_inference_audit.sh`: passed; audit tail-gain correlation `0.9864`, repair-predicted N64 CI mean `0.3489`, predicted N128-N1 scaling gain `0.0255`.
- `python scripts/claims_status.py`: passed with `60` verified, `0` partial, `0` unsupported, `0` failed, and `0` README/paper overclaims.
