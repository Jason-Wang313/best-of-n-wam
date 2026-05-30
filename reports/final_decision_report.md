# Final Decision Report

## 1. Tier

Benchmark-full plus visual-validated: learned-toy, multi-env toy validation, Gymnasium/MuJoCo Reacher-v5 benchmark validation, toy visual mode, and benchmark RGB render sanity check.

## 2. Strongest Verified Claims

- 1. Exact finite binary law verified. Evidence: success MAE=0.006959852139023545
- 2. Utility-valued finite law verified. Evidence: utility MAE=0.0451116620102325
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

- ManiSkill/LIBERO/RoboCasa integration.
- RGB benchmark WAM validation beyond render sanity.
- Universal WAM training and train-inference scaling.

## 6. Skeptical Reviewer Attack

The project still lacks real robot artifacts and harder manipulation benchmarks such as ManiSkill/LIBERO/RoboCasa.

## 7. Current Answer

The repo answers the mathematical and controlled toy-science objections with tests, multi-env artifacts, learned WAM-lite backbones, falsification, an anti-overclaim system, and a state-based Gymnasium/MuJoCo benchmark. It does not yet answer real-robot realism.

## 8. Unresolved

- ManiSkill/LIBERO/RoboCasa rollout collection.
- Real robot validation.
- Strong RGB WAM evidence beyond render sanity checks.

## 9. Workshop Readiness

Yes, as a theory-plus-controlled-learned-toy paper artifact.

## 10. Main-Conference Readiness

Plausible as a pre-paper artifact with one lightweight benchmark; still not ideal for a robotics main conference without harder benchmark or real-system validation.

## 11. Single Highest-Value Next Step

Add one harder contact-rich benchmark task, preferably ManiSkill or LIBERO, end to end.

## Command Results

- `python -m pytest -q`: passed with `27 passed, 1 skipped`.
- Canonical artifact refresh sequence: passed.
- `bash scripts/run_smoke.sh`: passed; EXP1 success MAE `0.00696`, utility MAE `0.04511`; EXP8 smoke conditional-law MAE `0.0055`.
- `bash scripts/run_learned_wam_toy.sh`: passed; learned validation utility MAE `0.8624`, final-position L2 MAE `0.1117`; learned-vs-analytic N64 real-utility delta `1.170 +/- 0.219`.
- `bash scripts/run_multi_env.sh`: passed with `envs=5`, `backbones=3`, `seeds=5`.
- robust EXP8 rerun: passed; stale post-pre CI lower bound `0.0255`, stale-adaptive post CI lower bound `0.0613`.
- `bash scripts/run_benchmark_full.sh`: passed with Gymnasium/MuJoCo `Reacher-v5`; benchmark exact-law utility MAE `0.01875`; benchmark closed-loop learned-random CI lower bound `0.4102`.
- `bash scripts/run_visual_optional.sh`: passed; toy visual MAE `0.0185`.
- `python scripts/claims_status.py`: passed with `42` verified, `0` partial, `0` unsupported, `0` failed, and `0` README/paper overclaims.
