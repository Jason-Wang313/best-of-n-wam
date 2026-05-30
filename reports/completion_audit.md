# Completion Audit

Audit date: 2026-05-30.

## Objective

Build a standalone robotics/WAM inference-value project whose experiments support the best paper claims without relying on the original LLM pipeline.

## Evidence Summary

- Project directory: `C:\Users\wangz\best-of-n-wam`
- Fresh git repository initialized in that directory.
- Test command: `python -m pytest -q`
  - Result: `13 passed`
- Smoke command: `bash scripts/run_smoke.sh`
  - Result: completed successfully on the final current code.
- Full pipeline command: `bash scripts/run_all.sh`
  - Result: completed successfully on the final current code and wrote full-scale artifacts.
- Claim gate command: `python scripts/claims_status.py`
  - Result: 9 verified, 0 partial, 0 unsupported, 0 README overclaims.

## Requirement Checks

| Requirement | Current evidence | Status |
| --- | --- | --- |
| New clean project, no LLM pipeline pollution | Standalone directory and git repo at `C:\Users\wangz\best-of-n-wam`; original repo not edited | Complete |
| Audit old repo and document reuse/removal | `reports/repo_audit.md` | Complete |
| Exact binary finite law | `src/wam_inference_value/theorem.py`; `tests/test_theorem_binary.py` | Complete |
| Tie-aware utility-valued law | `theorem.py`; `tests/test_theorem_utility.py` | Complete |
| N=2 AUC identity | `auc_kappa`, `n2_auc_identity`; exp2 and unit tests | Complete |
| Moment hierarchy | exp2, same-p/kappa counterexample, claim 3 verified | Complete |
| CPU-only WAM-lite environment | `envs/block_push_2d.py`; no GPU/cloud dependencies | Complete |
| Multiple scorers | `scorers.py` includes predicted, random, safety/uncertainty, oracle scorers | Complete |
| Experiments 1-5 run and write artifacts | `results/exp1...json` through `results/exp5...json`; tables and figures present | Complete |
| Experiments 6-8 run and strengthen paper claims | `results/exp6...json`, `exp7...json`, `exp8...json`; all verified | Complete |
| Real-vs-imagined gap under multiple mismatch levels | `results/tables/exp5_gap_summary.csv`; severe and stuck/slip gap growth > 16 | Complete |
| Adaptive allocation under fixed budget | `results/exp6_adaptive_rollout_allocation.json`; moment-law beats uniform by 0.0767 | Complete |
| Closed-loop receding horizon | `results/exp7_closed_loop_receding_horizon_eval.json`; useful scorer improves, corrected first-action oracle diagnostic | Complete |
| Claims status script | `scripts/claims_status.py`; 9/9 verified | Complete |
| README/paper outline | `README.md`, `paper_outline.md` | Complete |
| No unsupported README claims | `claims_status.py` reports `readme_overclaims: []` | Complete |

## Verified Paper Claims

1. Exact finite theorem matches Monte Carlo: success MAE `0.00210`, utility MAE `0.01376`.
2. N=2 AUC identity: max error `0.0`.
3. Higher-N needs upper-tail moments: same-p/kappa N=64 gap `0.99882`.
4. Pilot-to-heldout prediction improves with K: relative MAE reduction `0.728`.
5. Score function controls inference value: best non-oracle beats random by `3.772` utility at N=64.
6. WAM mismatch amplifies imagined-real utility gaps: severe mismatch gap growth `16.726`; stuck/slip `16.388`.
7. Adaptive allocation beats uniform: moment-law improvement `0.0767` success at reference budget.
8. Closed-loop useful scorer improves with more rollouts: N=64 vs N=1 success gain `0.208`; beats random at N=64 by `0.236`.
9. Nonstationary conditional law: N=16 conditional-law MAE `0.00262`; pre/post p shift `0.168`.

## Remaining Scope Notes

- The project proves and validates inference-time laws for fixed rollout distributions and controlled WAM-lite dynamics.
- It does not claim real-robot generalization or a universal WAM training-scaling law.
- Oracle scorers are diagnostic upper bounds, not deployable policies.

