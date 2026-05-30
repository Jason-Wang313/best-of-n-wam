# Codex Handoff

Updated: 2026-05-30

## Current Goal

The current user action is `$context-handoff save current state other than push to github repo`: preserve verified project state before context is cleared. Do not push to GitHub as part of this handoff.

Previous user requested pushing everything to `https://github.com/Jason-Wang313/best-of-n-wam`, then interrupted and requested this handoff instead. No push was performed during the handoff update.

Project goal verified from repo docs: `Best-of-N WAM` is a standalone Python research project for exact test-time inference laws for World-Action Model rollout planning.

## Repo Facts Verified From Files

- Repo path: `C:\Users\wangz\best-of-n-wam`.
- Project display name in `README.md`: `Best-of-N WAM`.
- Python project name in `pyproject.toml`: `best-of-n-wam`.
- Python import package remains `wam_inference_value`.
- `README.md` describes the paper subtitle: `How Much Should a Robot Imagine? Exact Test-Time Inference Laws for World-Action Planning.`
- `AGENTS.md` says this repo uses the personal `context-handoff` skill and stores handoffs at `docs/CODEX_HANDOFF.md`.
- Git status is `## No commits yet on master`; all project files are currently untracked.
- `git remote -v` produced no output; no Git remote is currently configured.
- README says ManiSkill is optional and the adapter is a readiness skeleton unless benchmark artifacts exist.
- `scripts/claims_status.py` gates the 9 canonical analytic claims from analytic artifact filenames.

## Important Current Files

- `src/wam_inference_value/theorem.py`: finite tie-aware theorem implementations.
- `src/wam_inference_value/envs/block_push_2d.py`: CPU-only BlockPush2D environment.
- `src/wam_inference_value/learned_wam.py`: CPU-only learned WAM-lite ridge model, BlockPush2D train/validation/OOD dataset generation, model save/load, learned prediction/evaluation.
- `src/wam_inference_value/rollouts.py`: rollout pool construction with `dynamics_backend` values `analytic`, `learned`, and `oracle_true`.
- `src/wam_inference_value/evaluation.py`: shared experiment helpers, normalized utility metrics, seed/CI helpers, learned-backend CLI helpers.
- `src/wam_inference_value/benchmarks/maniskill_adapter.py`: optional ManiSkill adapter skeleton with `reset_task()`, `sample_rollouts()`, `score_rollouts()`, `evaluate_real_success()`, and `run_closed_loop()`.
- `experiments/train_learned_wam_lite.py`: trains learned WAM-lite and writes model/dataset/metric artifacts.
- `experiments/learned_wam_vs_analytic_wam.py`: compares analytic nominal WAM, learned WAM-lite, and oracle true dynamics.
- Experiments 1, 4, 5, 6, and 7 accept learned backend options.
- `scripts/run_learned_wam_toy.sh`: trains learned WAM-lite and runs learned toy experiments over 5 seeds.
- `tests/test_learned_wam.py` and `tests/test_maniskill_adapter.py`: learned backend and optional adapter tests.

## Commands And Results Verified

- `python -m pytest -q`
  - Result from final validation before handoff: `17 passed, 1 skipped`.
  - The skipped test is the optional ManiSkill runtime path when ManiSkill is not installed.
- `bash scripts/run_smoke.sh`
  - Result from final validation before handoff: completed successfully.
  - It refreshed canonical analytic artifacts and `scripts/claims_status.py` reported 9 verified, 0 partial, 0 unsupported.
- `bash scripts/run_learned_wam_toy.sh`
  - Result from final validation before handoff: completed successfully.
  - It refreshed learned artifacts after the smoke run.
- `python scripts\claims_status.py`
  - Result verified during this handoff update: 9 verified, 0 partial, 0 unsupported.
  - Claim evidence printed:
    - Claim 1 success MAE `0.006959852139023545`, utility MAE `0.0451116620102325`.
    - Claim 2 max identity error `0.0`.
    - Claim 3 same-p/kappa N64 gap `0.9988209815422198`.
    - Claim 4 relative MAE reduction `0.43698115249631736`.
    - Claim 5 best non-oracle minus random N64 `3.7519062324000982`.
    - Claim 6 severe-none gap growth `16.435472824968038`, stuck-none `16.91232327094248`.
    - Claim 7 moment-uniform delta `0.0658970885777631`.
    - Claim 8 useful scorer `ideal_action`, useful N64-N1 `0.5`, useful-random N64 `0.25`.
    - Claim 9 MAE `0.005451802849936604`, p shift `0.1293402777777778`.

## Current Learned WAM-Lite Artifacts

Verified current learned training summary in `results/learned_wam_lite_training.json`:

- Model path: `results\models\learned_wam_lite_toy.npz`.
- ID mismatch: `mild`.
- Train dataset: 2048 samples, mismatch `mild`.
- Validation dataset: 768 samples, mismatch `mild`.
- OOD datasets: `severe`, `stuck_slip`, `nonstationary`, each 192 samples.
- Dataset artifacts currently present:
  - `results/datasets/learned_wam_lite_train.npz`
  - `results/datasets/learned_wam_lite_validation.npz`
  - `results/datasets/learned_wam_lite_ood_severe.npz`
  - `results/datasets/learned_wam_lite_ood_stuck_slip.npz`
  - `results/datasets/learned_wam_lite_ood_nonstationary.npz`
- Validation metrics: final position L2 MAE `0.11172250197070284`, utility MAE `0.8623689603284173`, utility correlation `0.8944539743322287`.

Verified current learned experiment summaries:

- `results/exp1_exact_rollout_law_validation_learned.json`
  - backend `learned`, `num_seeds=5`, success MAE `0.010342304092721358`, utility MAE `0.07191977453350686`.
- `results/exp4_score_function_comparison_learned.json`
  - backend `learned`, `num_seeds=5`, best non-oracle minus random N64 `5.978381498022797`; CI lower bound `4.8424544190936105`.
- `results/exp5_real_vs_imagined_utility_gap_learned.json`
  - backend `learned`, `num_seeds=5`, severe gap growth minus none `13.8807751738171`, stuck-slip gap growth minus none `13.911924009159284`.
  - CI lower bounds: severe `13.266704546605894`, stuck-slip `13.191108866198107`.
- `results/exp6_adaptive_rollout_allocation_learned.json`
  - backend `learned`, `num_seeds=5`, mismatch `severe`.
  - moment-law improvement over uniform `0.30066408929367955`; CI lower bound `0.27537836690635276`.
  - oracle improvement over uniform `0.3055113080821638`; CI lower bound `0.27631964284001653`.
- `results/exp7_closed_loop_receding_horizon_eval_learned.json`
  - backend `learned`, `num_seeds=5`, mismatch `mild`, useful scorer `learned_horizon_goal_hybrid`.
  - useful N64-N1 success gain `0.21666666666666667`; CI lower bound `0.11866666666666671`.
  - useful minus random success at N64 `0.31666666666666665`; CI lower bound `0.255552929349359`.
- `results/learned_wam_vs_analytic_wam.json`
  - compares `analytic_nominal_wam`, `learned_wam_lite`, and `oracle_true_dynamics`.
  - `num_seeds=5`, mismatch `mild`, scorer `predicted_utility`.
  - learned minus analytic N64 real utility mean `1.169836264042242`, CI lower bound `0.95101430403342`.
  - oracle minus learned N64 real utility mean `1.9951266518211337`, CI lower bound `1.830104099553172`.
  - oracle minus analytic N64 real utility mean `3.164962915863376`, CI lower bound `2.9960923410618494`.

## Known Caveats

- The repository still has no initial commit; all project files are untracked.
- No Git remote is configured. The GitHub URL requested earlier has not been added as `origin`.
- `tests/test_experiments_smoke.py` can rewrite canonical `results/` artifacts with tiny smoke settings. The final artifact order before this handoff was `bash scripts/run_smoke.sh` followed by `bash scripts/run_learned_wam_toy.sh`, then `python scripts\claims_status.py`, leaving canonical claims at 9 verified and learned artifacts in their supportive 5-seed state.
- The full-scale analytic results in `README.md` are recorded from `reports/completion_audit.md`; current canonical `results/` are smoke-scale but claim-gated verified.

## Open Questions

- Whether the user wants the repo pushed to `https://github.com/Jason-Wang313/best-of-n-wam`: requested before interruption, but not done yet.
- Whether to add/set `origin` to the GitHub URL: `UNKNOWN`.
- Whether to commit all current untracked project files before pushing: likely yes, but not yet done.
- Whether generated `results/` artifacts and model/data `.npz` files should be committed to GitHub: `UNKNOWN`.

## Next Recommended Steps

1. If the next task is GitHub publishing, decide whether to include generated artifacts/models/datasets in the commit.
2. If pushing everything literally, set `origin` to `https://github.com/Jason-Wang313/best-of-n-wam`, make an initial commit, and push `master` or rename/push `main` as desired.
3. Before any paper-facing claim update, run `python scripts\claims_status.py` and require 9 verified, 0 partial, 0 unsupported.
4. If tests are run after paper-facing artifacts are regenerated, remember that smoke tests can rewrite `results/`; rerun the desired artifact pipeline afterward.

Safe to clear after handoff is updated.
