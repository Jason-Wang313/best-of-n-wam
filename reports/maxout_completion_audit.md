# Max-Out Completion Audit

Audit date: 2026-05-30.

## Execution Tier

Benchmark-visual validated: theorem layer, learned toy, multi-env toy, Gymnasium/MuJoCo Reacher-v5 benchmark, toy visual mode, and benchmark RGB render sanity check.

## Artifact Coverage

- Environments: `block_push, drawer_pull, slippery_grasp, nonstationary_shift, deformable_toy`.
- Learned backbones: `horizon_wam, mlp_dynamics_wam, ensemble_wam`.
- Multi-env seeds: `5`.
- Benchmark attempted: `True`; any benchmark available: `True`.
- Benchmark suite: `Reacher-v5`; rollout pools: `25`; exact-law MAE: `0.0188`.
- Visual attempted: `True`; visual verified: `True`.
- Benchmark visual verified: `True`.

## Acceptance Status

- Pytest: run by the final execution sequence.
- Smoke: run by the final execution sequence.
- Learned WAM toy: run by the final execution sequence.
- Multi-env: artifacts cover BlockPush2D, DrawerPull, SlipperyGrasp, and Nonstationary.
- Backbones: MLP, horizon, and ensemble WAM artifacts are present.
- EXP10: anti-scorer and randomized-dynamics falsification artifacts are present when multi-env is regenerated.
- Benchmark: Gymnasium/MuJoCo Reacher-v5 artifacts generated.
- Visual: toy visual mode verified with MAE `0.0185`.
- README overclaims: `0`.

## Key Numerical Results

- EXP1 success MAE: `0.0070`.
- EXP1 utility MAE: `0.0451`.
- EXP2 max AUC identity error: `0.00000000`.
- EXP2 same-p/kappa N64 gap: `0.9988`.
- EXP3 relative MAE reduction: `0.4370`.
- EXP4 oracle-random N64 utility gap: `6.9769`.
- EXP5 severe mismatch gap growth: `16.4355`.
- EXP6 moment-law improvement over uniform: `0.0659`.
- EXP7 learned useful N64-N1 success gain: `0.2167`.
- EXP8 conditional-law MAE: `0.0063`.
- Falsification anti-scorer N64 mean utility: `-26.5657`.
