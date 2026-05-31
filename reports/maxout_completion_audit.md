# Max-Out Completion Audit

Audit date: 2026-05-30.

## Execution Tier

Benchmark-visual validated: theorem layer, learned toy, multi-env toy, Gymnasium/MuJoCo Reacher-v5 benchmark, Gymnasium Robotics Fetch benchmark, Meta-World ML1, RoboSuite Panda benchmark, ManiSkill3 state-mode benchmark, toy visual mode, Reacher RGB WAM-lite, and Fetch RGB WAM-lite.

## Artifact Coverage

- Environments: `block_push, drawer_pull, slippery_grasp, nonstationary_shift, deformable_toy`.
- Learned backbones: `horizon_wam, mlp_dynamics_wam, ensemble_wam`.
- Multi-env seeds: `5`.
- Benchmark attempted: `True`; any benchmark available: `True`.
- Benchmark suite: `Reacher-v5`; rollout pools: `25`; exact-law MAE: `0.0188`.
- Gymnasium Robotics suite: `['FetchReach-v4', 'FetchPush-v4', 'FetchPickAndPlace-v4']`; rollout pools: `60`; exact-law MAE: `0.0126`; learned-random N32 CI lower: `0.4409`.
- Meta-World suite: `['reach-v3', 'push-v3', 'drawer-open-v3']`; rollout pools: `45`; exact-law MAE: `0.0298`; learned-random N32 CI lower: `0.0975`; reward-random N32 CI lower: `0.4666`.
- RoboSuite suite: `['Lift', 'Stack', 'Door']`; rollout pools: `30`; exact-law MAE: `0.0024`; learned-random N32 CI lower: `0.2447`; reward-random N32 CI lower: `0.2617`; closed-loop learned-random N8 CI lower: `0.0798`.
- ManiSkill suite: `['PickCube-v1', 'PushCube-v1', 'PegInsertionSide-v1']`; rollout pools: `30`; exact-law MAE: `0.0034`; control: `pd_joint_delta_pos`.
- Visual attempted: `True`; visual verified: `True`.
- Benchmark visual verified: `True`.
- Benchmark RGB WAM-lite: `extra_trees_visual_wam`; verified: `True`; utility corr: `0.2199`; utility MAE: `0.5208`; exact-law MAE: `0.0157`.
- Gymnasium Robotics RGB WAM-lite: verified: `True`; mean utility corr: `0.7325`; exact-law MAE: `0.0106`; visual-random N32 CI lower: `0.3475`.
- ManiSkill visual/EE probe: attempted `True`; state baseline ok `True`; any visual success `False`; blocker `vk::Device::allocateDescriptorSetsUnique: ErrorOutOfPoolMemory`.
- ManiSkill dependency probe: Pinocchio import `False`; binary `pin` wheel `False`; source install attempted `True`.
- Inference audit tail/gain correlation: `0.9864`.
- Learned-backend inference audit present: `True`.
- Scorer repair N64 gain over predicted utility: `0.3489`.
- Compute frontier predicted N128-N1 gain: `0.0255`.

## Acceptance Status

- Pytest: run by the final execution sequence.
- Smoke: run by the final execution sequence.
- Learned WAM toy: run by the final execution sequence.
- Multi-env: artifacts cover BlockPush2D, DrawerPull, SlipperyGrasp, and Nonstationary.
- Backbones: MLP, horizon, and ensemble WAM artifacts are present.
- EXP10: anti-scorer and randomized-dynamics falsification artifacts are present when multi-env is regenerated.
- Benchmark: Gymnasium/MuJoCo Reacher-v5 artifacts generated.
- Gymnasium Robotics: FetchReach-v4, FetchPush-v4, and FetchPickAndPlace-v4 artifacts generated.
- Meta-World: reach-v3, push-v3, and drawer-open-v3 ML1 artifacts generated.
- RoboSuite: Lift, Stack, and Door Panda manipulation artifacts generated, including small closed-loop traces.
- ManiSkill: PickCube-v1, PushCube-v1, and PegInsertionSide-v1 state-mode artifacts generated.
- Visual: toy visual mode verified with MAE `0.0185`.
- Benchmark visual WAM: Reacher-v5 RGB-frame/action-sequence model verified with visual-random N32 CI lower bound `0.1998`.
- Gymnasium Robotics visual WAM: Fetch RGB-frame/action-sequence models verified with visual-random N32 CI lower bound `0.3475`.
- ManiSkill visual/EE-control probe: generated artifact-backed blocker report when local RGB/RGB-D and EE-control attempts failed.
- Audit framework: inference-value profiles, deployment gates, scorer repair, and compute frontiers generated.
- README overclaims: `0`.

## Key Numerical Results

- EXP1 success MAE: `0.0021`.
- EXP1 utility MAE: `0.0138`.
- EXP2 max AUC identity error: `0.00000000`.
- EXP2 same-p/kappa N64 gap: `0.9988`.
- EXP3 relative MAE reduction: `0.3698`.
- EXP4 oracle-random N64 utility gap: `6.6877`.
- EXP5 severe mismatch gap growth: `15.1800`.
- EXP6 moment-law improvement over uniform: `0.0333`.
- EXP7 learned useful N64-N1 success gain: `0.2167`.
- EXP8 conditional-law MAE: `0.0063`.
- Gymnasium Robotics Fetch exact-law MAE: `0.0126`.
- Meta-World exact-law MAE: `0.0298`.
- RoboSuite exact-law MAE: `0.0024`.
- Falsification anti-scorer N64 mean utility: `-26.5657`.
