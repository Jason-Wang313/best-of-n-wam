# Ablation Report

## Environments

- block_push
- drawer_pull
- slippery_grasp
- nonstationary_shift
- deformable_toy

## Backbones

- horizon_wam
- mlp_dynamics_wam
- ensemble_wam

## Main Ablation Axes

- Analytic nominal versus learned WAM versus oracle true dynamics.
- Random, distance, utility, safety-penalized, uncertainty-penalized, oracle, and anti-real-utility scorers.
- Mild versus severe mismatch.
- Low N versus high N, especially N64-N1.

## Current Interpretation

Oracle scoring remains the diagnostic upper bound. Learned backbones reproduce several inference-value effects, but their gains vary by environment and scorer alignment. The multi-env suite should be treated as robustness evidence for the inference law and failure modes, not as proof of real manipulation performance.
