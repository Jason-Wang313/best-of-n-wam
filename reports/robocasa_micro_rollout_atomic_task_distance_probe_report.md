# RoboCasa Micro-Rollout Probe

- status: `verified`
- candidate task IDs: `2`
- runnable task IDs: `2`
- nondegenerate task IDs: `2`
- rollouts per task: `2`
- horizon: `1`
- total wall-clock seconds: `539.2655725479126`

## Runnable Task IDs

- `robocasa/OpenElectricKettleLid`
- `robocasa/CloseElectricKettleLid`

This is a reset/clone/short-rollout viability probe. It does not promote these task IDs to learned-WAM, exact-law, closed-loop, or solved-policy evidence; those require the heavier rollout-pool and CI artifacts.
