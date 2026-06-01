# RoboCasa Micro-Rollout Probe

- status: `verified`
- candidate task IDs: `4`
- runnable task IDs: `4`
- nondegenerate task IDs: `4`
- rollouts per task: `2`
- horizon: `1`
- total wall-clock seconds: `650.4587645530701`

## Runnable Task IDs

- `robocasa/PickPlaceCounterToStandMixer`
- `robocasa/PickPlaceCounterToToasterOven`
- `robocasa/PickPlaceDrawerToCounter`
- `robocasa/PickPlaceMicrowaveToCounter`

This is a reset/clone/short-rollout viability probe. It does not promote these task IDs to learned-WAM, exact-law, closed-loop, or solved-policy evidence; those require the heavier rollout-pool and CI artifacts.
