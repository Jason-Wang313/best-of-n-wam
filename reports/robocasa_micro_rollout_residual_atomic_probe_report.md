# RoboCasa Micro-Rollout Probe

- status: `verified`
- candidate task IDs: `25`
- runnable task IDs: `13`
- nondegenerate task IDs: `2`
- rollouts per task: `2`
- horizon: `1`
- total wall-clock seconds: `2547.0286798477173`

## Runnable Task IDs

- `robocasa/CloseBlenderLid`
- `robocasa/CloseElectricKettleLid`
- `robocasa/CloseStandMixerHead`
- `robocasa/CloseToasterOvenDoor`
- `robocasa/OpenBlenderLid`
- `robocasa/OpenElectricKettleLid`
- `robocasa/OpenStandMixerHead`
- `robocasa/OpenToasterOvenDoor`
- `robocasa/TurnOffSimmeredSauceHeat`
- `robocasa/TurnOnElectricKettle`
- `robocasa/TurnOnToasterOven`
- `robocasa/TurnSinkSpout`
- `robocasa/ManipulateDrawer`

This is a reset/clone/short-rollout viability probe. It does not promote these task IDs to learned-WAM, exact-law, closed-loop, or solved-policy evidence; those require the heavier rollout-pool and CI artifacts.
