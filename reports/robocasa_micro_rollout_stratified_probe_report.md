# RoboCasa Micro-Rollout Probe

- status: `verified`
- candidate task IDs: `28`
- runnable task IDs: `28`
- nondegenerate task IDs: `23`
- rollouts per task: `2`
- horizon: `1`
- total wall-clock seconds: `4305.403089284897`

## Runnable Task IDs

- `robocasa/CleanMicrowave`
- `robocasa/ClearSink`
- `robocasa/ClearCuttingBoard`
- `robocasa/RinseCuttingBoard`
- `robocasa/RinseSinkBasin`
- `robocasa/WashFruitColander`
- `robocasa/WashLettuce`
- `robocasa/CollectWashingSupplies`
- `robocasa/MicrowavePressButton`
- `robocasa/PreheatOven`
- `robocasa/SlideOvenRack`
- `robocasa/SlideToasterOvenRack`
- `robocasa/ToastBagel`
- `robocasa/AdjustHeat`
- `robocasa/LowerHeat`
- `robocasa/TransportCookware`
- `robocasa/FillKettle`
- `robocasa/FillBlenderJug`
- `robocasa/LoadDishwasher`
- `robocasa/EmptyDishRack`
- `robocasa/LoadFridgeByType`
- `robocasa/ArrangeDrinkware`
- `robocasa/GatherVegetables`
- `robocasa/OrganizeCondiments`
- `robocasa/TurnOnMicrowave`
- `robocasa/TurnOnToaster`
- `robocasa/TurnOnToasterOven`
- `robocasa/TurnOnElectricKettle`

This is a reset/clone/short-rollout viability probe. It does not promote these task IDs to learned-WAM, exact-law, closed-loop, or solved-policy evidence; those require the heavier rollout-pool and CI artifacts.
