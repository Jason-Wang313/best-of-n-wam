# RoboCasa Micro-Rollout Probe

- status: `verified`
- candidate task IDs: `64`
- runnable task IDs: `52`
- nondegenerate task IDs: `42`
- rollouts per task: `2`
- horizon: `1`
- total wall-clock seconds: `8531.369538545609`

## Runnable Task IDs

- `robocasa/OpenBlenderLid`
- `robocasa/CloseBlenderLid`
- `robocasa/OpenElectricKettleLid`
- `robocasa/CloseElectricKettleLid`
- `robocasa/OpenStandMixerHead`
- `robocasa/CloseStandMixerHead`
- `robocasa/OpenToasterOvenDoor`
- `robocasa/CloseToasterOvenDoor`
- `robocasa/ManipulateDrawer`
- `robocasa/ManipulateSinkFaucet`
- `robocasa/ManipulateStoveKnob`
- `robocasa/MoveFreezerToFridge`
- `robocasa/MoveFridgeToFreezer`
- `robocasa/MoveToCounter`
- `robocasa/MoveToFreezerDrawer`
- `robocasa/PickPlaceCoffee`
- `robocasa/PickPlaceFridgeDrawerToShelf`
- `robocasa/PickPlaceFridgeShelfToDrawer`
- `robocasa/PickPlaceStoveToCounter`
- `robocasa/PickPlaceToasterOvenToCounter`
- `robocasa/PickPlaceToasterToCounter`
- `robocasa/AirDryFruit`
- `robocasa/CleanBlenderJug`
- `robocasa/ClearClutter`
- `robocasa/ClearFoodWaste`
- `robocasa/CountertopCleanup`
- `robocasa/FoodCleanup`
- `robocasa/GatherProduceWashing`
- `robocasa/RinseBowls`
- `robocasa/RinseFragileItem`
- `robocasa/WashFish`
- `robocasa/WashInSaucepan`
- `robocasa/AdjustToasterOvenTemperature`
- `robocasa/GetToastedBread`
- `robocasa/HeatMug`
- `robocasa/KettleBoiling`
- `robocasa/MicrowaveCorrectMeal`
- `robocasa/PlaceMicrowaveSafeItem`
- `robocasa/PrepareToast`
- `robocasa/ReturnHeatedFood`
- `robocasa/ToastBaguette`
- `robocasa/TurnOffMicrowave`
- `robocasa/AddIceCubes`
- `robocasa/ArrangeVegetables`
- `robocasa/BlendIngredients`
- `robocasa/CoffeeSetupMug`
- `robocasa/DrainVeggies`
- `robocasa/ArrangeTea`
- `robocasa/CondimentCollection`
- `robocasa/BreadAndCheese`
- `robocasa/CerealAndBowl`
- `robocasa/AddToSoupPot`

This is a reset/clone/short-rollout viability probe. It does not promote these task IDs to learned-WAM, exact-law, closed-loop, or solved-policy evidence; those require the heavier rollout-pool and CI artifacts.
