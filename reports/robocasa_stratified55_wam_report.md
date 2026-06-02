# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender', 'robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter', 'robocasa/CloseFridgeDrawer', 'robocasa/TurnOnBlender', 'robocasa/PickPlaceCabinetToCounter', 'robocasa/PickPlaceSinkToCounter', 'robocasa/CleanMicrowave', 'robocasa/ClearSink', 'robocasa/ClearCuttingBoard', 'robocasa/RinseCuttingBoard', 'robocasa/RinseSinkBasin', 'robocasa/WashFruitColander', 'robocasa/WashLettuce', 'robocasa/CollectWashingSupplies', 'robocasa/MicrowavePressButton', 'robocasa/ToastBagel', 'robocasa/AdjustHeat', 'robocasa/LowerHeat', 'robocasa/TransportCookware', 'robocasa/FillKettle', 'robocasa/FillBlenderJug', 'robocasa/LoadDishwasher', 'robocasa/EmptyDishRack', 'robocasa/LoadFridgeByType', 'robocasa/ArrangeDrinkware', 'robocasa/GatherVegetables', 'robocasa/OrganizeCondiments', 'robocasa/TurnOnMicrowave', 'robocasa/TurnOnToaster']`
- train samples: `880`
- validation samples: `440`
- eval samples: `880`
- eval rollout pools: `110`
- exact-law utility MAE: `0.0002758583967078402`
- validation utility correlation: `0.832609714619887`
- validation learned-physics correlation: `0.8326097146198878`
- promoted learned scorer: `learned_wam`
- promoted scorer minus random N8 CI: `{'n': 110, 'mean': 0.32174102442400815, 'std': 0.2539765598768874, 'stderr': 0.024215714838790526, 'ci95': 0.04746280108402943, 'lo': 0.2742782233399787, 'hi': 0.3692038255080376}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
