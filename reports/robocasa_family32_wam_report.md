# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender', 'robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter', 'robocasa/CloseFridgeDrawer', 'robocasa/TurnOnBlender', 'robocasa/PickPlaceCabinetToCounter', 'robocasa/PickPlaceSinkToCounter']`
- train samples: `512`
- validation samples: `256`
- eval samples: `512`
- eval rollout pools: `64`
- exact-law utility MAE: `0.0002689211543089739`
- validation utility correlation: `0.8384486971259033`
- validation learned-physics correlation: `0.838448697125904`
- promoted learned scorer: `learned_energy_regularized`
- promoted scorer minus random N8 CI: `{'n': 64, 'mean': 0.2881867956710381, 'std': 0.247375612716181, 'stderr': 0.030921951589522625, 'ci95': 0.06060702511546434, 'lo': 0.22757977055557377, 'hi': 0.34879382078650245}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
