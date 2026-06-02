# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender', 'robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter']`
- train samples: `448`
- validation samples: `224`
- eval samples: `448`
- eval rollout pools: `56`
- exact-law utility MAE: `0.0002738621850195251`
- validation utility correlation: `0.8335286692076076`
- validation learned-physics correlation: `0.8335286692076066`
- promoted learned scorer: `learned_wam`
- promoted scorer minus random N8 CI: `{'n': 56, 'mean': 0.30243909345319026, 'std': 0.2562827497116669, 'stderr': 0.034247222984335565, 'ci95': 0.0671245570492977, 'lo': 0.23531453640389255, 'hi': 0.36956365050248796}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
