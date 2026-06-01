# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven', 'robocasa/OpenDishwasher', 'robocasa/CloseDishwasher', 'robocasa/OpenFridge', 'robocasa/CloseFridge', 'robocasa/OpenFridgeDrawer', 'robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave', 'robocasa/PickPlaceCounterToSink', 'robocasa/PickPlaceCounterToStove', 'robocasa/PickPlaceCounterToOven', 'robocasa/PickPlaceCounterToBlender']`
- train samples: `192`
- validation samples: `192`
- eval samples: `384`
- eval rollout pools: `48`
- exact-law utility MAE: `0.0002730373845004557`
- validation utility correlation: `0.8523480255953249`
- validation learned-physics correlation: `0.8523480255953249`
- promoted learned scorer: `learned_energy_regularized`
- promoted scorer minus random N8 CI: `{'n': 48, 'mean': 0.30556743688204, 'std': 0.2342383401351318, 'stderr': 0.03380939218288737, 'ci95': 0.06626640867845925, 'lo': 0.2393010282035808, 'hi': 0.37183384556049925}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
