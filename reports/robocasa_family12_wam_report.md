# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven']`
- train samples: `96`
- validation samples: `96`
- eval samples: `192`
- eval rollout pools: `24`
- exact-law utility MAE: `0.0002923520485543551`
- validation utility correlation: `0.8330260116378324`
- validation learned-physics correlation: `0.8330260116378323`
- promoted learned scorer: `learned_energy_regularized`
- promoted scorer minus random N8 CI: `{'n': 24, 'mean': 0.2736319931949471, 'std': 0.2264236150811513, 'stderr': 0.04621852688876389, 'ci95': 0.09058831270197723, 'lo': 0.18304368049296985, 'hi': 0.36422030589692433}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
