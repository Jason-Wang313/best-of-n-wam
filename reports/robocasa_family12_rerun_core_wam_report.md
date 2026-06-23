# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet', 'robocasa/CloseDrawer', 'robocasa/CloseCabinet', 'robocasa/CloseMicrowave', 'robocasa/TurnOffSinkFaucet', 'robocasa/TurnOnStove', 'robocasa/TurnOffStove', 'robocasa/OpenOven', 'robocasa/CloseOven']`
- train samples: `48`
- validation samples: `48`
- eval samples: `96`
- eval rollout pools: `12`
- exact-law utility MAE: `0.00030818269764661853`
- validation utility correlation: `0.8060656835290034`
- validation learned-physics correlation: `0.806065683529003`
- promoted learned scorer: `learned_wam`
- promoted scorer minus random N8 CI: `{'n': 12, 'mean': 0.26498249058017326, 'std': 0.2571750269472579, 'stderr': 0.07424003551842431, 'ci95': 0.14551046961611167, 'lo': 0.11947202096406159, 'hi': 0.4104929601962849}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
