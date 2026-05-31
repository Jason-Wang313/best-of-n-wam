# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/OpenDrawer', 'robocasa/OpenCabinet', 'robocasa/OpenMicrowave', 'robocasa/TurnOnSinkFaucet']`
- train samples: `64`
- validation samples: `32`
- eval samples: `128`
- eval rollout pools: `16`
- exact-law utility MAE: `0.00035677762415927425`
- validation utility correlation: `0.8598503519742565`
- validation learned-physics correlation: `0.8598503519742552`
- promoted learned scorer: `learned_wam`
- promoted scorer minus random N8 CI: `{'n': 16, 'mean': 0.3045073082302883, 'std': 0.19051552386059456, 'stderr': 0.04762888096514864, 'ci95': 0.09335260669169133, 'lo': 0.211154701538597, 'hi': 0.39785991492197964}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
