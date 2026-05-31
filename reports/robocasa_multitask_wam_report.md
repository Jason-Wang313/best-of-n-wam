# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/PickPlaceCounterToCabinet', 'robocasa/PickPlaceCounterToDrawer', 'robocasa/PickPlaceCounterToMicrowave']`
- train samples: `144`
- validation samples: `96`
- eval samples: `240`
- eval rollout pools: `15`
- exact-law utility MAE: `0.000347041413626855`
- validation utility correlation: `0.6751791364461345`
- validation learned-physics correlation: `0.6751791364461309`
- promoted learned scorer: `learned_energy_regularized`
- promoted scorer minus random N8 CI: `{'n': 15, 'mean': 0.23492695125782703, 'std': 0.13017328673923353, 'stderr': 0.03361059811080895, 'ci95': 0.06587677229718554, 'lo': 0.1690501789606415, 'hi': 0.30080372355501256}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
