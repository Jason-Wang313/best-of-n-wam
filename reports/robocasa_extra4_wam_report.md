# RoboCasa Multi-Task WAM Report

- status: `verified`
- tasks: `['robocasa/PickPlaceCounterToStandMixer', 'robocasa/PickPlaceCounterToToasterOven', 'robocasa/PickPlaceDrawerToCounter', 'robocasa/PickPlaceMicrowaveToCounter']`
- train samples: `64`
- validation samples: `32`
- eval samples: `128`
- eval rollout pools: `16`
- exact-law utility MAE: `0.0003025652032180008`
- validation utility correlation: `0.7993479173648391`
- validation learned-physics correlation: `0.7993479173648383`
- promoted learned scorer: `learned_wam`
- promoted scorer minus random N8 CI: `{'n': 16, 'mean': 0.389104640308091, 'std': 0.2902967972911619, 'stderr': 0.07257419932279048, 'ci95': 0.14224543067266934, 'lo': 0.24685920963542166, 'hi': 0.5313500709807604}`

This is a task conditioned RoboCasa WAM-lite artifact over multiple kitchen task IDs. It is promoted only if the exact-law check passes and a learned scorer beats random with a positive heldout CI.
