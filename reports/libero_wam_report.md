# LIBERO WAM Report

- status: `verified`
- tasks: `['libero_spatial/0', 'libero_spatial/1', 'libero_spatial/2']`
- train samples: `192`
- validation samples: `96`
- eval samples: `240`
- eval rollout pools: `15`
- exact-law utility MAE: `0.0001402348651464976`
- validation utility correlation: `0.3526483541014925`
- validation learned-physics correlation: `0.3526483541014941`
- promoted learned scorer: `learned_energy_regularized`
- promoted scorer minus random CI: `{'n': 15, 'mean': 0.3375863022441559, 'std': 0.14290935532341625, 'stderr': 0.03689903687898863, 'ci95': 0.07232211228281771, 'lo': 0.2652641899613382, 'hi': 0.40990841452697363}`

This is an optional LIBERO state/action-sequence WAM-lite artifact. The dense utility is task local progress plus sparse success/reward, so it should be cited as LIBERO rollout-pool validation rather than solved-task policy performance.
