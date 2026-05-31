# RoboCasa Learned WAM Report

- status: `verified`
- env: `robocasa/PickPlaceCounterToCabinet`
- train samples: `80`
- validation samples: `32`
- eval rollout pools: `5`
- exact-law utility MAE: `0.0002343183527097224`
- validation utility correlation: `0.7639608394479505`
- learned minus random N8 CI: `{'n': 5, 'mean': 0.14832993183038654, 'std': 0.11178217873717447, 'stderr': 0.04999051006587074, 'ci95': 0.09798139972910665, 'lo': 0.05034853210127989, 'hi': 0.2463113315594932}`

This is a lightweight state/action-sequence RoboCasa WAM-lite artifact. It is promoted only if the heldout learned scorer beats random with a positive CI.
