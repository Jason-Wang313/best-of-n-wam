# RoboCasa Smoke Report

- status: verified smoke
- env: `robocasa/PickPlaceCounterToCabinet`
- split: `pretrain`
- rollout pools: `5`
- total rollouts: `80`
- exact-law utility MAE: `0.0002724664778796843`
- oracle minus random at N8 CI: `{'n': 5, 'mean': 0.2737591862068588, 'std': 0.13943158358342508, 'stderr': 0.06235569982059644, 'ci95': 0.12221717164836901, 'lo': 0.15154201455848978, 'hi': 0.3959763578552278}`
- distance-progress minus random at N8 CI: `{'n': 5, 'mean': 0.14581171415765518, 'std': 0.17153354068180754, 'stderr': 0.07671213147714945, 'ci95': 0.1503557776952129, 'lo': -0.004544063537557735, 'hi': 0.29616749185286806}`

This is a contact-rich RoboCasa kitchen reset/rollout smoke artifact. The oracle smoke gap is supported by the CI; the simple distance-progress scorer is positive on average but not promoted when its CI crosses zero. This is not a full RoboCasa learned-WAM benchmark or closed-loop validation.
