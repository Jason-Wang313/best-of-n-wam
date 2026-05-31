# Paper Result Summary

## Abstract-Safe Claims

- Exact finite best-of-N rollout selection laws for binary success and utility.
- `N=2` AUC identity and high-N moment hierarchy.
- More imagination helps only when scores align with real utility.
- Under model mismatch or bad scoring, high-N selection can amplify hallucinated futures.
- Inference-value audits diagnose tail alignment, stop rules, scorer repair, and compute-quality frontiers from artifacts.
- Gymnasium/MuJoCo, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, RGB WAM-lite on Reacher-v5 and Fetch frames, ManiSkill3 state-mode, RoboCasa three-task pick-place-family plus broad four-task and 12-task kitchen learned-WAM artifacts, LIBERO Spatial three-task rollout-pool learned-WAM artifacts, LIBERO Object sparse-success scripted smoke, and LIBERO learned action-head smoke validate the benchmark path without claiming hardware evidence.
- ManiSkill RGB/RGB-D and EE-control attempts are documented as a blocker artifact, not counted as visual validation.
- ManiSkill Pinocchio dependency probing documents why EE-control is not claimed in this environment.
- Learned toy and multi-env toy artifacts support these claims with confidence intervals where the claim gate marks them verified.

## Discussion-Only Claims

- Autonomous learned sparse-success LIBERO policy performance and full RoboCasa-wide learned-WAM validation.
- ManiSkill beyond state-mode joint-delta control.
- ManiSkill RGB/RGB-D WAM validation.
- Universal WAM train-inference optimization.
- Any analogy to DreamZero/UWM-level evidence.

## Do Not Claim

- Real robot validation.
- Autonomous learned sparse-success LIBERO policy validation or full RoboCasa-wide validation.
- ManiSkill RGB/RGB-D or EE-control validation.
- A universal WAM training recipe.
- That increasing N is intrinsically beneficial.
