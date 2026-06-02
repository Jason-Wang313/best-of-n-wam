# Reviewer Risk Assessment

## Strongest Points

- The mathematical law is exact for the fixed score/utility distribution and implemented with finite tie handling.
- The repo includes falsification: high N can hurt when the scorer is bad or dynamics predictions are unaligned.
- Learned WAM-lite evidence exists rather than only analytic nominal dynamics.
- Multi-env toy breadth tests friction, stuckness, grasp slip, and nonstationarity.

## Main Reviewer Attacks

- The empirical work is mostly state-based, but benchmark visual WAM evidence now exists for Gymnasium/MuJoCo Reacher-v5 and Gymnasium Robotics Fetch RGB frames; ManiSkill RGB/RGB-D remains unavailable locally and is documented by a generated renderer probe.
- ManiSkill EE-control remains unavailable locally because Pinocchio is absent and the `pin` dependency path lacks binary wheels for this Windows/Python stack; this is documented by a generated dependency probe.
- The strongest contact-rich external evidence is Gymnasium Robotics Fetch plus Meta-World ML1, RoboSuite Panda, ManiSkill state mode, RoboCasa three-task pick-place, broad four-task atomic-manipulation, 12-task, 24-task family, and extra four-task artifacts, LIBERO Spatial rollout-pool validation, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned low-dimensional autonomous BC smoke, and LIBERO RGB/proprio/language BC smoke, but still not real hardware.
- The learned models are intentionally lightweight and do not establish WAM training recipes.
- Pilot estimates are not exact laws and can be brittle under shift.
- Some analytic smoke artifacts are single-seed checks; paper figures should prefer five-seed learned/multi-env results.

## Evidence That Helps

- Exact theorem tests and docs separate identities from empirical predictions.
- Learned toy artifacts report ID and OOD errors.
- The anti-overclaim claim gate prevents unsupported real-robot or unavailable-benchmark claims from slipping into README/paper text.
- Falsification experiments make the score-alignment condition explicit.

## Remaining Gap

The single highest reviewer-risk gap is absence of real-robot evidence, modern VLA-style benchmark policy validation, and full benchmark-suite policy coverage. LIBERO is now present as a three-task dense-utility rollout-pool WAM artifact plus sparse-success scripted, learned action-head, time-conditioned low-dimensional BC, and RGB/proprio/language Object smokes, but not full LIBERO policy evidence.
