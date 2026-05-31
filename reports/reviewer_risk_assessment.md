# Reviewer Risk Assessment

## Strongest Points

- The mathematical law is exact for the fixed score/utility distribution and implemented with finite tie handling.
- The repo includes falsification: high N can hurt when the scorer is bad or dynamics predictions are unaligned.
- Learned WAM-lite evidence exists rather than only analytic nominal dynamics.
- Multi-env toy breadth tests friction, stuckness, grasp slip, and nonstationarity.

## Main Reviewer Attacks

- The empirical work is mostly state-based, but benchmark visual WAM evidence now exists for Gymnasium/MuJoCo Reacher-v5 and Gymnasium Robotics Fetch RGB frames; ManiSkill RGB/RGB-D remains unavailable locally and is documented by a generated renderer probe.
- The strongest contact-rich external evidence is Gymnasium Robotics Fetch plus Meta-World ML1 and ManiSkill state mode, but still not real hardware.
- The learned models are intentionally lightweight and do not establish WAM training recipes.
- Pilot estimates are not exact laws and can be brittle under shift.
- Some analytic smoke artifacts are single-seed checks; paper figures should prefer five-seed learned/multi-env results.

## Evidence That Helps

- Exact theorem tests and docs separate identities from empirical predictions.
- Learned toy artifacts report ID and OOD errors.
- The anti-overclaim claim gate prevents unsupported real-robot or unavailable-benchmark claims from slipping into README/paper text.
- Falsification experiments make the score-alignment condition explicit.

## Remaining Gap

The single highest reviewer-risk gap is absence of LIBERO/RoboCasa or real-robot evidence beyond the current Gymnasium Robotics, Meta-World, and ManiSkill state-mode suites.
