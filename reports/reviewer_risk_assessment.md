# Reviewer Risk Assessment

## Strongest Points

- The mathematical law is exact for the fixed score/utility distribution and implemented with finite tie handling.
- The repo includes falsification: high N can hurt when the scorer is bad or dynamics predictions are unaligned.
- Learned WAM-lite evidence exists rather than only analytic nominal dynamics.
- Multi-env toy breadth tests friction, stuckness, grasp slip, and nonstationarity.

## Main Reviewer Attacks

- The empirical work is still toy-scale and state-based.
- The current external benchmark is Reacher-v5 only; reviewers may ask for ManiSkill, LIBERO, or harder contact-rich tasks.
- The learned models are intentionally lightweight and do not establish WAM training recipes.
- Pilot estimates are not exact laws and can be brittle under shift.
- Some analytic smoke artifacts are single-seed checks; paper figures should prefer five-seed learned/multi-env results.

## Evidence That Helps

- Exact theorem tests and docs separate identities from empirical predictions.
- Learned toy artifacts report ID and OOD errors.
- The anti-overclaim claim gate prevents unsupported real-robot or unavailable-benchmark claims from slipping into README/paper text.
- Falsification experiments make the score-alignment condition explicit.

## Remaining Gap

The single highest reviewer-risk gap is absence of a harder manipulation benchmark or real-robot artifact beyond Gymnasium/MuJoCo Reacher-v5.
