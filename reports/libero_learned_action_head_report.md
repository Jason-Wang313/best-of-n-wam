# LIBERO Learned Action-Head Smoke Report

This optional artifact imitates the successful LIBERO Object scripted controller with a learned continuous action head. The phase schedule and target points are still scripted, so this is a narrow learned-control smoke, not a learned autonomous LIBERO policy.

## Summary

- Available: `True`.
- Verified: `True`.
- Train episodes: `20`.
- Train action examples: `4924`.
- Eval episodes: `30`.
- Eval successes: `30`.
- Eval success rate: `1.0` with bootstrap CI [`1.0`, `1.0`].
- Action-head model: `knn`.
- Action MAE on collected train examples: `0.00019167935088328474`.

## Limitations

- The learned component is only the continuous action head.
- High-level phase ordering and target-point construction are still scripted.
- The default artifact evaluates all ten LIBERO Object tasks, not all LIBERO suites.
- The phase targets inherit hand-coded object-conditioned grasp heights from the scripted smoke; this is not learned policy discovery.
