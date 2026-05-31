# LIBERO Learned Action-Head Smoke Report

This optional artifact imitates the successful LIBERO Object scripted controller with a learned ridge action head. The phase schedule and target points are still scripted, so this is a narrow learned-control smoke, not a learned autonomous LIBERO policy.

## Summary

- Available: `True`.
- Verified: `True`.
- Train episodes: `12`.
- Train action examples: `2580`.
- Eval episodes: `18`.
- Eval successes: `18`.
- Eval success rate: `1.0` with bootstrap CI [`1.0`, `1.0`].
- Action MAE on collected train examples: `0.013739316733548198`.

## Limitations

- The learned component is only the continuous action head.
- High-level phase ordering and target-point construction are still scripted.
- The artifact evaluates the scripted-success LIBERO Object subset, not all LIBERO suites.
