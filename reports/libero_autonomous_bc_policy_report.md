# LIBERO Autonomous BC Policy Report

This optional artifact evaluates a low-dimensional behavior-cloned kNN policy on LIBERO Object tasks. The policy receives simulator state features, task ID, previous action memory, and a finite-horizon step clock; it does not receive scripted phase indices or commanded target points.

## Summary

- Available: `True`.
- Verified: `True`.
- Train action examples: `6450`.
- Eval episodes: `30`.
- Eval successes: `30`.
- Eval success rate: `1.0` with bootstrap CI [`1.0`, `1.0`].

## Claim Boundary

- This is low-dimensional simulator-state behavior cloning, not image-based or language-conditioned LIBERO.
- It does not use scripted phase labels or target-point commands at evaluation time.
- It is time-conditioned; this is stronger than a scripted target/action-head smoke but still not a broad robust autonomous LIBERO policy.
- It is limited to the Object tasks where the scripted controller produced successful demonstrations.
