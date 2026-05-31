# LIBERO Visual-Language BC Policy Report

This optional artifact evaluates a low-dimensional RGB/proprio/language behavior-cloned kNN policy on LIBERO Object tasks. The policy receives rendered `agentview` and wrist RGB features, robot proprioception, task language, previous action memory, and a finite-horizon step clock.

## Summary

- Available: `True`.
- Verified: `True`.
- Train action examples: `6450`.
- Eval episodes: `30`.
- Eval successes: `30`.
- Eval success rate: `1.0` with bootstrap CI [`1.0`, `1.0`].

## Claim Boundary

- This uses RGB observations and task language, but it is still a lightweight feature-kNN behavior clone, not a modern vision-language policy.
- It does not use simulator object state, scripted phase labels, task IDs, or commanded target points at evaluation time.
- It is time-conditioned and limited to the Object tasks where the scripted controller produced successful demonstrations.
