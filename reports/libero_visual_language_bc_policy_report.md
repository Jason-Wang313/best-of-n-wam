# LIBERO Visual-Language BC Policy Report

This optional artifact evaluates a low-dimensional RGB/proprio/language behavior-cloned kNN policy on LIBERO Object tasks. The policy receives rendered `agentview` and wrist RGB features, robot proprioception, task language, previous action memory, and a finite-horizon step clock.

## Summary

- Available: `True`.
- Verified: `True`.
- Train action examples: `7386`.
- Eval episodes: `30`.
- Eval successes: `28`.
- Eval success rate: `0.9333333333333333` with bootstrap CI [`0.8333333333333334`, `1.0`].

## Claim Boundary

- This uses RGB observations and task language, but it is still a lightweight feature-kNN behavior clone, not a modern vision-language policy.
- It does not use simulator object state, scripted phase labels, task IDs, or commanded target points at evaluation time.
- The default artifact evaluates all ten LIBERO Object tasks, not all LIBERO suites.
- Demonstrations come from the hand-coded object-tuned scripted controller.
