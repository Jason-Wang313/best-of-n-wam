# LIBERO Visual-Language BC Policy Report

This optional artifact evaluates a tiny neural RGB/proprio/language behavior-cloned policy on LIBERO Object tasks. The policy receives rendered `agentview` and wrist RGB features, robot proprioception, task language, previous action memory, and a finite-horizon step clock.

## Summary

- Available: `True`.
- Verified: `True`.
- Policy type: `tiny_neural_vla_behavior_cloning`.
- Train action examples: `253`.
- Eval episodes: `1`.
- Eval successes: `0`.
- Eval success rate: `0.0` with bootstrap CI [`0.0`, `0.0`].

## Claim Boundary

- This is a tiny neural visual-language-action style smoke, not VLA-scale pretraining or broad LIBERO policy evidence.
- It does not use simulator object state, scripted phase labels, task IDs, or commanded target points at evaluation time.
- It encodes task language as hashed text features and does not restrict evaluation by task ID or simulator object state.
- This tagged smoke artifact evaluates `1` task(s); it is auxiliary model-class evidence and does not replace the canonical all-task LIBERO artifact.
- Demonstrations come from the hand-coded object-tuned scripted controller.
