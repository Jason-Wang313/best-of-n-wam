# Falsification Report

## Bad Scorer

Anti-real-utility scoring is intentionally adversarial. Current artifacts report:

- N1 mean real utility under anti-scorer: `-14.1016`.
- N64 mean real utility under anti-scorer: `-26.5657`.

This supports the negative claim that more imagination can amplify a bad scorer.

## Randomized Dynamics

Randomized dynamics prediction is tracked separately when `scripts/run_multi_env.sh` is rerun:

- randomized dynamics N64 mean real utility: `-11.7522`.
- oracle true N64 mean real utility: `1.6417`.
- oracle-randomized gap: `13.3939`.

If these fields are missing, rerun `bash scripts/run_multi_env.sh`.
