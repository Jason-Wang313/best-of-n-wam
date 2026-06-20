# V6 Evidence Plan

V6 turns the V5 audit into a real-benchmark, CPU-reproducible hardening pass.
It uses existing committed rollout-curve artifacts; it does not rerun optional
simulators, claim real-robot validation, claim GPU-scale training, or promote a
new broad robotics leaderboard result.

## Scope

V6 promotes only this claim:

> On existing simulated real-benchmark rollout-pool curve artifacts, a
> conservative score-tail audit can be frozen, transferred across benchmark
> families, calibrated with abstention, stress-tested by negative controls, and
> reported with compute/label accounting.

V6 must not claim:

- real-world robot or HIL validation;
- modern VLA-scale training or inference;
- broad robotics state of the art;
- a universal WAM training law;
- that curve-level evidence replaces candidate-level or hardware validation.

## Low-RAM Rules

- Read one CSV artifact at a time and keep only compact per-pool summaries.
- Do not materialize candidate tensors or simulator states.
- Use existing selected-utility curves as the source of truth.
- Write all promoted evidence as CSV/JSON under `results/v6/`.
- Smoke outputs go under `results/v6_smoke/` and must not overwrite canonical
  evidence.
- Canonical V6 may use all committed curve rows because the data are already
  compact CSV summaries.

## Required Evidence

1. Real-benchmark prospective audit over existing curve artifacts.
2. Leave-one-family-out frozen transfer.
3. Hash-locked predictions before outcome merge.
4. Calibration and abstention analysis.
5. Selector/metric ablation ladder.
6. Robustness grid over thresholds, pilot budgets, and confidence radii.
7. Negative controls on real benchmark artifacts.
8. Compute-normalized reporting with rollout and label accounting.
9. Finite-sample audit theory table.
10. Reviewer-attack appendix and claim-ledger integration.

## Verification

Required commands:

```bash
python experiments/v6_real_benchmark_evidence.py
python experiments/v6_frozen_evidence.py
python scripts/reproduce_v6_cpu.py
python scripts/run_v6_claim_audit.py
python scripts/build_v6_paper.py
pytest
```

The goal is complete only when the V6 evidence, manuscript text, tests, claim
ledger, audit scripts, and final PDF all pass with scoped claims.
