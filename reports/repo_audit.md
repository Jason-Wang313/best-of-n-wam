# Audit Of `inference-value-theorem`

Source inspected: local checkout at `C:\Users\wangz\Downloads\inference-value-theorem` plus the public GitHub repository.

## Reused Conceptually

- Exact finite empirical best-of-N selector law with random tie-breaking.
- Tie-aware rank interval formula over sorted score groups.
- N=2 AUC/kappa identity: `f_2 = p^2 + 2p(1-p)kappa`.
- Moment hierarchy viewpoint: high-N selection depends on upper-tail rank moments, not AUC alone.
- Pilot-to-heldout forecasting as a separate statistical estimation problem.
- Adaptive allocation by estimated marginal inference value under a fixed sample budget.
- Artifact-driven claim checking before paper-facing claims are upgraded.

## Rewritten For Robotics/WAM

- Rollout data model: action sequences, imagined dynamics metrics, true dynamics metrics, real utility, real success.
- CPU-only `BlockPush2D` environment with hidden mass, friction, slip, and stuckness.
- WAM/planner scorers: predicted goal distance, predicted utility, predicted success, uncertainty/safety penalties, random score, oracle real utility.
- Experiments for score alignment, real-vs-imagined utility gaps, closed-loop receding-horizon planning, and nonstationary dynamics.
- Tests, README, paper outline, result schemas, and claim-status script.

## Removed

- MATH answer parsing and correctness grading.
- Text response caches and mean-logprob scoring defaults.
- NIM client, API-key loading, provider/model catalogues, and live LLM judge code.
- Benchmark-specific logic for GPQA, IFEval, LiveBench, LiveCodeBench, and MATH500.
- Private path fallbacks and cloud/API dependencies.

## Claim Boundaries

- The finite empirical law is exact for a fixed rollout pool or a known conditional rollout score/utility distribution.
- Continuous plug-in formulas are diagnostics; tie-aware finite formulas are the implementation source of truth.
- Pilot-to-heldout prediction is not exact by theorem. It is a statistical estimation problem whose reliability depends on pilot size and distribution shift.
- Adaptive allocation is only supported when heldout artifacts show improvement over uniform under the same budget.
- Closed-loop claims are conditional: at each visited state, the theorem applies to the rollout distribution sampled from that state. The project does not claim a closed-form global law for arbitrary nonstationary robotics.
