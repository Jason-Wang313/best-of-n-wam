# Theory Notes

## Setup

At a robot state `x`, a rollout generator samples candidate action/future rollouts. Each rollout has a score `S` used by the planner and a real utility `R` measured under true dynamics. Best-of-N inference samples `N` rollouts and executes the top-score candidate, with uniform random tie-breaking.

## Finite Tie-Aware Law

For a finite empirical rollout pool of size `m`, sort scores in ascending order and group ties. A tie group `g` has mean utility `mean_R_g` and occupies 1-indexed ranks `[r_min_g, r_max_g]`. The exact expected selected utility is:

```text
V_N = sum_g mean_R_g * [(r_max_g / m)^N - ((r_min_g - 1) / m)^N].
```

Binary success is the special case `R in {0, 1}`. For `N=1`, the expression reduces to the empirical mean utility. All-success and no-success binary pools reduce to 1 and 0 respectively.

## Continuous Binary Form

Let `p = P(Y=1)` and let `S+` denote a score conditioned on success. With continuous scores and no ties:

```text
f_N = N p E[F_mix(S+)^(N-1)].
```

This is a useful population identity, but finite experiments use the tie-aware law above as the source of truth.

## N=2 AUC Identity

For `N=2`,

```text
f_2 = p^2 + 2p(1-p) kappa,
```

where `kappa = P(S+ > S-) + 0.5 P(S+ = S-)` is the tie-aware AUC. This identity is exact and is tested directly.

## Moment Hierarchy

Define `U = F_mix(S+)`. Then:

```text
f_N = Np E[U^(N-1)].
```

AUC fixes only the first rank moment needed for `N=2`. Higher `N` depends on higher upper-tail moments. Therefore two scorers can have the same `p` and `kappa` but very different high-N inference value.

## Utility-Valued Law

For real-valued rollout utility `R`, the continuous no-tie form is:

```text
V_N(x) = N E[R F_S(S)^(N-1)].
```

The finite tie-aware version replaces point utilities by tie-group mean utilities and rank interval masses. Binary success is exactly the special case `R=Y`.

## Receding-Horizon Corollary

At time `t`, the robot observes state `s_t`, samples `N` rollouts from the conditional rollout distribution, selects the top-score rollout, executes only the first action, observes `s_{t+1}`, and repeats. The theorem applies conditionally at each `s_t`. This does not imply a global closed form for arbitrary nonstationary closed-loop dynamics.

## Model-Error Amplification

If a scorer aligns with imagined utility `R_imagined` but not real utility `R_real`, increasing `N` can raise selected imagined utility while real utility saturates or drops. The law still predicts selected real utility exactly from the joint empirical distribution of score and real utility; the failure is score/utility misalignment, not the theorem.

## Adaptive Allocation

With a fixed total rollout budget across states, the exact or estimated inference curves define marginal gains for allocating the next rollout. Moment-law allocation greedily assigns samples to the state with largest predicted marginal value. This can dominate uniform allocation when pilot estimates are reliable, and it can fail when pilot estimates are stale or noisy.

## Pilot Estimation

The exact law assumes the score/utility distribution is known. Pilot-to-heldout prediction is a separate statistical estimation problem. Reports must include uncertainty, confidence intervals, and regret against oracle allocation rather than treating pilot estimates as theorem guarantees.

## Safety/Risk Scoring

Safety can be handled by defining utility with a safety penalty, by constraining selection to low-risk rollouts, or by scoring rollouts as `utility - lambda * risk`. High-N selection can amplify risky false positives if risk is not included in the score.
