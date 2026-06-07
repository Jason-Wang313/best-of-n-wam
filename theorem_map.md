# Theorem Map

## Finite Empirical Tie-Aware Law

### Statement

For a fixed planning state x, a fixed finite rollout pool of size m, planner scores S, and real utilities R, best-of-N top-score selection has an exact expected selected utility. Sampling is i.i.d. with replacement from the pool. The planner chooses the maximum-score sampled rollout. If multiple sampled rollouts share the maximum score, the planner breaks ties uniformly at random.

Sort score tie groups in ascending score order. A tie group g occupies 1-indexed ranks [r_min_g, r_max_g] and has mean real utility mean_R_g. Then:

```text
V_N(x) = sum_g mean_R_g * [(r_max_g / m)^N - ((r_min_g - 1) / m)^N].
```

### Assumptions

- Fixed planning state x.
- Finite empirical pool of size m.
- Sampling uniformly with replacement.
- N i.i.d. sampled rollouts.
- Each rollout has planner score S and real utility R.
- The selected rollout is the maximum-score sampled rollout.
- Maximum-score ties are broken uniformly at random.
- Score tie groups are sorted ascending.

### Interpretation

The value of sampling more rollouts is determined by how much real utility sits in the upper score tail.

### Proof Status

Appendix proof drafted. The repo includes finite-law implementation, unit tests, exact-enumeration/counterexample audit coverage, and artifact checks. The manuscript proof text should still receive a final human line audit before submission.

## Binary Success Special Case

Set R = Y with Y in {0, 1}. The same finite tie-aware law gives:

```text
f_N(x) = sum_g mean_Y_g * [(r_max_g / m)^N - ((r_min_g - 1) / m)^N].
```

Interpretation: the expected selected success probability is the same rank-tail law with group mean success replacing group mean utility.

## Continuous No-Tie Utility Formula

When the score distribution is continuous and has no ties:

```text
V_N(x) = N E[R F_S(S)^(N-1)].
```

Interpretation: a rollout contributes to selected utility in proportion to the probability that the other N-1 samples have lower score.

### Proof Status

Appendix derivation drafted. This formula is a population no-tie identity and must not be used as the finite tie-aware source of truth; its assumptions should still receive a final human audit before submission.

## Continuous No-Tie Binary Formula

Let p = P(Y = 1), let S_+ be a score conditioned on success, and let F_mix be the marginal score CDF. Under continuous no-tie assumptions:

```text
f_N = N p E[F_mix(S_+)^(N-1)].
```

Interpretation: binary success at high N depends on upper-tail rank moments of successful rollouts.

## N=2 Tie-Aware AUC Identity

For N = 2:

```text
f_2 = p^2 + 2p(1-p) kappa
```

where:

```text
kappa = P(S_+ > S_-) + 0.5 P(S_+ = S_-).
```

Interpretation: for two samples, success is either guaranteed by drawing two successes or decided by whether a success outranks a failure. Tie-aware AUC is therefore sufficient for N = 2.

### Proof Status

Appendix proof drafted. Repo tests verify this identity against the finite tie-aware law; keep the N=2-only boundary visible before submission.

## High-N Moment Hierarchy

For binary success under the continuous no-tie setup, define U = F_mix(S_+). Then:

```text
f_N = N p E[U^(N-1)].
```

Interpretation: AUC fixes only the first rank moment needed for N = 2. Larger N depends on higher upper-tail moments, so two scorers can share p and kappa but differ sharply at high N.

## Limitations

- No claim that best-of-N always improves real utility.
- No claim that AUC controls all N.
- No claim that the continuous formula is tie-aware for general N.
- No claim about training compute scaling.
- No claim that finite pilot estimates are exact theorem guarantees.
- No claim that the laws solve closed-loop nonstationary control globally.
