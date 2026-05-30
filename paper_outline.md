# Paper Outline

## 1. Introduction

Robotic planners increasingly use world-action models to sample candidate futures at test time. This paper asks a narrower question than WAM training: given a fixed rollout generator and scorer, how much real utility should additional imagined rollouts buy?

## 2. Related Work

Cover best-of-N and test-time compute, world models and world-action models, random shooting and CEM-style MPC, model-based RL, verifier/scorer alignment, and planner exploitation under model error.

## 3. Problem Setup

Define state, rollout/action sequence, imagined dynamics, true dynamics, rollout score `S`, binary success `Y`, real utility `R`, and best-of-N top-score selection. Separate the exact known-distribution law from finite pilot estimation.

## 4. Binary Rollout Inference Theorem

State the finite empirical tie-aware law, the continuous-distribution form, `N=1`, all-success/no-success edge cases, and random tie handling. Prove the `N=2` AUC identity:

```text
f_2 = p^2 + 2p(1-p)kappa
```

Then show the moment hierarchy `f_N = Np E[U^(N-1)]`, where `U = F_mix(S+)`, and explain why AUC is insufficient for `N > 2`.

## 5. Utility-Valued Inference Theorem

Replace binary success with real utility:

```text
V_N(x) = N E[R F_S(S)^(N-1)]
```

Use the finite tie-aware group formula for implementation. Binary success is a special case.

## 6. Receding-Horizon Corollary

At each visited state, the theorem applies conditionally to the rollout distribution sampled at that state. The controller samples `N` rollouts, chooses the top-scoring rollout, executes the first action, observes the next state, and replans. Do not claim a global closed-form law under arbitrary nonstationarity.

## 7. Pilot Estimation And Adaptive Allocation

Discuss finite pilot estimation, bootstrap uncertainty, pilot regret, fixed-budget rollout allocation across states, moment-law allocation, utility-valued marginal gains, and comparison to uniform and oracle allocation.

## 8. Experimental Setup

Analytic and learned toy environments:

- BlockPush2D
- DrawerPull1D
- SlipperyGrasp1D
- NonstationaryPhysicalShiftEnv
- DeformableToyEnv

Backends:

- analytic nominal WAM
- learned horizon WAM
- learned MLP dynamics WAM
- learned ensemble WAM
- oracle true dynamics

Scorers include random, predicted distance, predicted utility, predicted success, uncertainty-penalized utility, safety-penalized utility, learned scorers, oracle utility, and anti-real-utility falsification.

## 9. Results

Report only artifact-backed claims:

1. exact finite law validation
2. `N=2` AUC identity and high-N moment hierarchy
3. pilot-to-heldout prediction
4. score-function comparison
5. real-vs-imagined utility gap
6. adaptive allocation
7. closed-loop receding-horizon evaluation
8. nonstationary dynamics stress test
9. bad-scorer falsification
10. learned-vs-analytic-vs-oracle WAM comparison
11. multi-environment breadth
12. optional visual toy stress test

## 10. Benchmark Status

Gymnasium/MuJoCo Reacher-v5 is integrated as a state-based external benchmark fallback with rollout pools, learned WAM-lite training, exact-law validation, score comparison, real-vs-predicted gap, closed-loop evaluation, and RGB render sanity artifacts. ManiSkill, LIBERO, and RoboCasa remain engineering scaffolds until their dependencies and task adapters are completed.

## 11. Falsification And Limitations

Emphasize that more imagination helps only when scores align with real utility. Under model mismatch or a bad scorer, best-of-N can amplify hallucinated or unsafe futures. The paper does not claim real robot evidence, DreamZero/UWM-level integration, or a universal WAM training recipe.

## 12. Future Work

Robot Chinchilla: a universal WAM train-inference optimizer that jointly selects data scale, model class, rollout horizon, scorer quality, safety constraints, and test-time sampling budget.
