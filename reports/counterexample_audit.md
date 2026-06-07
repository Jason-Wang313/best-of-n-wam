# Counterexample Audit: Best-of-N / WAM Finite Inference Law

Agent: Agent 2 hostile Counterexample / Fuzzer Agent

Scope: README, paper draft, theorem map, claim evidence map, theory notes, finite theorem implementation, and theorem/adaptive tests in the local `best-of-n-wam` repo. This audit treats the local folder as source of truth, not the GitHub URL.

## 1. Executive verdict

Verdict: theorem appears correct under the stated assumptions.

Secondary verdict: paper overclaims are still possible despite theorem correctness, mostly through wording drift around "scaling laws", "robot", "adaptive allocation", and simulator benchmark validation.

I did not find a counterexample to the finite theorem when all theorem assumptions are enforced:

- fixed state / fixed empirical rollout pool
- finite pool of size `m`
- uniform sampling with replacement
- `N` i.i.d. sampled rollouts
- each rollout has planner score `S` and real utility `R`
- maximum-score selection
- uniform random tie-breaking among sampled maximum-score rollouts
- score tie groups handled by group mean utility

The finite formula

```text
V_N(x) = sum_g mean_R_g * [(r_max_g / m)^N - ((r_min_g - 1) / m)^N]
```

matched exact enumeration over all `m^N` sampled tuples on the adversarial cases below. The strongest failures I could construct are not theorem failures; they are counterexamples to applying the theorem outside its assumptions.

## 2. Passed sanity cases

The repo already contains an exact enumerator in `tests/test_theorem_binary.py` named `brute_force`. It loops over all sampled tuples in `itertools.product(range(m), repeat=N)`, finds the maximum sampled score, averages utilities across tied top-score sampled positions, and averages over all `m^N` tuples.

I also used the same exact enumeration logic against the utility-valued implementation. Formula-vs-enumerator results:

| Case | Scores | Utilities / success | N values | Finite formula curve | Max absolute diff vs exact enumeration |
|---|---:|---:|---:|---:|---:|
| `N=1` | `[3,1,2,4]` | `[1,0,0,1]` | `[1]` | `{1: 0.5}` | `0` |
| all scores equal | `[0,0,0,0]` | `[1,-2,3,4]` | `[1,2,5]` | `{1: 1.5, 2: 1.5, 5: 1.5}` | `1.1e-15` |
| all utilities equal | `[0,1,2,3]` | `[7,7,7,7]` | `[1,4]` | `{1: 7.0, 4: 7.0}` | `0` |
| negative utilities | `[0,1,2]` | `[-1,-10,-100]` | `[1,2,5]` | `{1: -37.0, 2: -59.0, 5: -88.1111}` | `0` |
| binary utilities | `[0,1,1,2]` | `[0,1,0,1]` | `[1,2,4]` | `{1: 0.5, 2: 0.6875, 4: 0.83984375}` | `0` |
| many ties | `[0,0,1,1,1,2]` | `[5,-5,0,2,4,-10]` | `[1,3]` | `{1: -0.6667, 3: -3.1296}` | `1.8e-15` |
| top-score group has multiple rollouts | `[0,1,2,2]` | `[0,1,10,-10]` | `[1,3]` | `{1: 0.25, 3: 0.109375}` | `5.6e-17` |
| one high-score low-utility outlier | `[0,1,2,3]` | `[10,9,8,-100]` | `[1,2,8]` | `{1: -18.25, 2: -38.9375, 8: -89.1839}` | `0` |
| anti-correlated score and utility | `[0,1,2,3]` | `[4,3,2,1]` | `[1,2,8]` | `{1: 2.5, 2: 1.875, 8: 1.1040}` | `0` |
| random-looking independent small pool | `[0.3,-1.2,0.8,2.0,-0.4]` | `[5,-3,1,0,2]` | `[1,3]` | `{1: 1.0, 3: 1.144}` | `2.2e-16` |

Sanity-case conclusions:

- `N=1` reduces to empirical mean utility.
- All scores equal reduce to empirical mean utility for every `N`, because tie-breaking is uniform over the sampled top-score set and all sampled rollouts are in the only score group.
- All utilities equal stay constant for every `N`.
- Negative utilities are handled correctly; the theorem is linear in real utility and does not require nonnegativity.
- Binary success is exactly the utility-valued special case.
- Many ties and top-score multi-rollout groups are handled by the group mean, as required.
- A high-score bad outlier and anti-correlated score/utility do not falsify the theorem; they show the theorem predicts high-`N` degradation.

## 3. Failed or suspicious cases

These cases fail only when the theorem is used outside its assumptions.

### Without-replacement sampling mismatch

Example:

```text
scores    = [0, 1, 2, 3]
utilities = [0, 0, 0, 100]
N = 2
```

- With-replacement finite formula: `43.75`
- Exact without-replacement enumeration: `50.0`

This is an explicit counterexample to silently using the finite theorem for without-replacement candidate sampling. The paper must keep "with replacement" visible wherever the formula is stated or interpreted.

### Non-i.i.d. rollout sampling mismatch

Example:

```text
scores    = [0, 1, 2, 3]
utilities = [0, 10, 20, 30]
N = 2
first draw support  = {0, 1}
second draw support = {2, 3}
```

- I.i.d. finite formula: `21.25`
- Exact non-i.i.d. enumeration: `25.0`

This is an explicit counterexample to applying the theorem to adaptive, stateful, stratified, curriculum, or otherwise non-identically distributed rollout proposals without restating the distributional law.

### Continuous formula incorrectly applied to finite ties

Example:

```text
scores  = [0, 0, 1, 1]
success = [1, 0, 0, 1]
N = 3
```

- Finite tie-aware binary law: `0.5`
- Continuous plug-in estimate using right CDF ranks: `0.9375`

This is a large failure. It is not a theorem counterexample because the continuous formula assumes no score ties. It is a boundary hazard: empirical rollout pools with discrete/scored/tied values must use the finite tie-aware law.

## 4. Exact counterexamples if any

No exact counterexample found under the theorem assumptions.

Exact counterexamples to invalid stronger claims:

1. Without-replacement generalization is false.
2. Non-i.i.d. rollout generalization is false.
3. Continuous no-tie formula as a finite tied empirical formula is false.
4. AUC-only prediction for `N > 2` is false.
5. "More rollouts always help" is false for real utility.

The high-score bad-outlier case is the clearest anti-"more is always better" example:

```text
scores    = [0, 1, 2, 3]
utilities = [10, 9, 8, -100]
curve     = {1: -18.25, 2: -38.9375, 8: -89.1839}
```

The theorem predicts degradation because high `N` increasingly selects the high-score low-utility outlier.

## 5. Continuous/tie/AUC boundary issues

### Continuous no-tie boundary

The paper draft and docs currently say the continuous formula is population/no-tie intuition and that finite tied pools should use the finite law. That boundary is necessary and should not be softened.

Risky misuse sentence pattern:

```text
V_N = N E[R F_S(S)^(N-1)] gives the value of best-of-N selection.
```

Safer version:

```text
Under a population distribution with continuous score marginal and zero tie probability, V_N = N E[R F_S(S)^(N-1)]. Finite empirical pools, especially tied pools, use the tie-aware rank-interval law.
```

### AUC is exact only for N=2

The `N=2` tie-aware AUC identity appears correct:

```text
f_2 = p^2 + 2p(1-p)kappa
```

But AUC alone cannot control `N > 2`. An explicit finite tied pair:

Case A:

```text
scores  = [0, 0, 0, 0, 0]
success = [0, 0, 0, 0, 1]
p       = 0.2
kappa   = 0.5
N=2     = 0.2
N=3     = 0.2
```

Case B:

```text
scores  = [0, 0, 2, 2, 1]
success = [0, 0, 0, 0, 1]
p       = 0.2
kappa   = 0.5
N=2     = 0.2
N=3     = 0.152
```

Same `p`, same tie-aware AUC, same `N=2`, different `N=3`. This directly falsifies any claim that AUC alone predicts larger-`N` best-of-N success.

## 6. Adaptive allocation claim risks

The adaptive allocation story is not a theorem-level guarantee unless the exact curves are known and the objective is exactly the greedy marginal allocation objective over those curves.

Risks to keep bounded:

- Pilot-estimated curves can be stale, noisy, or shifted.
- Greedy marginal allocation can be a diagnostic/baseline, not a globally optimal controller.
- Fixed-budget allocation across empirical states is not a universal train/test compute scaling law.
- Closed-loop receding-horizon use changes the visited-state distribution; the finite theorem only applies conditionally at each visited state/pool.

Current paper wording is mostly careful: it says adaptive allocation is not a globally optimal controller claim and is supported in tested settings. Preserve that wording.

Recommended hostile wording constraint:

```text
Do not say "optimal adaptive rollout allocation" unless the sentence explicitly means optimal for known finite inference-value curves under the stated discrete allocation objective. Otherwise say "moment-law/adaptive allocation baseline" or "diagnostic allocation rule".
```

## 7. Model-error amplification claim risks

The model-error amplification claim is valid as a theorem interpretation only in the weak form:

```text
High-N selection amplifies whatever occupies the high-score tail; if that tail is model-error-heavy or real-utility-poor, selected real utility can saturate or worsen.
```

It should not be stated as:

- a universal claim that model error always gets amplified
- a safety certificate
- a proof about arbitrary model-based controllers
- a real-world robot failure guarantee

The theorem itself does not know about model error. It only knows the joint distribution of score and real utility. "Model-error amplification" is therefore an interpretation of score/real-utility misalignment in the high-score tail, plus empirical evidence from toy/simulator artifacts.

## 8. Sentences/claims to weaken

The most suspicious current wording is not mathematically false, but it can invite reviewer attack.

### Title/subtitle risk

Current title/subtitle pattern:

```text
How Much Should a Robot Imagine?
Exact Test-Time Scaling Laws for World-Action Planning
```

Risk: "scaling laws" may sound like a broad empirical/training scaling law, and "robot" may sound like hardware evidence.

Safer variants:

```text
How Much Should a Planner Sample?
Exact Test-Time Best-of-N Laws for World-Action Rollout Selection
```

or

```text
Exact Test-Time Inference Laws for Best-of-N World-Action Rollout Selection
```

### "More imagination helps only when..."

Current abstract-style sentence:

```text
The law reveals that more imagination helps only when the score ranks high-real-utility rollouts into the upper score tail.
```

Risk: "only" can be challenged because expected utility can stay equal under degenerate cases, and "helps" needs a fixed selection-rule/distribution context.

Safer version:

```text
For this fixed rollout/scorer distribution, gains from larger N are governed by real utility in the upper score tail; misaligned tails can make additional samples saturate or reduce real utility.
```

### "Experiments validate..."

Current pattern:

```text
Experiments validate the laws ... in analytic, learned WAM-lite, and simulator/benchmark rollout-pool settings.
```

Risk: "validate" can sound broader than exact-law checks and finite artifacts.

Safer version:

```text
Experiments check the laws and illustrate the predicted regimes in analytic, learned WAM-lite, and simulator/benchmark rollout-pool artifacts.
```

### "Robot executes..."

Current intro style:

```text
the robot executes the top-scoring rollout
```

Risk: hardware implication.

Safer version:

```text
the simulated agent or planner executes the selected action/rollout in the evaluated setting
```

If "robot" remains in motivation, keep explicit limitations nearby: no real-world robot deployment is reported.

### Benchmark/RoboCasa/LIBERO claims

Do not upgrade current benchmark artifacts into any of these claims:

- real-world robot evidence
- modern VLA-scale validation
- full RoboCasa-wide validation
- solved LIBERO policy performance
- universal OOD robustness

The README and draft currently contain many explicit disclaimers here. Keep them.

### Adaptive allocation wording

Avoid:

```text
optimal allocation
```

unless heavily qualified.

Prefer:

```text
fixed-budget diagnostic allocation using known or estimated inference-value curves
```

### Model-error wording

Avoid:

```text
proves model-error amplification in robots
```

Prefer:

```text
provides an exact score-tail accounting that explains how high-N selection can amplify model/scorer mismatch in tested rollout-pool settings
```

## 9. Final recommendation before ICLR submission

Recommendation: proceed only after preserving the narrow theorem boundary and weakening broad framing.

Submission checklist from hostile audit:

- Keep the finite theorem stated with all assumptions every time it is used: finite empirical pool, uniform with-replacement sampling, i.i.d. draws, max-score selection, uniform maximum-score tie-breaking.
- Keep the continuous formula visibly labeled as no-tie population intuition, not the empirical source of truth.
- Keep AUC identity visibly restricted to `N=2`; explicitly state AUC alone is insufficient for `N>2`.
- Avoid "scaling law" wording unless it is immediately narrowed to exact inference-time Best-of-N rollout-selection laws.
- Avoid any unqualified real-world robot, universal WAM training, modern VLA, full RoboCasa-wide, or OOD robustness claim.
- Rephrase adaptive allocation as a tested diagnostic/baseline, not global optimal control.
- Rephrase model-error amplification as a possible high-score-tail effect, not a universal theorem about model error.

Bottom line: I could not falsify the finite theorem. I could falsify several tempting stronger claims. The paper is defensible if it stays narrow; it becomes vulnerable if the title, abstract, or discussion lets readers infer a universal WAM scaling law or real-robot evidence.
