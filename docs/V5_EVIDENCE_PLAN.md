# V5 Evidence Plan

This plan defines the V5 upgrade for `best-of-n-wam`. It is intentionally
scoped as a top-venue mechanism, audit, and benchmark-artifact paper, not as a
GPU-scale world-model, real-robot, or robotics-leaderboard paper.

## Submission Identity

The V5 paper should make one central claim:

> Extra rollout imagination is valuable only when the scorer-selected tail
> contains real utility. That tail can be measured, stress-tested, and used to
> decide whether to allow, stop, block, or label before spending more test-time
> compute.

The paper must not claim:

- real-robot validation;
- GPU-scale training;
- a universal WAM training recipe;
- broad robotics state of the art;
- universal OOD generalization;
- that any audit can repair unobservable real-utility tails without labels.

## Low-RAM Execution Rules

V5 experiments must remain commodity-CPU and low-RAM without weakening evidence
quality.

- Run one experiment family at a time.
- Stream rows to CSV or keep only compact per-pool/per-seed summaries.
- Avoid storing full dense tensors of all tasks, budgets, selectors, and seeds
  when a group-by summary is enough.
- Prefer exact finite-pool calculations over Monte Carlo reruns when possible.
- Use deterministic seeds and write frozen JSON/CSV summaries for paper
  generation.
- Keep the one-command reproduction path below 2 GiB peak RAM.
- Keep full-quality runs sequential; do not reduce seeds or baselines silently to
  make a claim pass.
- Smoke runs may be smaller, but they must be labeled as smoke and cannot
  overwrite canonical evidence.

## V5 Evidence Modules

### 1. Exact Law Hardening

Purpose: make the theorem layer unambiguous and hard to misread.

Artifacts:

- `results/v5/exact_law_hardening.csv`
- `results/v5/exact_law_hardening.json`

Required checks:

- finite utility law handles ties, all-equal scores, negative utility,
  non-binary utility, and degenerate success rates;
- finite law is source of truth when scores tie;
- continuous/no-tie diagnostics are explicitly secondary.

Gate:

- all deterministic edge-case errors are at numerical tolerance;
- no generated text claims the continuous law is the finite tied-pool law.

### 2. AUC/Correlation Insufficiency

Purpose: prove common summary metrics are insufficient for high-N selection.

Artifacts:

- `results/v5/auc_correlation_insufficiency.csv`
- `results/v5/auc_correlation_insufficiency.json`

Required checks:

- construct pool pairs with matched success rate, AUC, mean score, mean utility,
  and score/utility correlation bins;
- show materially different selected utility at high N;
- include at least one pair where high N helps and one where high N harms.

Gate:

- matched-summary pairs differ by at least the predeclared high-N gap threshold;
- N=2 AUC identity remains correct while higher N diverges.

### 3. Exhaustive Finite-Pool Census

Purpose: turn CPU-only compute into exhaustive coverage.

Artifacts:

- `results/v5/finite_pool_census.csv`
- `results/v5/finite_pool_census_summary.json`

Design:

- enumerate small binary-utility finite pools by score-rank/tie pattern and
  success pattern;
- classify each pool as help, harm, saturation, nonmonotonic, or flat;
- count AUC-misleading and correlation-misleading cases;
- do not keep all raw pools in memory; stream rows and aggregate counters.

Gate:

- census covers all predeclared small-pool settings;
- regime counts sum to the exact number of enumerated configurations.

### 4. Blind Prospective Held-Out Audit

Purpose: prevent the post-hoc-story objection.

Artifacts:

- `results/v5/prospective_audit_predictions.csv`
- `results/v5/prospective_audit_predictions.sha256`
- `results/v5/prospective_audit_outcomes.csv`
- `results/v5/prospective_audit_summary.json`

Protocol:

1. Split task families and seeds into pilot/dev/held-out sets.
2. Generate predictions from pilot/dev data only.
3. Write prediction CSV and hash before loading held-out outcomes.
4. Evaluate held-out pools and closed-loop outcomes.
5. Report prediction accuracy, false allow rate, false block rate, and regret
   avoided.

Gate:

- predictions file hash exists before outcome file;
- held-out summary includes allow/stop/block/request-label decisions;
- false allow rate is reported even if it is nonzero.

### 5. Closed-Loop Execution Validation

Purpose: show selected-tail audits matter when acting, not only in static pools.

Artifacts:

- `results/v5/closed_loop_validation.csv`
- `results/v5/closed_loop_validation_summary.json`

Required selectors:

- `N=1`;
- raw high-N;
- random high-N;
- audit/adaptive selector;
- conservative block/stop policy;
- oracle upper bound.

Gate:

- reports return, success, harm rate, regret, rollouts used, and CPU time;
- paired CIs are computed across seeds/families;
- failures and blocked decisions are visible, not hidden.

### 6. Label-Budget Sample Complexity

Purpose: quantify how many real-utility labels the audit needs.

Artifacts:

- `results/v5/label_budget_sample_complexity.csv`
- `results/v5/label_budget_sample_complexity_summary.json`

Budgets:

- `0, 1, 2, 4, 8, 16, 32, 64` labels per family or state bucket.

Metrics:

- sign prediction accuracy;
- lower-bound coverage;
- regret avoided;
- false allow rate;
- false block rate.

Gate:

- zero-label behavior is included;
- request-label decisions are separated from repair/allow decisions.

### 7. Strong Selector Baseline Gauntlet

Purpose: avoid weak-baseline criticism.

Artifacts:

- `results/v5/selector_gauntlet.csv`
- `results/v5/selector_gauntlet_summary.json`

Required baselines:

- raw score;
- random;
- `N=1`;
- uncertainty penalty;
- score clipping;
- rank averaging;
- CVaR or lower quantile selector;
- lower-confidence bound;
- early stop by marginal gain;
- adaptive allocation;
- oracle upper bound.

Gate:

- all selectors use the same candidate pools and evaluation splits;
- oracle rows are labeled as nondeployable.

### 8. Equal-Compute CPU Frontier

Purpose: directly answer "why not just sample more?"

Artifacts:

- `results/v5/equal_compute_frontier.csv`
- `results/v5/equal_compute_frontier_summary.json`

Compare:

- blind more rollouts;
- fewer rollouts plus audit;
- pilot labels plus calibrated selector;
- adaptive allocation;
- conservative block/stop policy.

Gate:

- reports wall-clock time, rollouts, labels, peak RSS estimate, return, and
  regret;
- uses a fixed CPU budget grid and does not compare unequal budgets as if equal.

### 9. Impossibility Boundary

Purpose: make the method mature by proving when it must refuse.

Artifacts:

- `results/v5/impossibility_boundary.csv`
- `results/v5/impossibility_boundary_summary.json`

Required checks:

- construct paired pools with identical observable score features but different
  hidden real-utility tails;
- show score-only selectors must make the same decision;
- show the correct audit action is block or request labels.

Gate:

- no paper text says score-only evidence can solve these cases;
- request-label/block behavior is treated as a valid outcome, not a failure to
  hide.

### 10. ScoreTailBench Artifact

Purpose: make the paper useful beyond its own claims.

Package:

- `scoretailbench/manifest.json`
- `scoretailbench/pools/*.csv`
- `scoretailbench/splits/*.json`
- `scoretailbench/baselines/*.json`
- `scoretailbench/README.md`

Benchmark rows must include:

- `pool_id`;
- `family`;
- `split`;
- `candidate_id`;
- `score`;
- `imagined_utility`;
- `real_utility`;
- optional observable diagnostics;
- hidden fields only when marked nondeployable.

Gate:

- benchmark runner can reproduce baseline selected-utility curves from the
  package alone;
- hidden/test split separation is documented.

### 11. One-Command Low-RAM Reproduction

Purpose: turn CPU-only into a reproducibility strength.

Command:

```bash
python scripts/reproduce_v5_core_cpu.py
python scripts/reproduce_v5_cpu.py
```

Required output:

- exact-law check;
- AUC/correlation counterexample;
- mini prospective audit;
- mini closed-loop validation;
- mini equal-compute frontier;
- RAM/time report.

Gate:

- smoke artifacts are written under `results/v5_smoke/`;
- the command does not overwrite canonical V5 evidence;
- peak memory target is below 2 GiB.

## Paper Integration

V5 paper text should be organized around:

1. finite selected-tail law;
2. why AUC/correlation/mean error do not determine high-N value;
3. audit decisions: allow, stop, block, request labels;
4. blind prospective challenge;
5. closed-loop validation;
6. equal-compute frontier;
7. ScoreTailBench;
8. limitations and impossibility boundary.

Every promoted number must map to a CSV/JSON artifact and pass claim audit.

## Verification Checklist

- `python experiments/v5_frozen_evidence.py`
- `python scripts/run_v5_claim_audit.py`
- `python scripts/build_v5_paper.py`
- `python scripts/reproduce_v5_core_cpu.py`
- `python scripts/reproduce_v5_cpu.py`
- `pytest`
- final PDF build
- LaTeX warning scan
- source-map/hash/page-count audit

The V5 goal is complete only when all evidence modules above exist, paper text
uses them, claim audits pass, tests pass, and final PDF verification proves the
generated paper matches the scoped claims.
