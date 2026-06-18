# Final Audit

## 1. What Is The Discovered Main Thesis?

World-action models can become worse when a planner samples more candidate action sequences if the learned imagination score has optimistic pockets that are rare under execution. The paper is not a generic best-of-n wrapper: it is a score-tail audit for WAM rollout planning that measures when selection pressure turns model optimism into downstream utility loss.

## 2. What Is Genuinely New?

The v4 paper centers the domain-specific WAM mechanism: score-tail calibration, same-p versus off-policy tail gaps, anti-scorer negative controls, adaptive rescoring, and robotics coverage accounting. The contribution is a frozen audit protocol that separates real rollout utility from imagined score and asks whether larger candidate pools amplify or repair that mismatch.

## 3. What Theorem/Proof Survived Adversarial Checking?

The surviving theory is a bounded score-tail diagnostic: if model scores are unbiased under identity/noise controls, then the selected-tail gap isolates the utility loss caused by score optimism rather than by more sampling itself. The proof is used as a claim gate and diagnostic decomposition, not as a universal WAM performance guarantee.

## 4. What Is The Strongest Empirical Result?

The frozen v4 evidence verifies 127 manuscript claims with zero unsupported or partial claims. It contains 462 artifacts, 232 CSVs, 149 JSON files, 36 figures, and 45 NPZ files. The exact simulation reconstruction checks are tight: success MAE is `0.002104612037108505`, utility MAE is `0.013762680427723635`, AUC identity error is `0.0`, and the same-p kappa gap at n=64 is `0.9988209815422198`.

## 5. What Is The Strongest Real-Benchmark Upgrade?

The v4 benchmark ledger covers 8 benchmark rows and 454 held-out rollout pools across Fetch, RGB Fetch, MetaWorld, Robosuite, RoboCasa, ManiSkill, and LIBERO-facing cards. All 8 benchmark rows have positive lower confidence bounds, and the weakest benchmark CI lower bound remains positive at `0.09753950450146918`. RoboCasa coverage is explicitly accounted for: 97-task and 35-task ledgers, 396 registered tasks, 132 rollout task IDs, and 136 task IDs with any artifact.

## 6. What Are The Strongest Negative Controls And Stress Tests?

All 4 negative controls pass. The identity-score AUC error is zero, the anti-scorer gets worse as n grows (`-14.101561337022474` at n=1 and `-26.565706083129044` at n=64), and severe/stuck stress gaps grow by `16.725668702334993` and `16.38847839532815`. These failures are kept in the paper as adversarial teachers: they show exactly when score selection is unsafe and where the proposed audit must gate claims.

## 7. What Are The Biggest Weaknesses?

- The paper is an audit-and-selection-safety paper, not a new universal WAM training recipe.
- Some robotics evidence is frozen from cached benchmark artifacts rather than rerun end-to-end during final manuscript compilation.
- RoboCasa/LIBERO/ManiSkill evidence is used for scoped score-tail and coverage claims, not for leaderboard claims.
- The method diagnoses and gates harmful score-tail selection; it does not prove that every WAM planner can be repaired by the same rescoring rule.
- Claims must stay tied to frozen artifacts, exact seeds, benchmark rows, and pass/fail gates.

## 8. V4 Paper Status

This is a submission-ready v4 bounded mechanism paper. The final PDF is versioned as `paper/final/best-of-n-wam-v4.pdf` and mirrored to the visible Desktop as `C:\Users\wangz\OneDrive\Desktop\best-of-n-wam-v4.pdf`. The paper has been visually rendered and checked page by page at representative front, result, related-work, experiment, robotics, acceptance-criteria, and final-scope pages.

The v4 protocol avoids duplicate-template risk by using a WAM-specific title, mechanism, theory framing, benchmark ledger, robotics coverage accounting, and claim gates. The main story is: learned WAM score selection fails for a specific score-tail reason; the audit mechanism identifies and gates that reason; the evidence survives real benchmark cards, ablations, negative controls, stress tests, citation/related-work checks, reproducibility checks, visual PDF QA, and scoped claims.

## 9. V4 Acceptance Gates

- Final PDF is versioned and at least 25 pages.
- Desktop PDF and repository PDF have identical SHA-256 hashes.
- `results/v4_frozen_evidence/summary.json` reports 127 verified claims, zero partial claims, zero unsupported claims, 462 artifact files, 8 benchmark rows, 454 benchmark pools, 8 positive-CI rows, 4 passing negative controls, and 10 passing protocol gates.
- `scripts/claims_status.py` reports all 127 claims verified.
- `scripts/run_v4_claim_audit.py` passes.
- Final LaTeX log scan has no undefined citations, undefined references, rerun warnings, overfull boxes, natbib warnings, or hyperref warnings.
- Compile and test checks pass: `python -m compileall src tests experiments scripts -q` and `python -m pytest -q`.
- Old v2/v3 Desktop WAM PDFs are not treated as the final artifact.
- The source map points to the v4 Desktop PDF, this local source folder, and the GitHub repository.
- Final PDF SHA256: `B95203EDF4AC040FDD911ADE316C44B00B02D01106CEB68970CB0EC9F29D9FED`.
- Visual QA inspected rendered pages 1, 6, 8, 13, 16, 22, and 27.
