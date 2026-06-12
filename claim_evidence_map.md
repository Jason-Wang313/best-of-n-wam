# Claim Evidence Map

Every major paper claim should map to theorem text, an experiment, a figure, or a clearly marked gap. Wording using "exact", "prove", "optimal", "robot", "real-world", or "always" must be checked against this table before submission.

| Claim | Evidence type | Theorem/experiment/figure/citation | Current status | Risk/softening needed |
|---|---|---|---|---|
| For a fixed rollout generator and scorer, the score-tail audit curve is determined by the joint score/real-utility distribution. | Theorem + implementation | Finite score-tail rollout identity; `docs/theory.md`; `src/wam_inference_value/theorem.py`; Claims 1-2 | Supported | Say fixed distribution; do not imply training law. |
| The finite empirical law is exact under sampling with replacement and uniform maximum-score tie-breaking. | Theorem + tests + proof draft | Claims 1-2; theorem unit tests; result consistency checks; `iclr_appendix.tex` | Supported in repo; appendix proof drafted | Use "exact under stated assumptions"; keep proof audited before upload. |
| Binary success is a special case of utility-valued selection. | Theorem + tests | `binary_best_of_n_finite` delegates to utility law; Claims 1-2 | Supported | Keep R=Y special-case wording. |
| Continuous utility law is a no-tie population identity. | Theorem | `docs/theory.md`; theorem map | Needs manuscript proof audit | Do not use for finite tied pools. |
| For N=2, tie-aware AUC kappa determines binary best-of-2 success with p. | Theorem + tests | Claim 3; `auc_kappa`; `n2_auc_identity` | Supported | Specify N=2 only. |
| For N > 2, AUC alone is insufficient. | Experiment + theorem interpretation | Claim 4; same-p/kappa counterexample gap | Supported | Avoid saying AUC is useless; say insufficient alone. |
| More imagination helps only when high-score rollouts have high real utility. | Theorem interpretation + experiments | Claims 7-12, 42-47 | Supported conceptually | Avoid "only" without linking to fixed distribution; soften to "for this selection rule, gains require upper-tail alignment." |
| More rollouts can saturate or harm under bad scoring or mismatch. | Experiments | Claims 10-13, 43; falsification artifacts | Supported | Use "can" not "will"; specify tested settings. |
| Adaptive rollout budgeting can improve fixed-budget allocation in tested settings. | Experiments | Claims 14-15, 44; adaptive allocation artifacts | Supported in artifacts | Do not claim a global controller guarantee. |
| Stop rules and audit gates can diagnose risky high-N deployment. | Experiments/audits | Claims 41-47 | Supported as audit artifacts | Frame as diagnostic, not certified safety. |
| Learned WAM-lite artifacts check and illustrate the inference-value mechanism beyond analytic toys. | Experiments | Claims 22-25, 26-30, benchmark claims | Supported in tested artifacts | Emphasize lightweight models and simulator scope. |
| Benchmark rollout-pool experiments check the law in simulated robotics settings. | Experiments | Claims 48-80, 83-98 | Supported as simulator/benchmark evidence | Say "simulated robotics rollout-pool"; no hardware claim. |
| RoboCasa 12-task family learned WAM-lite scorer beats random with positive CI. | Experiment | Claim 85; prior rerun also reproduced a smaller verified spot | Supported | State rollout-pool dense-utility evidence, not policy success. |
| LIBERO evidence supports simulator smokes and dense rollout-pool WAM-lite checks. | Experiments | Claims 83, 86-89 | Supported | Do not claim modern VLA policy performance. |
| Closed-loop receding-horizon behavior is conditionally explained by applying the law at visited states. | Theorem interpretation + experiments | Claims 16-18, 36, 53, 62, 77 | Partially supported | Do not claim global closed-form closed-loop law. |
| The paper is about test-time inference value, not WAM training scaling. | Scope/limitation | `reports/paper_result_summary.md`; Claims 126-127 | Supported as boundary | Keep in abstract/introduction limitations. |
| The paper contains real-world robot evidence. | Evidence gap | None | Unsupported | Avoid. Mark as future work/limitation. |
| The method solves OOD generalization. | Evidence gap | None | Unsupported | Avoid. Mention pilot estimates can fail under shift. |
| The method gives an arbitrary-controller guarantee. | Evidence gap | None | Unsupported | Avoid; use "exact value for this selection rule" or "greedy allocation baseline." |
| Full RoboCasa-wide validation is complete. | Evidence gap | Registry audit: Claim 91 covers subset | Unsupported | Say broad but not full RoboCasa-wide. |
| ManiSkill RGB/RGB-D or EE-control validation is complete. | Blocker evidence | Claim 68 and blocker reports | Unsupported as validation | Present as blocker-documented limitation. |

## High-Risk Words Checklist

- "Exact": allowed for finite identity under stated assumptions, N=2 identity, and exact-law empirical checks.
- "Prove": allowed only in theorem/proof sections after proof text is finalized.
- "Optimal": avoid unless referring to oracle diagnostics or exact selection-rule value, not a general controller.
- "Robot": avoid in the title/body unless discussing related work or explicitly caveating simulator-only evidence.
- "Real-world": use only in limitations or future work.
- "Always": avoid for performance claims.
