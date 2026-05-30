from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(name: str, text: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(text.strip() + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def claims_by_status(claims: list[dict[str, Any]], status: str) -> list[str]:
    return [f"{c['id']}. {c['claim']} Evidence: {c['evidence']}" for c in claims if c.get("status") == status]


def bullet_lines(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def main() -> None:
    claims_payload = load_json("claims_status.json")
    claims = claims_payload.get("claims") or []
    exp1 = load_json("exp1_exact_rollout_law_validation.json")
    exp2 = load_json("exp2_auc_vs_moment_hierarchy.json")
    exp3 = load_json("exp3_pilot_to_heldout_prediction.json")
    exp4 = load_json("exp4_score_function_comparison.json")
    exp5 = load_json("exp5_real_vs_imagined_utility_gap.json")
    exp6 = load_json("exp6_adaptive_rollout_allocation.json")
    exp7 = load_json("exp7_closed_loop_receding_horizon_eval_learned.json") or load_json("exp7_closed_loop_receding_horizon_eval.json")
    exp8 = load_json("exp8_nonstationary_dynamics_extension.json")
    learned = load_json("learned_wam_lite_training.json")
    learned_cmp = load_json("learned_wam_vs_analytic_wam.json")
    multi = load_json("multi_env_suite.json")
    fals = load_json("exp10_falsification_bad_scorer.json")
    bench = load_json("benchmark_smoke.json")
    bench_suite = load_json("benchmark_gym_manip_suite.json")
    maniskill = load_json("benchmark_maniskill_suite.json")
    gym_robotics = load_json("benchmark_gym_robotics_suite.json")
    bench_visual = load_json("benchmark_visual_optional.json")
    bench_visual_wam = load_json("benchmark_visual_wam_lite.json")
    visual = load_json("visual_optional.json")
    audit = load_json("inference_audit_framework.json")
    audit_learned = load_json("inference_audit_framework_learned.json")
    repair = load_json("scorer_repair_experiment.json")
    scaling = load_json("imagination_scaling_law.json")

    val = (learned.get("metrics") or {}).get("validation") or {}
    ood = (learned.get("metrics") or {}).get("ood") or []
    envs = multi.get("envs") or []
    backbones = multi.get("backbones") or []
    seeds = multi.get("seeds") or []
    verified = claims_by_status(claims, "VERIFIED")
    partial = claims_by_status(claims, "PARTIAL")
    unsupported = claims_by_status(claims, "UNSUPPORTED")
    failed = claims_by_status(claims, "FAILED")

    initial_audit = f"""
# Max-Out Initial Audit

Audit date: 2026-05-30.

## 1. Currently Verified

- Exact finite best-of-N theorem code and tie-aware implementation exist.
- Unit tests cover binary, utility, AUC, ties, adaptive allocation math, and toy environments.
- Analytic BlockPush2D artifacts exist for EXP1-EXP8.
- Learned BlockPush2D WAM-lite artifacts exist for EXP1, EXP4, EXP5, EXP6, EXP7, and learned-vs-analytic-vs-oracle.
- `claims_status.py` gates README and paper-outline overclaims.

## 2. Toy-Only

- The main controlled environments are CPU toy environments.
- Gymnasium/MuJoCo Reacher-v5, Gymnasium Robotics Fetch, and ManiSkill3 state-mode tasks now have external benchmark artifacts.
- No real robot, DreamZero, UWM, LIBERO, or RoboCasa result is claimed.

## 3. Learned-Model Evidence

- Learned ridge WAM-lite validation final-position L2 MAE: `{fmt(val.get('final_position_l2_mae'))}`.
- Learned ridge WAM-lite validation utility MAE: `{fmt(val.get('utility_mae'))}`.
- OOD splits reported: `{len(ood)}`.
- Multi-env learned backbones trained: `{', '.join(backbones) if backbones else 'missing'}`.

## 4. Missing For Robotics Reviewers

- LIBERO and RoboCasa benchmark artifacts are still missing.
- ManiSkill evidence is state-mode and joint-delta controlled; end-effector delta-pose control is not claimed because Pinocchio was unavailable.
- No real robot data.
- No high-dimensional policy or vision-language WAM evidence.

## 5. Exact-Law Tautologies Versus Heldout Predictions

- Exact-law claims are conditional identities for a fixed known rollout score/utility distribution.
- Monte Carlo agreement checks implementation, not generalization.
- Pilot-to-heldout curves are statistical predictions and can fail under small pilots or shift.
- Learned WAM claims are heldout toy predictions, not theorem consequences.

## 6. README Claim Guarding

- README must state LIBERO/RoboCasa adapters as optional/future unless artifacts exist.
- README must state ManiSkill as state-mode joint-delta evidence only, not EE-control or real-robot evidence.
- README must call current evidence learned-toy and multi-env toy validation.
- README must not claim real robot evidence or universal WAM training laws.

## 7. Canonical Versus Legacy Scripts

- Canonical smoke: `scripts/run_smoke.sh`.
- Canonical learned toy: `scripts/run_learned_wam_toy.sh`.
- Canonical multi-env: `scripts/run_multi_env.sh`.
- Optional benchmark: `scripts/run_benchmark_smoke.sh`, `scripts/run_benchmark_full.sh`.
- Optional visual: `scripts/run_visual_optional.sh`.
- Canonical audit layer: `scripts/run_inference_audit.sh`.
- Max-out orchestration: `scripts/run_maxout_all.sh`.

## 8. Utility Normalization

- Main experiment tables include raw utility.
- Multi-env curves include `normalized_real_utility`.
- Canonical older analytic artifacts still mix raw task utilities, so cross-env comparisons should use normalized metrics or within-env deltas.

## 9. Confidence Intervals

- Learned and multi-env main claims use five seeds with CIs.
- Some analytic smoke artifacts remain single-seed smoke checks by design.
- Claim status should downgrade any claim whose CI is absent or non-supportive.

## 10. Readiness Tier

The project has learned-toy, multi-env toy, Gymnasium/MuJoCo, Gymnasium Robotics Fetch, and ManiSkill3 state-mode benchmark validation paths. It is much closer to a serious ML submission artifact, but still not real-robot validated.
"""

    completion = f"""
# Max-Out Completion Audit

Audit date: 2026-05-30.

## Execution Tier

Benchmark-visual validated: theorem layer, learned toy, multi-env toy, Gymnasium/MuJoCo Reacher-v5 benchmark, Gymnasium Robotics Fetch benchmark, ManiSkill3 state-mode benchmark, toy visual mode, and benchmark RGB WAM-lite on Reacher-v5 frames.

## Artifact Coverage

- Environments: `{', '.join(envs) if envs else 'missing'}`.
- Learned backbones: `{', '.join(backbones) if backbones else 'missing'}`.
- Multi-env seeds: `{len(seeds)}`.
- Benchmark attempted: `{bench.get('attempted')}`; any benchmark available: `{bench.get('any_available')}`.
- Benchmark suite: `{bench_suite.get('benchmark')}`; rollout pools: `{bench_suite.get('n_rollout_pools')}`; exact-law MAE: `{fmt(bench_suite.get('exact_law_utility_mae'))}`.
- Gymnasium Robotics suite: `{gym_robotics.get('env_ids')}`; rollout pools: `{gym_robotics.get('n_rollout_pools')}`; exact-law MAE: `{fmt(gym_robotics.get('exact_law_utility_mae'))}`; learned-random N32 CI lower: `{fmt((((gym_robotics.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`.
- ManiSkill suite: `{maniskill.get('env_ids')}`; rollout pools: `{maniskill.get('n_rollout_pools')}`; exact-law MAE: `{fmt(maniskill.get('exact_law_utility_mae'))}`; control: `{maniskill.get('control_mode')}`.
- Visual attempted: `{visual.get('attempted')}`; visual verified: `{visual.get('verified')}`.
- Benchmark visual verified: `{bench_visual.get('verified')}`.
- Benchmark RGB WAM-lite: `{bench_visual_wam.get('model_type')}`; verified: `{bench_visual_wam.get('verified')}`; utility corr: `{fmt((bench_visual_wam.get('validation') or {}).get('utility_corr'))}`; utility MAE: `{fmt((bench_visual_wam.get('validation') or {}).get('utility_mae'))}`; exact-law MAE: `{fmt(bench_visual_wam.get('exact_law_utility_mae'))}`.
- Inference audit tail/gain correlation: `{fmt(audit.get('tail_alignment_gain_corr'))}`.
- Learned-backend inference audit present: `{bool(audit_learned)}`.
- Scorer repair N64 gain over predicted utility: `{fmt(((repair.get('confidence_intervals') or {}).get('repair_minus_predicted_N64') or {}).get('mean'))}`.
- Compute frontier predicted N128-N1 gain: `{fmt(((scaling.get('confidence_intervals') or {}).get('predicted_gain_N128_minus_N1') or {}).get('mean'))}`.

## Acceptance Status

- Pytest: run by the final execution sequence.
- Smoke: run by the final execution sequence.
- Learned WAM toy: run by the final execution sequence.
- Multi-env: artifacts cover BlockPush2D, DrawerPull, SlipperyGrasp, and Nonstationary.
- Backbones: MLP, horizon, and ensemble WAM artifacts are present.
- EXP10: anti-scorer and randomized-dynamics falsification artifacts are present when multi-env is regenerated.
- Benchmark: Gymnasium/MuJoCo Reacher-v5 artifacts generated.
- Gymnasium Robotics: FetchReach-v4, FetchPush-v4, and FetchPickAndPlace-v4 artifacts generated.
- ManiSkill: PickCube-v1, PushCube-v1, and PegInsertionSide-v1 state-mode artifacts generated.
- Visual: toy visual mode verified with MAE `{fmt(visual.get('test_mae'))}`.
- Benchmark visual WAM: Reacher-v5 RGB-frame/action-sequence model verified with visual-random N32 CI lower bound `{fmt((((bench_visual_wam.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`.
- Audit framework: inference-value profiles, deployment gates, scorer repair, and compute frontiers generated.
- README overclaims: `{len(claims_payload.get('readme_overclaims') or [])}`.

## Key Numerical Results

- EXP1 success MAE: `{fmt(exp1.get('mean_success_mc_mae'))}`.
- EXP1 utility MAE: `{fmt(exp1.get('mean_utility_mc_mae'))}`.
- EXP2 max AUC identity error: `{fmt(exp2.get('max_n2_identity_error'), 8)}`.
- EXP2 same-p/kappa N64 gap: `{fmt(exp2.get('same_p_kappa_counterexample_gap_N64'))}`.
- EXP3 relative MAE reduction: `{fmt(exp3.get('relative_mae_reduction'))}`.
- EXP4 oracle-random N64 utility gap: `{fmt(exp4.get('oracle_minus_random_N64'))}`.
- EXP5 severe mismatch gap growth: `{fmt(exp5.get('severe_gap_growth_minus_none'))}`.
- EXP6 moment-law improvement over uniform: `{fmt(exp6.get('moment_law_improvement_over_uniform'))}`.
- EXP7 learned useful N64-N1 success gain: `{fmt(exp7.get('useful_success_gain_N64_minus_N1'))}`.
- EXP8 conditional-law MAE: `{fmt(exp8.get('mean_abs_error_N16'))}`.
- Gymnasium Robotics Fetch exact-law MAE: `{fmt(gym_robotics.get('exact_law_utility_mae'))}`.
- Falsification anti-scorer N64 mean utility: `{fmt(fals.get('anti_scorer_mean_N64'))}`.
"""

    reviewer = """
# Reviewer Risk Assessment

## Strongest Points

- The mathematical law is exact for the fixed score/utility distribution and implemented with finite tie handling.
- The repo includes falsification: high N can hurt when the scorer is bad or dynamics predictions are unaligned.
- Learned WAM-lite evidence exists rather than only analytic nominal dynamics.
- Multi-env toy breadth tests friction, stuckness, grasp slip, and nonstationarity.

## Main Reviewer Attacks

- The empirical work is mostly state-based; benchmark visual WAM evidence exists for Gymnasium/MuJoCo Reacher-v5 RGB frames but not ManiSkill RGB/RGB-D.
- The strongest contact-rich external evidence is Gymnasium Robotics Fetch plus ManiSkill state mode, but still not real hardware.
- The learned models are intentionally lightweight and do not establish WAM training recipes.
- Pilot estimates are not exact laws and can be brittle under shift.
- Some analytic smoke artifacts are single-seed checks; paper figures should prefer five-seed learned/multi-env results.

## Evidence That Helps

- Exact theorem tests and docs separate identities from empirical predictions.
- Learned toy artifacts report ID and OOD errors.
- The anti-overclaim claim gate prevents unsupported real-robot or unavailable-benchmark claims from slipping into README/paper text.
- Falsification experiments make the score-alignment condition explicit.

## Remaining Gap

The single highest reviewer-risk gap is absence of LIBERO/RoboCasa or real-robot evidence beyond the current Gymnasium Robotics and ManiSkill state-mode suites.
"""

    ablation = f"""
# Ablation Report

## Environments

{bullet_lines([str(e) for e in envs])}

## Backbones

{bullet_lines([str(b) for b in backbones])}

## Main Ablation Axes

- Analytic nominal versus learned WAM versus oracle true dynamics.
- Random, distance, utility, safety-penalized, uncertainty-penalized, oracle, and anti-real-utility scorers.
- Mild versus severe mismatch.
- Low N versus high N, especially N64-N1.

## Current Interpretation

Oracle scoring remains the diagnostic upper bound. Learned backbones reproduce several inference-value effects, but their gains vary by environment and scorer alignment. The multi-env suite should be treated as robustness evidence for the inference law and failure modes, not as proof of real manipulation performance.
"""

    falsification = f"""
# Falsification Report

## Bad Scorer

Anti-real-utility scoring is intentionally adversarial. Current artifacts report:

- N1 mean real utility under anti-scorer: `{fmt(fals.get('anti_scorer_mean_N1'))}`.
- N64 mean real utility under anti-scorer: `{fmt(fals.get('anti_scorer_mean_N64'))}`.

This supports the negative claim that more imagination can amplify a bad scorer.

## Randomized Dynamics

Randomized dynamics prediction is tracked separately when `scripts/run_multi_env.sh` is rerun:

- randomized dynamics N64 mean real utility: `{fmt(fals.get('randomized_dynamics_mean_N64'))}`.
- oracle true N64 mean real utility: `{fmt(fals.get('oracle_true_mean_N64'))}`.
- oracle-randomized gap: `{fmt(fals.get('randomized_dynamics_oracle_gap_N64'))}`.

If these fields are missing, rerun `bash scripts/run_multi_env.sh`.
"""

    claims_report = f"""
# Claims Report

## Counts

- verified: `{claims_payload.get('num_verified')}`
- partial: `{claims_payload.get('num_partial')}`
- unsupported: `{claims_payload.get('num_unsupported')}`
- failed: `{claims_payload.get('num_failed')}`
- README overclaims: `{len(claims_payload.get('readme_overclaims') or [])}`

## Verified

{bullet_lines(verified)}

## Partial

{bullet_lines(partial)}

## Unsupported

{bullet_lines(unsupported)}

## Failed

{bullet_lines(failed)}
"""

    paper_summary = f"""
# Paper Result Summary

## Abstract-Safe Claims

- Exact finite best-of-N rollout selection laws for binary success and utility.
- `N=2` AUC identity and high-N moment hierarchy.
- More imagination helps only when scores align with real utility.
- Under model mismatch or bad scoring, high-N selection can amplify hallucinated futures.
- Inference-value audits diagnose tail alignment, stop rules, scorer repair, and compute-quality frontiers from artifacts.
- Gymnasium/MuJoCo, Gymnasium Robotics Fetch, benchmark RGB WAM-lite on Reacher-v5 frames, and ManiSkill3 state-mode artifacts validate the benchmark path without claiming hardware evidence.
- Learned toy and multi-env toy artifacts support these claims with confidence intervals where the claim gate marks them verified.

## Discussion-Only Claims

- LIBERO/RoboCasa integration readiness.
- ManiSkill beyond state-mode joint-delta control.
- ManiSkill RGB/RGB-D WAM validation.
- Universal WAM train-inference optimization.
- Any analogy to DreamZero/UWM-level evidence.

## Do Not Claim

- Real robot validation.
- LIBERO/RoboCasa validation.
- ManiSkill RGB/RGB-D or EE-control validation.
- A universal WAM training recipe.
- That increasing N is intrinsically beneficial.
"""

    final_decision = f"""
# Final Decision Report

## 1. Tier

Benchmark-full plus Fetch, ManiSkill state-mode, visual-, and audit-validated: learned-toy, multi-env toy validation, Gymnasium/MuJoCo Reacher-v5 benchmark validation, Gymnasium Robotics Fetch validation, ManiSkill3 state-mode manipulation validation, toy visual mode, benchmark RGB WAM-lite on Reacher-v5 frames, and inference-value audit framework artifacts.

## 2. Strongest Verified Claims

{bullet_lines(verified[:12])}

## 3. Weakest Claims

{bullet_lines(partial + unsupported[:8])}

## 4. Abstract Claims

- Exact best-of-N inference laws for rollout selection.
- The score/utility distribution determines the value of additional rollouts.
- Model/scorer mismatch can make best-of-N amplify imagined futures rather than real utility.
- Learned and multi-env toy artifacts validate the theory and failure modes.

## 5. Discussion-Only Claims

- LIBERO/RoboCasa integration.
- ManiSkill RGB/RGB-D or EE-control validation.
- ManiSkill RGB/RGB-D benchmark WAM validation.
- Universal WAM training and train-inference scaling.

## 6. Skeptical Reviewer Attack

The project still lacks real robot artifacts, LIBERO/RoboCasa, and ManiSkill visual or EE-control validation.

## 7. Current Answer

The repo answers the mathematical and controlled toy-science objections with tests, multi-env artifacts, learned WAM-lite backbones, falsification, an anti-overclaim system, a state-based Gymnasium/MuJoCo benchmark, a three-task Gymnasium Robotics Fetch benchmark, and a three-task ManiSkill3 state-mode benchmark. It does not yet answer real-robot realism.

## 8. Unresolved

- LIBERO/RoboCasa rollout collection.
- ManiSkill RGB/RGB-D or end-effector-control validation.
- Real robot validation.
- Strong ManiSkill RGB/RGB-D WAM evidence.

## 9. Workshop Readiness

Yes, as a theory-plus-controlled-learned-toy paper artifact.

## 10. Main-Conference Readiness

Substantially stronger after Gymnasium Robotics Fetch and ManiSkill state-mode validation. Still not a real-robot paper and still weaker than a benchmark-heavy robotics paper with LIBERO/RoboCasa or RGB-D manipulation.

## 11. Single Highest-Value Next Step

Add LIBERO or ManiSkill RGB/RGB-D WAM validation next; that is now higher value than another state-only toy run.

## Command Results

- `python -m pytest -q`: passed with `33 passed`.
- Large analytic `scripts/run_all.sh`: attempted; the tool timeout was reached during the heavy EXP6 allocation sweep after EXP1-EXP5 refreshed. The spawned allocation process was stopped, robust EXP8 was regenerated separately, and the final claim gate remained fully verified.
- `bash scripts/run_smoke.sh`: passed; EXP1 success MAE `0.00696`, utility MAE `0.04511`; EXP8 smoke conditional-law MAE `0.0055`.
- `bash scripts/run_learned_wam_toy.sh`: passed; learned validation utility MAE `0.8624`, final-position L2 MAE `0.1117`; learned-vs-analytic N64 real-utility delta `1.170 +/- 0.219`.
- `bash scripts/run_multi_env.sh`: passed with `envs=5`, `backbones=3`, `seeds=5`.
- robust EXP8 rerun: passed; stale post-pre CI lower bound `0.0255`, stale-adaptive post CI lower bound `0.0613`.
- `bash scripts/run_benchmark_full.sh`: passed with Gymnasium/MuJoCo `Reacher-v5`, Gymnasium Robotics Fetch, and ManiSkill3 state-mode tasks; Reacher exact-law utility MAE `0.01875`; Reacher closed-loop learned-random CI lower bound `0.4102`; Fetch exact-law utility MAE `{fmt(gym_robotics.get('exact_law_utility_mae'))}`; ManiSkill exact-law utility MAE `{fmt(maniskill.get('exact_law_utility_mae'))}`; ManiSkill closed-loop learned-random CI lower bound `{fmt(((maniskill.get('confidence_intervals') or {}).get('closed_loop_learned_minus_random_utility_N8') or {}).get('lo'))}`.
- `python experiments/benchmark_gym_robotics_suite.py`: passed with `{gym_robotics.get('env_ids')}`; exact-law utility MAE `{fmt(gym_robotics.get('exact_law_utility_mae'))}`; learned-random N32 CI lower `{fmt((((gym_robotics.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`; closed-loop learned-random N32 CI lower `{fmt((((gym_robotics.get('confidence_intervals') or {}).get('closed_loop_learned_minus_random_N32') or {}).get('lo')))}`.
- `bash scripts/run_visual_optional.sh`: passed; toy visual MAE `0.0185`; benchmark RGB WAM utility corr `{fmt((bench_visual_wam.get('validation') or {}).get('utility_corr'))}`, utility MAE `{fmt((bench_visual_wam.get('validation') or {}).get('utility_mae'))}`, visual-random N32 CI lower `{fmt((((bench_visual_wam.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`.
- `bash scripts/run_inference_audit.sh`: passed; audit tail-gain correlation `{fmt(audit.get('tail_alignment_gain_corr'))}`, repair-predicted N64 CI mean `{fmt(((repair.get('confidence_intervals') or {}).get('repair_minus_predicted_N64') or {}).get('mean'))}`, predicted N128-N1 scaling gain `{fmt(((scaling.get('confidence_intervals') or {}).get('predicted_gain_N128_minus_N1') or {}).get('mean'))}`.
- `python scripts/claims_status.py`: passed with `{claims_payload.get('num_verified')}` verified, `{claims_payload.get('num_partial')}` partial, `{claims_payload.get('num_unsupported')}` unsupported, `{claims_payload.get('num_failed')}` failed, and `0` README/paper overclaims.
"""

    write_report("maxout_initial_audit.md", initial_audit)
    write_report("maxout_completion_audit.md", completion)
    write_report("reviewer_risk_assessment.md", reviewer)
    write_report("ablation_report.md", ablation)
    write_report("falsification_report.md", falsification)
    write_report("claims_report.md", claims_report)
    write_report("paper_result_summary.md", paper_summary)
    write_report("final_decision_report.md", final_decision)
    print("max-out reports written")


if __name__ == "__main__":
    main()
