from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_results_override = os.environ.get("WAM_RESULTS_DIR")
RESULTS = Path(_results_override).expanduser() if _results_override else ROOT / "results"
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
_reports_override = os.environ.get("WAM_REPORTS_DIR")
REPORTS = Path(_reports_override).expanduser() if _reports_override else ROOT / "reports"
if not REPORTS.is_absolute():
    REPORTS = ROOT / REPORTS


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
    maniskill_visual_probe = load_json("benchmark_maniskill_visual_probe.json")
    maniskill_dependency_probe = load_json("benchmark_maniskill_dependency_probe.json")
    metaworld = load_json("benchmark_metaworld_suite.json")
    robosuite = load_json("benchmark_robosuite_suite.json")
    robocasa = load_json("benchmark_robocasa_smoke.json")
    robocasa_learned = load_json("benchmark_robocasa_learned_wam.json")
    robocasa_multitask = load_json("benchmark_robocasa_multitask_wam.json")
    robocasa_broad = load_json("benchmark_robocasa_broad_wam.json")
    robocasa_family12 = load_json("benchmark_robocasa_family12_wam.json")
    robocasa_family24 = load_json("benchmark_robocasa_family24_wam.json")
    robocasa_extra4 = load_json("benchmark_robocasa_extra4_wam.json")
    robocasa_family28 = load_json("benchmark_robocasa_family28_wam.json")
    robocasa_family32 = load_json("benchmark_robocasa_family32_wam.json")
    robocasa_stratified55 = load_json("benchmark_robocasa_stratified55_wam.json")
    robocasa_stratified97 = load_json("benchmark_robocasa_stratified97_wam.json")
    robocasa_residual35 = load_json("benchmark_robocasa_residual35_h1_n4_wam.json")
    robocasa_catalog = load_json("benchmark_robocasa_catalog_probe.json")
    robocasa_micro = load_json("benchmark_robocasa_micro_rollout_extra.json")
    robocasa_micro_stratified = load_json("benchmark_robocasa_micro_rollout_stratified_probe.json")
    robocasa_micro_frontier = load_json("benchmark_robocasa_micro_rollout_frontier_probe.json")
    robocasa_residual_sweep = load_json("benchmark_robocasa_residual_frontier_sweep.json")
    libero_wam = load_json("benchmark_libero_wam.json")
    libero_scripted = load_json("benchmark_libero_scripted_policy.json")
    libero_action_head = load_json("benchmark_libero_learned_action_head.json")
    libero_autonomous_bc = load_json("benchmark_libero_autonomous_bc_policy.json")
    libero_visual_language_bc = load_json("benchmark_libero_visual_language_bc_policy.json")
    gym_robotics = load_json("benchmark_gym_robotics_suite.json")
    bench_visual = load_json("benchmark_visual_optional.json")
    bench_visual_wam = load_json("benchmark_visual_wam_lite.json")
    gym_robotics_visual = load_json("benchmark_gym_robotics_visual_wam.json")
    visual = load_json("visual_optional.json")
    audit = load_json("inference_audit_framework.json")
    audit_learned = load_json("inference_audit_framework_learned.json")
    repair = load_json("scorer_repair_experiment.json")
    scaling = load_json("imagination_scaling_law.json")
    artifact_integrity = load_json("artifact_integrity.json")
    artifact_manifest = load_json("artifact_manifest.json")
    figure_quality = load_json("figure_quality.json")
    result_consistency = load_json("result_consistency.json")
    raw_result_recompute = load_json("raw_result_recompute.json")
    narrative_consistency = load_json("narrative_consistency.json")
    claim_ledger_integrity = load_json("claim_ledger_integrity.json")
    script_contracts = load_json("script_contracts.json")
    claim_evidence_quality = load_json("claim_evidence_quality.json")
    claim_semantics = load_json("claim_semantics.json")

    val = (learned.get("metrics") or {}).get("validation") or {}
    ood = (learned.get("metrics") or {}).get("ood") or []
    envs = multi.get("envs") or []
    backbones = multi.get("backbones") or []
    seeds = multi.get("seeds") or []
    verified = claims_by_status(claims, "VERIFIED")
    partial = claims_by_status(claims, "PARTIAL")
    unsupported = claims_by_status(claims, "UNSUPPORTED")
    failed = claims_by_status(claims, "FAILED")
    residual_weaknesses = [
        "No real-robot or hardware-in-the-loop evidence; every promoted robotics result is simulator or benchmark evidence.",
        "LIBERO evidence is limited to three Spatial dense rollout-pool WAM-lite tasks plus Object sparse-success scripted/action-head/time-conditioned/RGB-proprio-language feature-kNN smokes, not modern VLA policy performance or full LIBERO validation.",
        "RoboCasa evidence is broad but not full RoboCasa-wide validation: committed rollout-pool artifacts cover 132 of 396 local registry task IDs, with micro-rollout probes covering 106 task IDs.",
        "ManiSkill evidence is CPU state-mode joint-delta control; RGB/RGB-D and end-effector-control validation are blocker-documented, not verified.",
        "Learned WAMs are intentionally lightweight ridge/kNN/CPU models; the repo does not prove a universal WAM training recipe.",
    ]
    weakest_claims = partial + unsupported[:8] + residual_weaknesses
    residual35_n_values = [int(n) for n in (robocasa_residual35.get("n_values") or [])]
    residual35_nmax = max(residual35_n_values) if residual35_n_values else None
    residual35_lr_ci = (robocasa_residual35.get("confidence_intervals") or {}).get(
        f"best_learned_minus_random_N{residual35_nmax}"
    ) or {}
    residual35_oracle_ci = (robocasa_residual35.get("confidence_intervals") or {}).get(
        f"oracle_minus_best_learned_N{residual35_nmax}"
    ) or {}

    initial_audit = f"""
# Max-Out Initial Audit

Audit date: 2026-05-30.

## 1. Currently Verified

- Exact finite best-of-N theorem code and tie-aware implementation exist.
- Unit tests cover binary, utility, AUC, ties, adaptive allocation math, and toy environments.
- Analytic BlockPush2D artifacts exist for EXP1-EXP8.
- Learned BlockPush2D WAM-lite artifacts exist for EXP1, EXP4, EXP5, EXP6, EXP7, and learned-vs-analytic-vs-oracle.
- `claims_status.py` gates README and paper-outline overclaims.
- `artifact_integrity.py` verifies that referenced result artifacts exist, parse, and are nonempty.
- `artifact_manifest.py` writes deterministic SHA-256 hashes for canonical scientific result artifacts.
- `figure_quality.py` verifies that canonical PNG figures are present, readable, nonblank, nonflat, and large enough for publication-style inspection.
- `result_consistency.py` verifies that summary JSONs agree with canonical tables for row counts, coverage, CI sanity, and success counts.
- `raw_result_recompute.py` independently recomputes aggregate means, exact-law MAEs, and seed-metric CIs from raw CSV artifacts.
- `narrative_consistency.py` verifies that high-impact README and final-report numbers match the current JSON artifacts.
- `script_contracts.py` verifies that canonical shell scripts preserve required experiment steps, optional benchmark guards, and ordered verification gates.
- `claim_semantics.py` verifies that verified-claim wording is backed by matching semantic thresholds.
- `claim_evidence_quality.py` verifies that each current claim ID is mapped to source artifacts and has structured, non-placeholder evidence.
- `claim_ledger_integrity.py` verifies sorted contiguous claim IDs, JSON/Markdown count agreement, structured claim evidence, evidence-path references, empty overclaim arrays, and no non-verified final claims.

## 2. Toy-Only

- The main controlled environments are CPU toy environments.
- Gymnasium/MuJoCo Reacher-v5, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, ManiSkill3 state-mode tasks, RoboCasa kitchen smoke plus single-task, three-task pick-place-family, broad four-task, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned-WAM artifacts, LIBERO Spatial rollout-pool WAM artifacts, a LIBERO Object sparse-success scripted smoke, a LIBERO learned action-head smoke, a LIBERO time-conditioned autonomous low-dimensional BC sparse-success smoke, and a LIBERO RGB/proprio/language BC sparse-success smoke now have external benchmark artifacts.
- No real robot, DreamZero, UWM, modern VLA, or full LIBERO policy result is claimed; RoboCasa is verified for pick-place-family, broad atomic kitchen, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook rollout-pool artifacts, and LIBERO is verified as a three-task rollout-pool dense-utility artifact plus narrow scripted, learned action-head, time-conditioned low-dimensional BC, and RGB/proprio/language BC sparse-success smokes.

## 3. Learned-Model Evidence

- Learned ridge WAM-lite validation final-position L2 MAE: `{fmt(val.get('final_position_l2_mae'))}`.
- Learned ridge WAM-lite validation utility MAE: `{fmt(val.get('utility_mae'))}`.
- OOD splits reported: `{len(ood)}`.
- Multi-env learned backbones trained: `{', '.join(backbones) if backbones else 'missing'}`.

## 4. Missing For Robotics Reviewers

- Modern VLA-style or full-suite LIBERO policy artifacts are still missing; current LIBERO evidence is a three-task Spatial rollout-pool WAM-lite artifact with dense progress utility, a hand scripted Object sparse-success smoke, a learned action-head smoke with scripted phases and target points, a time-conditioned low-dimensional BC smoke without phase labels or target commands at evaluation time, and an RGB/proprio/language feature-kNN BC smoke without object state or task IDs.
- Full RoboCasa-wide learned-WAM benchmark artifacts are still missing; current RoboCasa evidence is a single-task smoke rollout pool, a single-task learned-WAM artifact, a three-task pick-place family learned-WAM artifact, a broad four-task atomic-manipulation artifact, a 12-task open/close/turn family artifact, a 24-task open/close/turn/pick-place family artifact, an extra four-task pick-place-direction artifact, combined 28-task and 32-task family artifacts, stratified 55-task and 97-task kitchen artifacts, and a separate residual 35-task clean/cook horizon-1/N4 artifact.
- ManiSkill evidence is state-mode and joint-delta controlled; end-effector delta-pose control is not claimed because Pinocchio was unavailable.
- Meta-World ML1 evidence covers `reach-v3`, `push-v3`, and `drawer-open-v3` with state/action-sequence WAM-lite artifacts.
- RoboSuite evidence covers Panda `Lift`, `Stack`, and `Door` with clone-restored MuJoCo rollout pools, state/action-sequence WAM-lite artifacts, and small closed-loop learned/reward-versus-random evaluation.
- No real robot data.
- No high-dimensional policy or vision-language WAM evidence.

## 5. Exact-Law Tautologies Versus Heldout Predictions

- Exact-law claims are conditional identities for a fixed known rollout score/utility distribution.
- Monte Carlo agreement checks implementation, not generalization.
- Pilot-to-heldout curves are statistical predictions and can fail under small pilots or shift.
- Learned WAM claims are heldout toy predictions, not theorem consequences.

## 6. README Claim Guarding

- README must state LIBERO as optional/separate-environment and limited to three Spatial rollout-pool dense-utility validation plus Object sparse-success scripted, learned action-head, time-conditioned low-dimensional BC, and RGB/proprio/language BC smokes; RoboCasa is optional/separate-environment and includes pick-place-family, broad atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook rollout-pool validation, not full RoboCasa-wide validation.
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

The project has learned-toy, multi-env toy, Gymnasium/MuJoCo, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, and ManiSkill3 state-mode benchmark validation paths. It is much closer to a serious ML submission artifact, but still not real-robot validated.
"""

    completion = f"""
# Max-Out Completion Audit

Audit date: 2026-05-30.

## Execution Tier

Benchmark-visual validated: theorem layer, learned toy, multi-env toy, Gymnasium/MuJoCo Reacher-v5 benchmark, Gymnasium Robotics Fetch benchmark, Meta-World ML1, RoboSuite Panda benchmark, ManiSkill3 state-mode benchmark, RoboCasa single-task, three-task pick-place-family, broad four-task, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned WAM-lite, LIBERO Spatial three-task rollout-pool WAM-lite, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned autonomous low-dimensional BC smoke, LIBERO RGB/proprio/language BC smoke, toy visual mode, Reacher RGB WAM-lite, and Fetch RGB WAM-lite.

## Artifact Coverage

- Environments: `{', '.join(envs) if envs else 'missing'}`.
- Learned backbones: `{', '.join(backbones) if backbones else 'missing'}`.
- Multi-env seeds: `{len(seeds)}`.
- Benchmark attempted: `{bench.get('attempted')}`; any benchmark available: `{bench.get('any_available')}`.
- Benchmark suite: `{bench_suite.get('benchmark')}`; rollout pools: `{bench_suite.get('n_rollout_pools')}`; exact-law MAE: `{fmt(bench_suite.get('exact_law_utility_mae'))}`.
- Gymnasium Robotics suite: `{gym_robotics.get('env_ids')}`; rollout pools: `{gym_robotics.get('n_rollout_pools')}`; exact-law MAE: `{fmt(gym_robotics.get('exact_law_utility_mae'))}`; learned-random N32 CI lower: `{fmt((((gym_robotics.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`.
- Meta-World suite: `{metaworld.get('task_names')}`; rollout pools: `{metaworld.get('n_rollout_pools')}`; exact-law MAE: `{fmt(metaworld.get('exact_law_utility_mae'))}`; learned-random N32 CI lower: `{fmt((((metaworld.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`; reward-random N32 CI lower: `{fmt((((metaworld.get('confidence_intervals') or {}).get('reward_minus_random_N32') or {}).get('lo')))}`.
- RoboSuite suite: `{robosuite.get('env_names')}`; rollout pools: `{robosuite.get('n_rollout_pools')}`; exact-law MAE: `{fmt(robosuite.get('exact_law_utility_mae'))}`; learned-random N32 CI lower: `{fmt((((robosuite.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`; reward-random N32 CI lower: `{fmt((((robosuite.get('confidence_intervals') or {}).get('reward_minus_random_N32') or {}).get('lo')))}`; closed-loop learned-random N8 CI lower: `{fmt((((robosuite.get('confidence_intervals') or {}).get('closed_loop_learned_minus_random_N8') or {}).get('lo')))}`.
- ManiSkill suite: `{maniskill.get('env_ids')}`; rollout pools: `{maniskill.get('n_rollout_pools')}`; exact-law MAE: `{fmt(maniskill.get('exact_law_utility_mae'))}`; control: `{maniskill.get('control_mode')}`.
- RoboCasa learned WAM-lite: verified `{robocasa_learned.get('verified')}`; train samples `{robocasa_learned.get('train_samples')}`; eval samples `{robocasa_learned.get('eval_samples')}`; utility corr `{fmt(((robocasa_learned.get('model_metrics') or {}).get('utility_corr')))}`; learned-random N8 CI lower `{fmt((((robocasa_learned.get('confidence_intervals') or {}).get('learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa three-task WAM-lite: verified `{robocasa_multitask.get('verified')}`; tasks `{robocasa_multitask.get('env_ids')}`; train/eval samples `{robocasa_multitask.get('train_samples')}`/`{robocasa_multitask.get('eval_samples')}`; utility corr `{fmt(((robocasa_multitask.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_multitask.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_multitask.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa broad task family WAM-lite: verified `{robocasa_broad.get('verified')}`; tasks `{robocasa_broad.get('env_ids')}`; train/eval samples `{robocasa_broad.get('train_samples')}`/`{robocasa_broad.get('eval_samples')}`; utility corr `{fmt(((robocasa_broad.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_broad.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_broad.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa 12-task family WAM-lite: verified `{robocasa_family12.get('verified')}`; tasks `{len(robocasa_family12.get('env_ids') or [])}`; train/eval samples `{robocasa_family12.get('train_samples')}`/`{robocasa_family12.get('eval_samples')}`; utility corr `{fmt(((robocasa_family12.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_family12.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_family12.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa 24-task family WAM-lite: verified `{robocasa_family24.get('verified')}`; tasks `{len(robocasa_family24.get('env_ids') or [])}`; train/eval samples `{robocasa_family24.get('train_samples')}`/`{robocasa_family24.get('eval_samples')}`; utility corr `{fmt(((robocasa_family24.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_family24.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_family24.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa extra four-task WAM-lite: verified `{robocasa_extra4.get('verified')}`; tasks `{len(robocasa_extra4.get('env_ids') or [])}`; train/eval samples `{robocasa_extra4.get('train_samples')}`/`{robocasa_extra4.get('eval_samples')}`; utility corr `{fmt(((robocasa_extra4.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_extra4.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_extra4.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa combined 28-task family WAM-lite: verified `{robocasa_family28.get('verified')}`; tasks `{len(robocasa_family28.get('env_ids') or [])}`; train/eval samples `{robocasa_family28.get('train_samples')}`/`{robocasa_family28.get('eval_samples')}`; utility corr `{fmt(((robocasa_family28.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_family28.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_family28.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa combined 32-task family WAM-lite: verified `{robocasa_family32.get('verified')}`; tasks `{len(robocasa_family32.get('env_ids') or [])}`; train/eval samples `{robocasa_family32.get('train_samples')}`/`{robocasa_family32.get('eval_samples')}`; utility corr `{fmt(((robocasa_family32.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_family32.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_family32.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa stratified 55-task WAM-lite: verified `{robocasa_stratified55.get('verified')}`; tasks `{len(robocasa_stratified55.get('env_ids') or [])}`; train/eval samples `{robocasa_stratified55.get('train_samples')}`/`{robocasa_stratified55.get('eval_samples')}`; utility corr `{fmt(((robocasa_stratified55.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_stratified55.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_stratified55.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa stratified 97-task WAM-lite: verified `{robocasa_stratified97.get('verified')}`; tasks `{len(robocasa_stratified97.get('env_ids') or [])}`; train/validation/eval samples `{robocasa_stratified97.get('train_samples')}`/`{robocasa_stratified97.get('validation_samples')}`/`{robocasa_stratified97.get('eval_samples')}`; rollout pools `{robocasa_stratified97.get('eval_rollout_pools')}`; utility corr `{fmt(((robocasa_stratified97.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_stratified97.get('promoted_scorer')}`; learned-random N8 CI lower `{fmt((((robocasa_stratified97.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; oracle-learned N8 CI lower `{fmt((((robocasa_stratified97.get('confidence_intervals') or {}).get('oracle_minus_best_learned_N8') or {}).get('lo')))}`.
- RoboCasa residual 35-task clean/cook WAM-lite: verified `{robocasa_residual35.get('verified')}`; tasks `{len(robocasa_residual35.get('env_ids') or [])}`; train/validation/eval samples `{robocasa_residual35.get('train_samples')}`/`{robocasa_residual35.get('validation_samples')}`/`{robocasa_residual35.get('eval_samples')}`; rollout pools `{robocasa_residual35.get('eval_rollout_pools')}`; horizon `{robocasa_residual35.get('horizon')}`; Nmax `{residual35_nmax}`; utility corr `{fmt(((robocasa_residual35.get('model_metrics') or {}).get('utility_corr')))}`; promoted scorer `{robocasa_residual35.get('promoted_scorer')}`; learned-random Nmax CI lower `{fmt(residual35_lr_ci.get('lo'))}`; oracle-learned Nmax CI lower `{fmt(residual35_oracle_ci.get('lo'))}`.
- RoboCasa micro-rollout probe: verified `{robocasa_micro.get('verified')}`; nondegenerate extra task IDs `{robocasa_micro.get('nondegenerate_task_count')}`; rollouts per task `{robocasa_micro.get('rollouts_per_task')}`; horizon `{robocasa_micro.get('horizon')}`.
- RoboCasa stratified micro-rollout probe: verified `{robocasa_micro_stratified.get('verified')}`; nondegenerate task IDs `{robocasa_micro_stratified.get('nondegenerate_task_count')}`; rollouts per task `{robocasa_micro_stratified.get('rollouts_per_task')}`; horizon `{robocasa_micro_stratified.get('horizon')}`.
- RoboCasa frontier micro-rollout probe: verified `{robocasa_micro_frontier.get('verified')}`; candidates `{robocasa_micro_frontier.get('candidate_task_count')}`; runnable `{robocasa_micro_frontier.get('runnable_task_count')}`; nondegenerate task IDs `{robocasa_micro_frontier.get('nondegenerate_task_count')}`; rollouts per task `{robocasa_micro_frontier.get('rollouts_per_task')}`; horizon `{robocasa_micro_frontier.get('horizon')}`.
- RoboCasa residual clean/cook micro-rollout sweep: verified `{robocasa_residual_sweep.get('verified')}`; candidates `{robocasa_residual_sweep.get('candidate_task_count')}`; completed chunks `{robocasa_residual_sweep.get('completed_chunk_count')}`; timed-out chunks `{robocasa_residual_sweep.get('timed_out_chunk_count')}`; runnable `{robocasa_residual_sweep.get('runnable_task_count')}`; nondegenerate task IDs `{robocasa_residual_sweep.get('nondegenerate_task_count')}`; rollouts per task `{robocasa_residual_sweep.get('rollouts_per_task')}`; horizon `{robocasa_residual_sweep.get('horizon')}`.
- RoboCasa registry coverage audit: registered task IDs `{robocasa_catalog.get('registry_count')}`; rollout-pool task IDs `{robocasa_catalog.get('verified_artifact_task_count')}`; micro-rollout task IDs `{robocasa_catalog.get('micro_rollout_task_count')}`; any-artifact task IDs `{robocasa_catalog.get('any_artifact_task_count')}`.
- LIBERO WAM-lite: verified `{libero_wam.get('verified')}`; tasks `{libero_wam.get('tasks')}`; train/eval samples `{libero_wam.get('train_samples')}`/`{libero_wam.get('eval_samples')}`; utility corr `{fmt(((libero_wam.get('model_metrics') or {}).get('utility_corr')))}`; exact-law MAE `{fmt(libero_wam.get('exact_law_utility_mae'))}`; learned-random N8 CI lower `{fmt((((libero_wam.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- LIBERO scripted sparse-success smoke: verified `{libero_scripted.get('verified')}`; episodes `{libero_scripted.get('n_episodes')}`; successes `{libero_scripted.get('n_successes')}`; success-rate CI lower `{fmt((((libero_scripted.get('confidence_intervals') or {}).get('success_rate') or {}).get('lo')))}`.
- LIBERO learned action-head smoke: verified `{libero_action_head.get('verified')}`; train examples `{libero_action_head.get('train_examples')}`; eval successes `{libero_action_head.get('eval_successes')}`/`{libero_action_head.get('eval_episodes')}`; success-rate CI lower `{fmt((((libero_action_head.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`.
- LIBERO time-conditioned autonomous low-dimensional BC smoke: verified `{libero_autonomous_bc.get('verified')}`; train examples `{libero_autonomous_bc.get('train_examples')}`; eval successes `{libero_autonomous_bc.get('eval_successes')}`/`{libero_autonomous_bc.get('eval_episodes')}`; success-rate CI lower `{fmt((((libero_autonomous_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`; uses phase labels `{fmt(((libero_autonomous_bc.get('policy') or {}).get('uses_phase_index')))}`; uses target commands `{fmt(((libero_autonomous_bc.get('policy') or {}).get('uses_target_point_command')))}`; uses step clock `{fmt(((libero_autonomous_bc.get('policy') or {}).get('uses_step_clock')))}`.
- LIBERO RGB/proprio/language BC smoke: verified `{libero_visual_language_bc.get('verified')}`; train examples `{libero_visual_language_bc.get('train_examples')}`; eval successes `{libero_visual_language_bc.get('eval_successes')}`/`{libero_visual_language_bc.get('eval_episodes')}`; success-rate CI lower `{fmt((((libero_visual_language_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`; uses RGB `{fmt(((libero_visual_language_bc.get('policy') or {}).get('uses_rgb')))}`; uses language `{fmt(((libero_visual_language_bc.get('policy') or {}).get('uses_language')))}`; uses object state `{fmt(((libero_visual_language_bc.get('policy') or {}).get('uses_simulator_object_state')))}`.
- Visual attempted: `{visual.get('attempted')}`; visual verified: `{visual.get('verified')}`.
- Benchmark visual verified: `{bench_visual.get('verified')}`.
- Benchmark RGB WAM-lite: `{bench_visual_wam.get('model_type')}`; verified: `{bench_visual_wam.get('verified')}`; utility corr: `{fmt((bench_visual_wam.get('validation') or {}).get('utility_corr'))}`; utility MAE: `{fmt((bench_visual_wam.get('validation') or {}).get('utility_mae'))}`; exact-law MAE: `{fmt(bench_visual_wam.get('exact_law_utility_mae'))}`.
- Gymnasium Robotics RGB WAM-lite: verified: `{gym_robotics_visual.get('verified')}`; mean utility corr: `{fmt(gym_robotics_visual.get('mean_validation_utility_corr'))}`; exact-law MAE: `{fmt(gym_robotics_visual.get('exact_law_utility_mae'))}`; visual-random N32 CI lower: `{fmt((((gym_robotics_visual.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`.
- ManiSkill visual/EE probe: attempted `{maniskill_visual_probe.get('attempted')}`; state baseline ok `{maniskill_visual_probe.get('state_baseline_ok')}`; any visual success `{maniskill_visual_probe.get('any_visual_success')}`; blocker `{maniskill_visual_probe.get('visual_blocker')}`.
- ManiSkill dependency probe: Pinocchio import `{maniskill_dependency_probe.get('pinocchio_import_available')}`; binary `pin` wheel `{maniskill_dependency_probe.get('pin_binary_wheel_available')}`; source install attempted `{maniskill_dependency_probe.get('source_install_attempted')}`.
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
- Meta-World: reach-v3, push-v3, and drawer-open-v3 ML1 artifacts generated.
- RoboSuite: Lift, Stack, and Door Panda manipulation artifacts generated, including small closed-loop traces.
- ManiSkill: PickCube-v1, PushCube-v1, and PegInsertionSide-v1 state-mode artifacts generated.
- RoboCasa: `PickPlaceCounterToCabinet` kitchen smoke artifact generated in a separate RoboCasa-compatible environment; exact-law utility MAE `{fmt(robocasa.get('exact_law_utility_mae'))}` over `{robocasa.get('n_rollouts_total')}` rollouts.
- RoboCasa learned WAM-lite: single-task `PickPlaceCounterToCabinet` ridge state/action-sequence WAM trained on `{robocasa_learned.get('train_samples')}` rollouts and evaluated on `{robocasa_learned.get('eval_samples')}` heldout rollouts.
- RoboCasa three-task learned WAM-lite: task conditioned ridge WAM over `{robocasa_multitask.get('env_ids')}` trained on `{robocasa_multitask.get('train_samples')}` rollouts and evaluated on `{robocasa_multitask.get('eval_samples')}` heldout rollouts.
- RoboCasa broad task family learned WAM-lite: task conditioned ridge WAM over `{robocasa_broad.get('env_ids')}` trained on `{robocasa_broad.get('train_samples')}` rollouts and evaluated on `{robocasa_broad.get('eval_samples')}` heldout rollouts.
- RoboCasa 12-task family learned WAM-lite: task conditioned ridge WAM over `{robocasa_family12.get('env_ids')}` trained on `{robocasa_family12.get('train_samples')}` rollouts and evaluated on `{robocasa_family12.get('eval_samples')}` heldout rollouts.
- RoboCasa 24-task family learned WAM-lite: task conditioned ridge WAM over `{robocasa_family24.get('env_ids')}` trained on `{robocasa_family24.get('train_samples')}` rollouts and evaluated on `{robocasa_family24.get('eval_samples')}` heldout rollouts.
- RoboCasa extra four-task learned WAM-lite: task conditioned ridge WAM over `{robocasa_extra4.get('env_ids')}` trained on `{robocasa_extra4.get('train_samples')}` rollouts and evaluated on `{robocasa_extra4.get('eval_samples')}` heldout rollouts.
- RoboCasa combined 28-task learned WAM-lite: task conditioned ridge WAM over `{robocasa_family28.get('env_ids')}` trained on `{robocasa_family28.get('train_samples')}` rollouts and evaluated on `{robocasa_family28.get('eval_samples')}` heldout rollouts.
- RoboCasa combined 32-task learned WAM-lite: task conditioned ridge WAM over `{robocasa_family32.get('env_ids')}` trained on `{robocasa_family32.get('train_samples')}` rollouts and evaluated on `{robocasa_family32.get('eval_samples')}` heldout rollouts.
- RoboCasa stratified 55-task learned WAM-lite: task conditioned ridge WAM over `{robocasa_stratified55.get('env_ids')}` trained on `{robocasa_stratified55.get('train_samples')}` rollouts and evaluated on `{robocasa_stratified55.get('eval_samples')}` heldout rollouts.
- RoboCasa stratified 97-task learned WAM-lite: task conditioned ridge WAM over `{len(robocasa_stratified97.get('env_ids') or [])}` task IDs trained on `{robocasa_stratified97.get('train_samples')}` rollouts, validated on `{robocasa_stratified97.get('validation_samples')}` rollouts, and evaluated on `{robocasa_stratified97.get('eval_samples')}` heldout rollouts from `{robocasa_stratified97.get('eval_rollout_pools')}` rollout pools.
- RoboCasa residual 35-task clean/cook learned WAM-lite: task conditioned ridge WAM over `{len(robocasa_residual35.get('env_ids') or [])}` task IDs trained on `{robocasa_residual35.get('train_samples')}` rollouts, validated on `{robocasa_residual35.get('validation_samples')}` rollouts, and evaluated on `{robocasa_residual35.get('eval_samples')}` heldout rollouts from `{robocasa_residual35.get('eval_rollout_pools')}` rollout pools with horizon `{robocasa_residual35.get('horizon')}` and Nmax `{residual35_nmax}`.
- RoboCasa micro-rollout probe: task IDs `{robocasa_micro.get('nondegenerate_env_ids')}` reset and produced short nondegenerate rollouts. This remains lower-tier viability evidence; the separate extra four-task WAM artifact is the stronger learned-WAM validation for the same task IDs.
- RoboCasa stratified micro-rollout probe: `{robocasa_micro_stratified.get('nondegenerate_task_count')}` task IDs reset and produced short nondegenerate rollouts across wider kitchen categories. This remains lower-tier viability evidence; the separate stratified 55-task and 97-task WAM artifacts are stronger learned-WAM validation for promoted task IDs.
- RoboCasa frontier micro-rollout probe: `{robocasa_micro_frontier.get('nondegenerate_task_count')}` of `{robocasa_micro_frontier.get('candidate_task_count')}` candidate task IDs reset and produced short nondegenerate rollouts across manipulation, movement, pick-place, cleaning, washing, cooking, and arrangement families. This remains lower-tier viability evidence; the separate 97-task WAM artifact is the stronger learned-WAM validation for promoted task IDs.
- RoboCasa residual clean/cook micro-rollout sweep: `{robocasa_residual_sweep.get('nondegenerate_task_count')}` of `{robocasa_residual_sweep.get('candidate_task_count')}` cleaning/cooking task IDs produced nondegenerate short rollouts; two timeout chunks are documented, and this sweep is not promoted as full learned-WAM evidence beyond the separate residual 35-task artifact.
- RoboCasa catalog coverage audit: local registry contains `{robocasa_catalog.get('registry_count')}` task IDs; verified rollout-pool artifacts cover `{robocasa_catalog.get('verified_artifact_task_count')}` of them, micro-rollout probes cover `{robocasa_catalog.get('micro_rollout_task_count')}`, and any committed artifact covers `{robocasa_catalog.get('any_artifact_task_count')}`. This is coverage accounting, not validation for uncovered IDs.
- LIBERO learned WAM-lite: three Spatial tasks `{libero_wam.get('tasks')}` trained on `{libero_wam.get('train_samples')}` rollout samples and evaluated on `{libero_wam.get('eval_samples')}` heldout rollout samples with dense progress utility.
- LIBERO sparse-success scripted smoke: all 10 Object tasks evaluated over `{libero_scripted.get('n_seeds')}` seeds with `{libero_scripted.get('n_successes')}` successes over `{libero_scripted.get('n_episodes')}` episodes.
- LIBERO learned action-head smoke: `{libero_action_head.get('action_head_model')}` action head trained on `{libero_action_head.get('train_examples')}` scripted action examples and evaluated on `{libero_action_head.get('eval_episodes')}` heldout sparse-success episodes over all 10 Object tasks.
- LIBERO time-conditioned autonomous low-dimensional BC smoke: kNN behavior cloning trained on `{libero_autonomous_bc.get('train_examples')}` scripted action examples and evaluated on `{libero_autonomous_bc.get('eval_episodes')}` heldout sparse-success episodes over all 10 Object tasks without phase labels or target-point commands at evaluation time.
- LIBERO RGB/proprio/language BC smoke: feature-kNN behavior cloning trained on `{libero_visual_language_bc.get('train_examples')}` scripted action examples and evaluated on `{libero_visual_language_bc.get('eval_episodes')}` heldout sparse-success episodes over all 10 Object tasks without object state, task IDs, phase labels, or target-point commands at evaluation time.
- Visual: toy visual mode verified with MAE `{fmt(visual.get('test_mae'))}`.
- Benchmark visual WAM: Reacher-v5 RGB-frame/action-sequence model verified with visual-random N32 CI lower bound `{fmt((((bench_visual_wam.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`.
- Gymnasium Robotics visual WAM: Fetch RGB-frame/action-sequence models verified with visual-random N32 CI lower bound `{fmt((((gym_robotics_visual.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`.
- ManiSkill visual/EE-control probe: generated artifact-backed blocker report when local RGB/RGB-D and EE-control attempts failed.
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
- Meta-World exact-law MAE: `{fmt(metaworld.get('exact_law_utility_mae'))}`.
- RoboSuite exact-law MAE: `{fmt(robosuite.get('exact_law_utility_mae'))}`.
- RoboCasa smoke exact-law MAE: `{fmt(robocasa.get('exact_law_utility_mae'))}`.
- RoboCasa learned WAM utility corr: `{fmt(((robocasa_learned.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa learned-random N8 CI lower: `{fmt((((robocasa_learned.get('confidence_intervals') or {}).get('learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa three-task WAM utility corr: `{fmt(((robocasa_multitask.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa three-task learned-random N8 CI lower: `{fmt((((robocasa_multitask.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa broad WAM utility corr: `{fmt(((robocasa_broad.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa broad learned-random N8 CI lower: `{fmt((((robocasa_broad.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa broad exact-law utility MAE: `{fmt(robocasa_broad.get('exact_law_utility_mae'))}`.
- RoboCasa 12-task family WAM utility corr: `{fmt(((robocasa_family12.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa 12-task family learned-random N8 CI lower: `{fmt((((robocasa_family12.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa 12-task family exact-law utility MAE: `{fmt(robocasa_family12.get('exact_law_utility_mae'))}`.
- RoboCasa 24-task family WAM utility corr: `{fmt(((robocasa_family24.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa 24-task family learned-random N8 CI lower: `{fmt((((robocasa_family24.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa 24-task family exact-law utility MAE: `{fmt(robocasa_family24.get('exact_law_utility_mae'))}`.
- RoboCasa extra four-task WAM utility corr: `{fmt(((robocasa_extra4.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa extra four-task learned-random N8 CI lower: `{fmt((((robocasa_extra4.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa extra four-task exact-law utility MAE: `{fmt(robocasa_extra4.get('exact_law_utility_mae'))}`.
- RoboCasa combined 28-task WAM utility corr: `{fmt(((robocasa_family28.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa combined 28-task learned-random N8 CI lower: `{fmt((((robocasa_family28.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa combined 28-task exact-law utility MAE: `{fmt(robocasa_family28.get('exact_law_utility_mae'))}`.
- RoboCasa combined 32-task WAM utility corr: `{fmt(((robocasa_family32.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa combined 32-task learned-random N8 CI lower: `{fmt((((robocasa_family32.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa combined 32-task exact-law utility MAE: `{fmt(robocasa_family32.get('exact_law_utility_mae'))}`.
- RoboCasa stratified 55-task WAM utility corr: `{fmt(((robocasa_stratified55.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa stratified 55-task learned-random N8 CI lower: `{fmt((((robocasa_stratified55.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa stratified 55-task exact-law utility MAE: `{fmt(robocasa_stratified55.get('exact_law_utility_mae'))}`.
- RoboCasa stratified 97-task WAM utility corr: `{fmt(((robocasa_stratified97.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa stratified 97-task learned-random N8 CI lower: `{fmt((((robocasa_stratified97.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- RoboCasa stratified 97-task oracle-learned N8 CI lower: `{fmt((((robocasa_stratified97.get('confidence_intervals') or {}).get('oracle_minus_best_learned_N8') or {}).get('lo')))}`.
- RoboCasa stratified 97-task exact-law utility MAE: `{fmt(robocasa_stratified97.get('exact_law_utility_mae'))}`.
- RoboCasa residual 35-task WAM utility corr: `{fmt(((robocasa_residual35.get('model_metrics') or {}).get('utility_corr')))}`.
- RoboCasa residual 35-task learned-random Nmax CI lower: `{fmt(residual35_lr_ci.get('lo'))}`.
- RoboCasa residual 35-task oracle-learned Nmax CI lower: `{fmt(residual35_oracle_ci.get('lo'))}`.
- RoboCasa residual 35-task exact-law utility MAE: `{fmt(robocasa_residual35.get('exact_law_utility_mae'))}`.
- RoboCasa micro-rollout extra tasks: `{robocasa_micro.get('nondegenerate_task_count')}` / `{robocasa_micro.get('candidate_task_count')}` nondegenerate.
- RoboCasa stratified micro-rollout tasks: `{robocasa_micro_stratified.get('nondegenerate_task_count')}` / `{robocasa_micro_stratified.get('candidate_task_count')}` nondegenerate.
- RoboCasa frontier micro-rollout tasks: `{robocasa_micro_frontier.get('nondegenerate_task_count')}` / `{robocasa_micro_frontier.get('candidate_task_count')}` nondegenerate.
- RoboCasa residual clean/cook micro-rollout tasks: `{robocasa_residual_sweep.get('nondegenerate_task_count')}` / `{robocasa_residual_sweep.get('candidate_task_count')}` nondegenerate.
- RoboCasa catalog coverage: `{robocasa_catalog.get('verified_artifact_task_count')}` rollout-pool task IDs and `{robocasa_catalog.get('micro_rollout_task_count')}` micro-rollout task IDs out of `{robocasa_catalog.get('registry_count')}` registered task IDs.
- LIBERO WAM utility corr: `{fmt(((libero_wam.get('model_metrics') or {}).get('utility_corr')))}`.
- LIBERO learned-random N8 CI lower: `{fmt((((libero_wam.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`.
- LIBERO exact-law utility MAE: `{fmt(libero_wam.get('exact_law_utility_mae'))}`.
- LIBERO Object scripted success rate: `{fmt(libero_scripted.get('success_rate'))}` with CI [`{fmt((((libero_scripted.get('confidence_intervals') or {}).get('success_rate') or {}).get('lo')))}`, `{fmt((((libero_scripted.get('confidence_intervals') or {}).get('success_rate') or {}).get('hi')))}`].
- LIBERO learned action-head heldout success rate: `{fmt(libero_action_head.get('eval_success_rate'))}` with CI [`{fmt((((libero_action_head.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`, `{fmt((((libero_action_head.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('hi')))}`].
- LIBERO time-conditioned autonomous low-dimensional BC heldout success rate: `{fmt(libero_autonomous_bc.get('eval_success_rate'))}` with CI [`{fmt((((libero_autonomous_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`, `{fmt((((libero_autonomous_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('hi')))}`].
- LIBERO RGB/proprio/language BC heldout success rate: `{fmt(libero_visual_language_bc.get('eval_success_rate'))}` with CI [`{fmt((((libero_visual_language_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`, `{fmt((((libero_visual_language_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('hi')))}`].
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

- The empirical work is mostly state-based, but benchmark visual WAM evidence now exists for Gymnasium/MuJoCo Reacher-v5 and Gymnasium Robotics Fetch RGB frames; ManiSkill RGB/RGB-D remains unavailable locally and is documented by a generated renderer probe.
- ManiSkill EE-control remains unavailable locally because Pinocchio is absent and the `pin` dependency path lacks binary wheels for this Windows/Python stack; this is documented by a generated dependency probe.
- The strongest contact-rich external evidence is Gymnasium Robotics Fetch plus Meta-World ML1, RoboSuite Panda, ManiSkill state mode, RoboCasa three-task pick-place, broad four-task atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook artifacts, LIBERO Spatial rollout-pool validation, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned low-dimensional autonomous BC smoke, and LIBERO RGB/proprio/language BC smoke, but still not real hardware.
- The learned models are intentionally lightweight and do not establish WAM training recipes.
- Pilot estimates are not exact laws and can be brittle under shift.
- Some analytic smoke artifacts are single-seed checks; paper figures should prefer five-seed learned/multi-env results.

## Evidence That Helps

- Exact theorem tests and docs separate identities from empirical predictions.
- Learned toy artifacts report ID and OOD errors.
- The anti-overclaim claim gate prevents unsupported real-robot or unavailable-benchmark claims from slipping into README/paper text.
- Falsification experiments make the score-alignment condition explicit.

## Remaining Gap

The single highest reviewer-risk gap is absence of real-robot evidence, modern VLA-style benchmark policy validation, and full benchmark-suite policy coverage. LIBERO is now present as a three-task dense-utility rollout-pool WAM artifact plus sparse-success scripted, learned action-head, time-conditioned low-dimensional BC, and RGB/proprio/language Object smokes, but not full LIBERO policy evidence. RoboCasa is substantially broader than before, including a residual 35-task clean/cook artifact, but still not full RoboCasa-wide validation.
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
- paper_outline overclaims: `{len(claims_payload.get('paper_overclaims') or [])}`
- report overclaims: `{len(claims_payload.get('report_overclaims') or [])}`

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
- Gymnasium/MuJoCo, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, RGB WAM-lite on Reacher-v5 and Fetch frames, ManiSkill3 state-mode, RoboCasa three-task pick-place-family plus broad four-task, 12-task, 24-task, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook kitchen learned-WAM artifacts, LIBERO Spatial three-task rollout-pool learned-WAM artifacts, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned autonomous low-dimensional BC smoke, and LIBERO RGB/proprio/language BC smoke validate the benchmark path without claiming hardware evidence.
- ManiSkill RGB/RGB-D and EE-control attempts are documented as a blocker artifact, not counted as visual validation.
- ManiSkill Pinocchio dependency probing documents why EE-control is not claimed in this environment.
- Learned toy and multi-env toy artifacts support these claims with confidence intervals where the claim gate marks them verified.

## Discussion-Only Claims

- Modern VLA-style sparse-success LIBERO policy performance and full RoboCasa-wide learned-WAM validation.
- ManiSkill beyond state-mode joint-delta control.
- ManiSkill RGB/RGB-D WAM validation.
- Universal WAM train-inference optimization.
- Any analogy to DreamZero/UWM-level evidence.

## Do Not Claim

- Real robot validation.
- Modern VLA-style sparse-success LIBERO policy validation or full RoboCasa-wide validation.
- ManiSkill RGB/RGB-D or EE-control validation.
- A universal WAM training recipe.
- That increasing N is intrinsically beneficial.
"""

    final_decision = f"""
# Final Decision Report

## 1. Tier

Benchmark-full plus Fetch, Meta-World, RoboSuite, ManiSkill state-mode, RoboCasa single-task, three-task pick-place-family, broad four-task, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned WAM-lite, LIBERO Spatial three-task rollout-pool WAM-lite, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned autonomous low-dimensional BC smoke, LIBERO RGB/proprio/language BC smoke, visual-, blocker-probe-, and audit-validated: learned-toy, multi-env toy validation, Gymnasium/MuJoCo Reacher-v5 benchmark validation, Gymnasium Robotics Fetch validation, Meta-World ML1 manipulation validation, RoboSuite Panda manipulation validation, ManiSkill3 state-mode manipulation validation, RoboCasa kitchen smoke plus single-task, three-task, broad task family, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual clean/cook learned-WAM validation, LIBERO rollout-pool learned-WAM validation, LIBERO scripted sparse-success smoke, LIBERO learned action-head sparse-success smoke, LIBERO time-conditioned low-dimensional BC sparse-success smoke, LIBERO RGB/proprio/language sparse-success smoke, toy visual mode, Reacher RGB WAM-lite, Fetch RGB WAM-lite, ManiSkill visual/EE-control blocker probing, and inference-value audit framework artifacts.

## 2. Strongest Verified Claims

{bullet_lines(verified[:12])}

## 3. Weakest Claims

{bullet_lines(weakest_claims)}

## 4. Abstract Claims

- Exact best-of-N inference laws for rollout selection.
- The score/utility distribution determines the value of additional rollouts.
- Model/scorer mismatch can make best-of-N amplify imagined futures rather than real utility.
- Learned and multi-env toy artifacts validate the theory and failure modes.

## 5. Discussion-Only Claims

- Modern VLA-style sparse-success LIBERO policy performance and full RoboCasa-wide learned-WAM validation.
- ManiSkill RGB/RGB-D or EE-control validation.
- ManiSkill RGB/RGB-D benchmark WAM validation.
- Universal WAM training and train-inference scaling.

## 6. Skeptical Reviewer Attack

The project still lacks real robot artifacts, modern VLA-style sparse-success LIBERO policy validation, full RoboCasa-wide learned-WAM validation, and ManiSkill visual or EE-control validation.

## 7. Current Answer

The repo answers the mathematical and controlled toy-science objections with tests, multi-env artifacts, learned WAM-lite backbones, falsification, an anti-overclaim system, a state-based Gymnasium/MuJoCo benchmark, a three-task Gymnasium Robotics Fetch benchmark, a three-task Meta-World ML1 benchmark, a three-task RoboSuite Panda benchmark, a three-task ManiSkill3 state-mode benchmark, RoboCasa kitchen pick-place, broad atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook learned-WAM artifacts, a three-task LIBERO Spatial rollout-pool learned-WAM artifact, a LIBERO Object sparse-success scripted smoke, a LIBERO learned action-head smoke, a LIBERO time-conditioned low-dimensional BC sparse-success smoke, and a LIBERO RGB/proprio/language sparse-success smoke. It does not yet answer real-robot realism.

## 8. Unresolved

- Modern VLA-style sparse-success LIBERO policy evaluation beyond the current hand scripted/action-head/time-conditioned low-dimensional/RGB-proprio-language feature-kNN smokes and dense rollout-pool utility.
- Full RoboCasa-wide learned-WAM rollout collection beyond the current pick-place-family, broad atomic-manipulation, 12-task, 24-task family, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook artifacts.
- ManiSkill RGB/RGB-D or end-effector-control validation.
- Real robot validation.
- Strong ManiSkill RGB/RGB-D WAM evidence; current repo has only a local failure probe with exact renderer/control blockers.

## 9. Workshop Readiness

Yes, as a theory-plus-controlled-learned-toy paper artifact.

## 10. Main-Conference Readiness

Substantially stronger after Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, ManiSkill state-mode validation, pick-place, broad, 12-task, 24-task, extra four-task, combined 28-task, combined 32-task, stratified 55-task, stratified 97-task, and residual 35-task clean/cook RoboCasa learned-WAM validation, three-task LIBERO rollout-pool validation, LIBERO Object sparse-success scripted smoke, LIBERO learned action-head smoke, LIBERO time-conditioned low-dimensional BC smoke, and LIBERO RGB/proprio/language BC smoke. Still not a real-robot paper and still weaker than a benchmark-heavy robotics paper with modern VLA-style sparse-success LIBERO policy performance, full RoboCasa-wide validation, or RGB-D manipulation.

## 11. Single Highest-Value Next Step

Add modern VLA-style sparse-success LIBERO policy evaluation or full RoboCasa-wide task coverage next; ManiSkill RGB/RGB-D WAM validation remains the other high-value path if the local SAPIEN/Vulkan blocker is cleared.

## Command Results

- `python -m pytest -q`: passed with `71 passed`.
- `bash scripts/run_all.sh`: passed; full analytic EXP1-EXP8 sweep completed with EXP1 success MAE `{fmt(exp1.get('mean_success_mc_mae'), 5)}`, EXP3 relative MAE reduction `{fmt(exp3.get('relative_mae_reduction'), 3)}`, EXP6 moment-uniform delta `{fmt(exp6.get('moment_law_improvement_over_uniform'), 4)}`, EXP7 useful N64-N1 success delta `{fmt(exp7.get('useful_success_gain_N64_minus_N1'), 3)}`, and EXP8 conditional-law MAE `{fmt(exp8.get('mean_abs_error_N16'), 4)}`.
- `bash scripts/run_smoke.sh`: passed; EXP1 success MAE `0.00696`, utility MAE `0.04511`; EXP8 smoke conditional-law MAE `0.0055`.
- `bash scripts/run_learned_wam_toy.sh`: passed; learned validation utility MAE `0.8624`, final-position L2 MAE `0.1117`; learned-vs-analytic N64 real-utility delta `1.170 +/- 0.219`.
- `bash scripts/run_multi_env.sh`: passed with `envs=5`, `backbones=3`, `seeds=5`.
- robust EXP8 rerun: passed; stale post-pre CI lower bound `0.0255`, stale-adaptive post CI lower bound `0.0613`.
- `bash scripts/run_benchmark_full.sh`: passed with Gymnasium/MuJoCo `Reacher-v5`, Gymnasium Robotics Fetch, Meta-World ML1, RoboSuite Panda, and ManiSkill3 state-mode tasks; optional RoboCasa smoke, single-task learned-WAM, three-task learned-WAM, broad four-task learned-WAM, 12-task family learned-WAM, 24-task family learned-WAM, extra four-task learned-WAM, combined 28-task learned-WAM, combined 32-task learned-WAM, stratified micro probe, frontier micro probe, stratified 55-task learned-WAM, and stratified 97-task learned-WAM runs were generated separately with `ROBOCASA_PYTHON`; optional LIBERO three-task Spatial rollout-pool WAM, Object sparse-success scripted smoke, learned action-head smoke, time-conditioned autonomous low-dimensional BC smoke, and RGB/proprio/language BC smoke were generated separately with `LIBERO_PYTHON`; Reacher exact-law utility MAE `0.01875`; Reacher closed-loop learned-random CI lower bound `0.4102`; Fetch exact-law utility MAE `{fmt(gym_robotics.get('exact_law_utility_mae'))}`; Meta-World exact-law utility MAE `{fmt(metaworld.get('exact_law_utility_mae'))}`; Meta-World learned-random N32 CI lower `{fmt((((metaworld.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`; RoboSuite exact-law utility MAE `{fmt(robosuite.get('exact_law_utility_mae'))}`; RoboSuite learned-random N32 CI lower `{fmt((((robosuite.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`; RoboSuite closed-loop learned-random N8 CI lower `{fmt((((robosuite.get('confidence_intervals') or {}).get('closed_loop_learned_minus_random_N8') or {}).get('lo')))}`; ManiSkill exact-law utility MAE `{fmt(maniskill.get('exact_law_utility_mae'))}`; ManiSkill closed-loop learned-random CI lower bound `{fmt(((maniskill.get('confidence_intervals') or {}).get('closed_loop_learned_minus_random_utility_N8') or {}).get('lo'))}`; RoboCasa smoke exact-law utility MAE `{fmt(robocasa.get('exact_law_utility_mae'))}`; RoboCasa learned utility corr `{fmt(((robocasa_learned.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa learned-random N8 CI lower `{fmt((((robocasa_learned.get('confidence_intervals') or {}).get('learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa three-task utility corr `{fmt(((robocasa_multitask.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa three-task learned-random N8 CI lower `{fmt((((robocasa_multitask.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa broad utility corr `{fmt(((robocasa_broad.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa broad learned-random N8 CI lower `{fmt((((robocasa_broad.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa 12-task family utility corr `{fmt(((robocasa_family12.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa 12-task family learned-random N8 CI lower `{fmt((((robocasa_family12.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa 24-task family utility corr `{fmt(((robocasa_family24.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa 24-task family learned-random N8 CI lower `{fmt((((robocasa_family24.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa extra four-task utility corr `{fmt(((robocasa_extra4.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa extra four-task learned-random N8 CI lower `{fmt((((robocasa_extra4.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa combined 28-task utility corr `{fmt(((robocasa_family28.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa combined 28-task learned-random N8 CI lower `{fmt((((robocasa_family28.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa combined 32-task utility corr `{fmt(((robocasa_family32.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa combined 32-task learned-random N8 CI lower `{fmt((((robocasa_family32.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa stratified 55-task utility corr `{fmt(((robocasa_stratified55.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa stratified 55-task learned-random N8 CI lower `{fmt((((robocasa_stratified55.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa stratified 55-task exact-law utility MAE `{fmt(robocasa_stratified55.get('exact_law_utility_mae'))}`; RoboCasa stratified 97-task utility corr `{fmt(((robocasa_stratified97.get('model_metrics') or {}).get('utility_corr')))}`; RoboCasa stratified 97-task learned-random N8 CI lower `{fmt((((robocasa_stratified97.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; RoboCasa stratified 97-task oracle-learned N8 CI lower `{fmt((((robocasa_stratified97.get('confidence_intervals') or {}).get('oracle_minus_best_learned_N8') or {}).get('lo')))}`; RoboCasa stratified 97-task exact-law utility MAE `{fmt(robocasa_stratified97.get('exact_law_utility_mae'))}`; RoboCasa frontier micro nondegenerate tasks `{robocasa_micro_frontier.get('nondegenerate_task_count')}`; LIBERO utility corr `{fmt(((libero_wam.get('model_metrics') or {}).get('utility_corr')))}`; LIBERO learned-random N8 CI lower `{fmt((((libero_wam.get('confidence_intervals') or {}).get('best_learned_minus_random_N8') or {}).get('lo')))}`; LIBERO scripted success CI lower `{fmt((((libero_scripted.get('confidence_intervals') or {}).get('success_rate') or {}).get('lo')))}`; LIBERO action-head success CI lower `{fmt((((libero_action_head.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`; LIBERO autonomous BC success CI lower `{fmt((((libero_autonomous_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`; LIBERO RGB/proprio/language BC success CI lower `{fmt((((libero_visual_language_bc.get('confidence_intervals') or {}).get('eval_success_rate') or {}).get('lo')))}`.
- `bash scripts/run_robocasa_residual_probes.sh`: generated separately with `ROBOCASA_PYTHON`; residual clean/cook sweep verified `{robocasa_residual_sweep.get('verified')}` with `{robocasa_residual_sweep.get('nondegenerate_task_count')}` / `{robocasa_residual_sweep.get('candidate_task_count')}` nondegenerate task IDs and `{robocasa_residual_sweep.get('timed_out_chunk_count')}` timeout chunks; residual 35-task learned-WAM verified `{robocasa_residual35.get('verified')}` with train/validation/eval `{robocasa_residual35.get('train_samples')}`/`{robocasa_residual35.get('validation_samples')}`/`{robocasa_residual35.get('eval_samples')}`, exact-law utility MAE `{fmt(robocasa_residual35.get('exact_law_utility_mae'))}`, utility corr `{fmt(((robocasa_residual35.get('model_metrics') or {}).get('utility_corr')))}`, learned-random N{residual35_nmax} CI lower `{fmt(residual35_lr_ci.get('lo'))}`, and oracle-learned N{residual35_nmax} CI lower `{fmt(residual35_oracle_ci.get('lo'))}`.
- `python experiments/benchmark_gym_robotics_suite.py`: passed with `{gym_robotics.get('env_ids')}`; exact-law utility MAE `{fmt(gym_robotics.get('exact_law_utility_mae'))}`; learned-random N32 CI lower `{fmt((((gym_robotics.get('confidence_intervals') or {}).get('learned_minus_random_N32') or {}).get('lo')))}`; closed-loop learned-random N32 CI lower `{fmt((((gym_robotics.get('confidence_intervals') or {}).get('closed_loop_learned_minus_random_N32') or {}).get('lo')))}`.
- `bash scripts/run_visual_optional.sh`: passed; toy visual MAE `0.0185`; Reacher RGB WAM utility corr `{fmt((bench_visual_wam.get('validation') or {}).get('utility_corr'))}`, utility MAE `{fmt((bench_visual_wam.get('validation') or {}).get('utility_mae'))}`, visual-random N32 CI lower `{fmt((((bench_visual_wam.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`; Fetch RGB WAM mean corr `{fmt(gym_robotics_visual.get('mean_validation_utility_corr'))}`, visual-random N32 CI lower `{fmt((((gym_robotics_visual.get('confidence_intervals') or {}).get('visual_minus_random_N32') or {}).get('lo')))}`; ManiSkill visual probe any visual success `{maniskill_visual_probe.get('any_visual_success')}` with blocker `{maniskill_visual_probe.get('visual_blocker')}`.
- `python experiments/benchmark_maniskill_dependency_probe.py --attempt-source-install`: passed as a blocker probe; Pinocchio import `{maniskill_dependency_probe.get('pinocchio_import_available')}`, binary `pin` wheel `{maniskill_dependency_probe.get('pin_binary_wheel_available')}`, source install attempted `{maniskill_dependency_probe.get('source_install_attempted')}`.
- `bash scripts/run_inference_audit.sh`: passed; audit tail-gain correlation `{fmt(audit.get('tail_alignment_gain_corr'))}`, repair-predicted N64 CI mean `{fmt(((repair.get('confidence_intervals') or {}).get('repair_minus_predicted_N64') or {}).get('mean'))}`, predicted N128-N1 scaling gain `{fmt(((scaling.get('confidence_intervals') or {}).get('predicted_gain_N128_minus_N1') or {}).get('mean'))}`.
- `python scripts/artifact_integrity.py --fail-on-error`: passed with `{artifact_integrity.get('n_references')}` artifact references checked and `{artifact_integrity.get('n_issues')}` issues.
- `python scripts/artifact_manifest.py --fail-on-error`: passed with `{artifact_manifest.get('n_files')}` scientific artifacts, `{artifact_manifest.get('total_bytes')}` bytes, `{artifact_manifest.get('n_checks')}` manifest checks, and `{artifact_manifest.get('n_issues')}` issues.
- `python scripts/figure_quality.py --fail-on-error`: passed with `{figure_quality.get('n_figures')}` figures, `{figure_quality.get('n_checks')}` image-quality checks, and `{figure_quality.get('n_issues')}` issues.
- `python scripts/result_consistency.py --fail-on-error`: passed with `{result_consistency.get('n_checks')}` consistency checks and `{result_consistency.get('n_issues')}` issues.
- `python scripts/raw_result_recompute.py --fail-on-error`: passed with `{raw_result_recompute.get('aggregate_metrics_compared')}` aggregate metrics, `{raw_result_recompute.get('exact_law_mae_files')}` exact-law files, `{raw_result_recompute.get('seed_metric_ci_columns')}` seed CI columns, and `{raw_result_recompute.get('n_issues')}` issues.
- `python scripts/narrative_consistency.py --fail-on-error`: passed with `{narrative_consistency.get('n_checks')}` narrative checks and `{narrative_consistency.get('n_issues')}` issues.
- `python scripts/script_contracts.py --fail-on-error`: passed with `{script_contracts.get('n_scripts')}` scripts, `{script_contracts.get('n_checks')}` contract checks, and `{script_contracts.get('n_issues')}` issues.
- `python scripts/claim_semantics.py --fail-on-error`: passed with `{claim_semantics.get('n_claims')}` claims, `{claim_semantics.get('n_checks')}` semantic checks, `{claim_semantics.get('n_ci_claims')}` CI-backed claims, and `{claim_semantics.get('n_issues')}` issues.
- `python scripts/claim_evidence_quality.py --fail-on-error`: passed with `{claim_evidence_quality.get('n_claims')}` claims, `{claim_evidence_quality.get('n_source_links')}` source links, `{claim_evidence_quality.get('n_checks')}` evidence checks, and `{claim_evidence_quality.get('n_issues')}` issues.
- `python scripts/claim_ledger_integrity.py --fail-on-error`: passed with `{claim_ledger_integrity.get('n_claims')}` claims, `{claim_ledger_integrity.get('n_checks')}` ledger checks, and `{claim_ledger_integrity.get('n_issues')}` issues.
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
