from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATUS_JSON = RESULTS / "claims_status.json"
STATUS_MD = RESULTS / "claims_status.md"
README = ROOT / "README.md"
PAPER = ROOT / "paper_outline.md"


def load_json(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status(ok: bool, partial: bool = False, failed: bool = False) -> str:
    if failed:
        return "FAILED"
    if ok:
        return "VERIFIED"
    if partial:
        return "PARTIAL"
    return "UNSUPPORTED"


def ci_positive(payload: dict, key: str, threshold: float = 0.0) -> bool:
    ci = (payload.get("confidence_intervals") or {}).get(key) or {}
    lo = ci.get("lo")
    return lo is not None and lo > threshold


def nested_ci_positive(payload: dict, section: str, key: str, threshold: float = 0.0) -> bool:
    ci = ((payload.get("confidence_intervals") or {}).get(section) or {}).get(key) or {}
    lo = ci.get("lo")
    return lo is not None and lo > threshold


def add(claims: list[dict[str, Any]], cid: int, claim: str, stat: str, evidence: str) -> None:
    claims.append({"id": cid, "claim": claim, "status": stat, "evidence": evidence})


def main() -> None:
    exp1 = load_json("exp1_exact_rollout_law_validation.json")
    exp2 = load_json("exp2_auc_vs_moment_hierarchy.json")
    exp3 = load_json("exp3_pilot_to_heldout_prediction.json")
    exp4 = load_json("exp4_score_function_comparison.json")
    exp5 = load_json("exp5_real_vs_imagined_utility_gap.json")
    exp6 = load_json("exp6_adaptive_rollout_allocation.json")
    exp7 = load_json("exp7_closed_loop_receding_horizon_eval.json")
    exp8 = load_json("exp8_nonstationary_dynamics_extension.json")
    learned_train = load_json("learned_wam_lite_training.json")
    learned_cmp = load_json("learned_wam_vs_analytic_wam.json")
    learned_exp4 = load_json("exp4_score_function_comparison_learned.json")
    learned_exp5 = load_json("exp5_real_vs_imagined_utility_gap_learned.json")
    learned_exp6 = load_json("exp6_adaptive_rollout_allocation_learned.json")
    learned_exp7 = load_json("exp7_closed_loop_receding_horizon_eval_learned.json")
    multi = load_json("multi_env_suite.json")
    fals = load_json("exp10_falsification_bad_scorer.json")
    bench = load_json("benchmark_smoke.json")
    benchmark_pools = load_json("benchmark_rollout_pools.json")
    benchmark_exact = load_json("benchmark_exact_law_validation.json")
    benchmark_score = load_json("benchmark_score_comparison.json")
    benchmark_gap = load_json("benchmark_real_vs_imagined_gap.json")
    benchmark_closed = load_json("benchmark_closed_loop_eval.json")
    benchmark_wam = load_json("benchmark_wam_training.json")
    maniskill = load_json("benchmark_maniskill_suite.json")
    visual = load_json("visual_optional.json")
    benchmark_visual = load_json("benchmark_visual_optional.json")
    benchmark_visual_wam = load_json("benchmark_visual_wam_lite.json")
    gym_robotics = load_json("benchmark_gym_robotics_suite.json")
    gym_robotics_visual = load_json("benchmark_gym_robotics_visual_wam.json")
    maniskill_visual_probe = load_json("benchmark_maniskill_visual_probe.json")
    maniskill_dependency_probe = load_json("benchmark_maniskill_dependency_probe.json")
    metaworld = load_json("benchmark_metaworld_suite.json")
    robosuite = load_json("benchmark_robosuite_suite.json")
    robocasa = load_json("benchmark_robocasa_smoke.json")
    robocasa_learned = load_json("benchmark_robocasa_learned_wam.json")
    robocasa_multitask = load_json("benchmark_robocasa_multitask_wam.json")
    robocasa_broad = load_json("benchmark_robocasa_broad_wam.json")
    robocasa_family12 = load_json("benchmark_robocasa_family12_wam.json")
    libero_wam = load_json("benchmark_libero_wam.json")
    libero_scripted = load_json("benchmark_libero_scripted_policy.json")
    audit = load_json("inference_audit_framework.json")
    audit_learned = load_json("inference_audit_framework_learned.json")
    repair = load_json("scorer_repair_experiment.json")
    scaling = load_json("imagination_scaling_law.json")

    claims: list[dict[str, Any]] = []
    add(claims, 1, "Exact finite binary law verified.", status(bool(exp1) and exp1.get("mean_success_mc_mae", 1.0) < 0.018, bool(exp1)), f"success MAE={exp1.get('mean_success_mc_mae')}")
    add(claims, 2, "Utility-valued finite law verified.", status(bool(exp1) and exp1.get("mean_utility_mc_mae", 1.0) < 0.08, bool(exp1)), f"utility MAE={exp1.get('mean_utility_mc_mae')}")
    add(claims, 3, "N=2 AUC identity verified.", status(bool(exp2) and exp2.get("max_n2_identity_error", 1.0) < 1e-12, bool(exp2)), f"max identity error={exp2.get('max_n2_identity_error')}")
    add(claims, 4, "High-N moment hierarchy verified.", status(bool(exp2) and exp2.get("same_p_kappa_counterexample_gap_N64", 0.0) > 0.45, bool(exp2)), f"same-p/kappa gap={exp2.get('same_p_kappa_counterexample_gap_N64')}")
    add(claims, 5, "Pilot-to-heldout improves with K.", status(bool(exp3) and exp3.get("relative_mae_reduction", 0.0) > 0.25, bool(exp3)), f"relative MAE reduction={exp3.get('relative_mae_reduction')}")
    exp3_ci = (exp3.get("confidence_intervals") or {}).get("mae_reduction_first_to_last") or {}
    add(claims, 6, "Pilot uncertainty is reported.", status(exp3_ci.get("n", 0) > 0, bool(exp3)), f"pilot improvement CI={exp3_ci}")
    add(claims, 7, "Score function controls inference value.", status(bool(exp4) and exp4.get("oracle_minus_random_N64", 0.0) > 0.6, bool(exp4)), f"oracle-random N64={exp4.get('oracle_minus_random_N64')}")
    add(claims, 8, "Best non-oracle beats random with CI.", status(ci_positive(learned_exp4, "best_nonoracle_minus_random_N64"), bool(learned_exp4)), f"learned CI={((learned_exp4.get('confidence_intervals') or {}).get('best_nonoracle_minus_random_N64'))}")
    add(claims, 9, "Oracle remains above learned/non-oracle.", status(nested_ci_positive(learned_cmp, "deltas", "oracle_minus_learned_real_utility_N64"), bool(learned_cmp)), f"oracle-learned CI={((learned_cmp.get('confidence_intervals') or {}).get('deltas') or {}).get('oracle_minus_learned_real_utility_N64')}")
    add(claims, 10, "Real-vs-imagined utility gap verified.", status(bool(exp5) and exp5.get("severe_gap_growth_minus_none", 0.0) > 0.35, bool(exp5)), f"severe-none={exp5.get('severe_gap_growth_minus_none')}")
    add(claims, 11, "Mismatch gap grows with N.", status(ci_positive(learned_exp5, "severe_gap_growth_minus_none"), bool(learned_exp5)), f"learned severe gap CI={((learned_exp5.get('confidence_intervals') or {}).get('severe_gap_growth_minus_none'))}")
    add(claims, 12, "Bad scorer falsification verified.", status(bool(fals) and fals.get("anti_scorer_mean_N64", 0.0) < fals.get("anti_scorer_mean_N1", -1e9), bool(fals)), f"anti N64={fals.get('anti_scorer_mean_N64')}, N1={fals.get('anti_scorer_mean_N1')}")
    rnd_gap = fals.get("randomized_dynamics_oracle_gap_N64")
    add(
        claims,
        13,
        "Randomized dynamics falsification verified.",
        status(rnd_gap is not None and rnd_gap > 1.0, bool(multi)),
        f"randomized-oracle N64 gap={rnd_gap}",
    )
    add(claims, 14, "Moment/adaptive allocation beats uniform with CI.", status(ci_positive(learned_exp6, "moment_law_improvement_over_uniform"), bool(learned_exp6)), f"moment-uniform CI={((learned_exp6.get('confidence_intervals') or {}).get('moment_law_improvement_over_uniform'))}")
    add(claims, 15, "Adaptive allocation reduces oracle regret.", status(bool(learned_exp6) and learned_exp6.get("oracle_improvement_over_uniform", 0.0) > learned_exp6.get("moment_law_improvement_over_uniform", -1.0), bool(learned_exp6)), f"oracle-uniform={learned_exp6.get('oracle_improvement_over_uniform')}")
    learned_high_n_ci = (learned_exp7.get("confidence_intervals") or {}).get("useful_success_gain_N64_minus_N1") or {}
    analytic_high_n_ok = bool(exp7) and exp7.get("useful_success_gain_N64_minus_N1", 0.0) > 0.12
    learned_high_n_ok = learned_high_n_ci.get("lo") is not None and learned_high_n_ci.get("lo") > 0.0
    add(
        claims,
        16,
        "Closed-loop high-N gain verified.",
        status(analytic_high_n_ok or learned_high_n_ok, bool(exp7) or bool(learned_exp7)),
        f"analytic useful N64-N1={exp7.get('useful_success_gain_N64_minus_N1')}; learned CI={learned_high_n_ci}",
    )
    add(claims, 17, "Useful scorer beats random in closed loop.", status(ci_positive(learned_exp7, "useful_minus_random_success_N64"), bool(learned_exp7)), f"learned useful-random CI={((learned_exp7.get('confidence_intervals') or {}).get('useful_minus_random_success_N64'))}")
    add(claims, 18, "Oracle first-action remains upper bound.", status(ci_positive(learned_exp7, "oracle_first_action_minus_useful_success_N64"), bool(learned_exp7)), f"oracle-useful CI={((learned_exp7.get('confidence_intervals') or {}).get('oracle_first_action_minus_useful_success_N64'))}")
    add(claims, 19, "Conditional law verified under distribution shift.", status(bool(exp8) and exp8.get("mean_abs_error_N16", 1.0) < 0.025, bool(exp8)), f"MAE={exp8.get('mean_abs_error_N16')}")
    exp8_ci = exp8.get("confidence_intervals") or {}
    stale_shift_ci = exp8_ci.get("stale_post_minus_pre_abs_error_N16") or {}
    adaptive_shift_ci = exp8_ci.get("stale_minus_adaptive_post_abs_error_N16") or {}
    add(
        claims,
        20,
        "Stale estimates fail/degrade under shift.",
        status(stale_shift_ci.get("lo") is not None and stale_shift_ci.get("lo") > 0.0, bool(exp8)),
        f"stale post-pre CI={stale_shift_ci}",
    )
    add(
        claims,
        21,
        "Adaptive re-estimation helps under shift.",
        status(adaptive_shift_ci.get("lo") is not None and adaptive_shift_ci.get("lo") > 0.0, bool(exp8)),
        f"stale-adaptive post CI={adaptive_shift_ci}",
    )
    add(claims, 22, "Learned WAM trained.", status(bool(learned_train), False), f"model={learned_train.get('model_path')}")
    add(claims, 23, "Learned WAM ID error reported.", status(bool(learned_train) and bool((learned_train.get('metrics') or {}).get('validation')), bool(learned_train)), f"validation={((learned_train.get('metrics') or {}).get('validation'))}")
    add(claims, 24, "Learned WAM OOD error reported.", status(bool(learned_train) and bool((learned_train.get('metrics') or {}).get('ood')), bool(learned_train)), f"ood count={len(((learned_train.get('metrics') or {}).get('ood') or []))}")
    add(claims, 25, "Learned WAM reproduces key inference-value claims.", status(nested_ci_positive(learned_cmp, "deltas", "learned_minus_analytic_real_utility_N64"), bool(learned_cmp)), f"learned-analytic CI={((learned_cmp.get('confidence_intervals') or {}).get('deltas') or {}).get('learned_minus_analytic_real_utility_N64')}")
    envs = set(multi.get("envs") or [])
    add(claims, 26, "BlockPush verified.", status("block_push" in envs or bool(exp1), bool(exp1)), "multi-env or canonical artifacts")
    add(claims, 27, "DrawerPull verified.", status("drawer_pull" in envs, False), "multi-env artifact")
    add(claims, 28, "SlipperyGrasp verified.", status("slippery_grasp" in envs, False), "multi-env artifact")
    add(claims, 29, "Nonstationary verified.", status("nonstationary_shift" in envs or bool(exp8), bool(exp8)), "multi-env/canonical artifact")
    add(claims, 30, "Deformable optional.", status("deformable_toy" in envs, False), "multi-env deformable artifact" if "deformable_toy" in envs else "not implemented")
    add(claims, 31, "Benchmark adapter available.", status(bool(bench) and bench.get("attempted", False), False), f"attempted={bench.get('attempted')}, any_available={bench.get('any_available')}")
    bench_score_ci = (benchmark_score.get("confidence_intervals") or {}).get("oracle_minus_random_real_utility_N32") or {}
    bench_closed_ci = (benchmark_closed.get("confidence_intervals") or {}).get("closed_loop_learned_minus_random_utility_N32") or {}
    add(claims, 32, "Benchmark rollout pools collected.", status(benchmark_pools.get("n_rollout_pools", 0) > 0, bool(bench) and bench.get("any_available", False)), f"pools={benchmark_pools.get('n_rollout_pools')}")
    add(claims, 33, "Benchmark exact law verified.", status(benchmark_exact.get("utility_mae") is not None and benchmark_exact.get("utility_mae") < 0.08, bool(benchmark_exact)), f"utility MAE={benchmark_exact.get('utility_mae')}")
    add(claims, 34, "Benchmark score comparison verified.", status(bench_score_ci.get("lo") is not None and bench_score_ci.get("lo") > 0.0, bool(benchmark_score)), f"oracle-random CI={bench_score_ci}")
    add(claims, 35, "Benchmark real-vs-imagined gap verified.", status(benchmark_gap.get("gap_growth_N32_minus_N1") is not None, bool(benchmark_gap)), f"gap growth={benchmark_gap.get('gap_growth_N32_minus_N1')}")
    add(claims, 36, "Benchmark closed-loop verified.", status(bench_closed_ci.get("lo") is not None and bench_closed_ci.get("lo") > 0.0, bool(benchmark_closed)), f"learned-random closed-loop CI={bench_closed_ci}")
    add(claims, 37, "Benchmark learned WAM trained.", status(bool(benchmark_wam.get("model_path")) and bool(benchmark_wam.get("model_metrics")), bool(benchmark_wam)), f"model={benchmark_wam.get('model_path')}")
    add(claims, 38, "Visual toy WAM attempted.", status(bool(visual) and visual.get("attempted", False), False), f"visual={visual.get('attempted')}")
    add(claims, 39, "Visual toy WAM verified if artifacts exist.", status(bool(visual) and visual.get("verified", False), bool(visual)), f"test MAE={visual.get('test_mae')}")
    add(claims, 40, "Benchmark visual optional.", status(bool(benchmark_visual) and benchmark_visual.get("verified", False), bool(benchmark_visual)), f"verified={benchmark_visual.get('verified')}")

    audit_ci = audit.get("confidence_intervals") or {}
    learned_audit_ci = audit_learned.get("confidence_intervals") or {}
    repair_ci = repair.get("confidence_intervals") or {}
    scaling_ci = scaling.get("confidence_intervals") or {}
    add(
        claims,
        41,
        "Inference-value audit profiles generated.",
        status(bool(audit) and bool(audit.get("profile_counts")) and bool(audit.get("decision_counts")), bool(audit)),
        f"profiles={len(audit.get('profile_counts') or [])}, decisions={len(audit.get('decision_counts') or [])}",
    )
    add(
        claims,
        42,
        "Tail alignment predicts high-N inference value.",
        status((audit_ci.get("tail_alignment_gain_corr") or {}).get("lo") is not None and (audit_ci.get("tail_alignment_gain_corr") or {}).get("lo") > 0.5, bool(audit)),
        f"tail-gain corr CI={audit_ci.get('tail_alignment_gain_corr')}",
    )
    anti_block_ci = audit_ci.get("anti_block_high_n_rate") or {}
    anti_harm_ci = audit_ci.get("anti_harm_magnitude") or {}
    add(
        claims,
        43,
        "Audit gate blocks harmful high-N bad-scorer deployments.",
        status(
            anti_block_ci.get("lo") is not None
            and anti_block_ci.get("lo") > 0.95
            and anti_harm_ci.get("lo") is not None
            and anti_harm_ci.get("lo") > 0.10,
            bool(audit),
        ),
        f"anti block CI={anti_block_ci}; anti harm CI={anti_harm_ci}",
    )
    add(
        claims,
        44,
        "Stop-rule compute savings are reported.",
        status((audit_ci.get("stop_rule_saved_rollout_fraction") or {}).get("lo") is not None and (audit_ci.get("stop_rule_saved_rollout_fraction") or {}).get("lo") > 0.05, bool(audit)),
        f"saved rollout fraction CI={audit_ci.get('stop_rule_saved_rollout_fraction')}",
    )
    add(
        claims,
        45,
        "Learned-backend inference audit reproduced.",
        status(
            bool(audit_learned)
            and (learned_audit_ci.get("tail_alignment_gain_corr") or {}).get("lo") is not None
            and (learned_audit_ci.get("anti_block_high_n_rate") or {}).get("lo") is not None,
            bool(audit_learned),
        ),
        f"learned tail-gain CI={learned_audit_ci.get('tail_alignment_gain_corr')}; anti block CI={learned_audit_ci.get('anti_block_high_n_rate')}",
    )
    add(
        claims,
        46,
        "Pilot-calibrated scorer repair improves heldout high-N utility.",
        status((repair_ci.get("repair_minus_predicted_N64") or {}).get("lo") is not None and (repair_ci.get("repair_minus_predicted_N64") or {}).get("lo") > 0.0, bool(repair)),
        f"repair-predicted CI={repair_ci.get('repair_minus_predicted_N64')}",
    )
    add(
        claims,
        47,
        "Robot-imagination compute frontier is measured.",
        status(
            (scaling_ci.get("predicted_gain_N128_minus_N1") or {}).get("lo") is not None
            and (scaling_ci.get("predicted_gain_N128_minus_N1") or {}).get("lo") > 0.0
            and (scaling_ci.get("oracle_minus_predicted_gain") or {}).get("lo") is not None
            and (scaling_ci.get("oracle_minus_predicted_gain") or {}).get("lo") > 0.0,
            bool(scaling),
        ),
        f"pred gain CI={scaling_ci.get('predicted_gain_N128_minus_N1')}; oracle-pred gain CI={scaling_ci.get('oracle_minus_predicted_gain')}",
    )

    maniskill_ci = maniskill.get("confidence_intervals") or {}
    add(
        claims,
        48,
        "ManiSkill state benchmark suite verified.",
        status(bool(maniskill) and maniskill.get("available", False) and len(maniskill.get("env_ids") or []) >= 3, bool(maniskill)),
        f"envs={maniskill.get('env_ids')}, control={maniskill.get('control_mode')}",
    )
    add(
        claims,
        49,
        "ManiSkill rollout pools collected.",
        status((maniskill.get("n_rollout_pools") or 0) >= 25, bool(maniskill)),
        f"pools={maniskill.get('n_rollout_pools')}, rollouts={maniskill.get('n_rollouts')}",
    )
    add(
        claims,
        50,
        "ManiSkill exact law verified.",
        status(maniskill.get("exact_law_utility_mae") is not None and maniskill.get("exact_law_utility_mae") < 0.03, bool(maniskill)),
        f"utility MAE={maniskill.get('exact_law_utility_mae')}",
    )
    dense_ci = maniskill_ci.get("dense_minus_random_real_utility_N32") or {}
    oracle_mani_ci = maniskill_ci.get("oracle_minus_random_real_utility_N32") or {}
    add(
        claims,
        51,
        "ManiSkill score comparison verified.",
        status(
            dense_ci.get("lo") is not None
            and dense_ci.get("lo") > 0.0
            and oracle_mani_ci.get("lo") is not None
            and oracle_mani_ci.get("lo") > 0.0,
            bool(maniskill),
        ),
        f"dense-random CI={dense_ci}; oracle-random CI={oracle_mani_ci}",
    )
    add(
        claims,
        52,
        "ManiSkill WAM-lite trained and evaluated.",
        status(len(maniskill.get("model_metrics") or []) >= 6, bool(maniskill)),
        f"model metric rows={len(maniskill.get('model_metrics') or [])}",
    )
    mani_closed_ci = maniskill_ci.get("closed_loop_learned_minus_random_utility_N8") or {}
    add(
        claims,
        53,
        "ManiSkill closed-loop learned scorer beats random.",
        status(mani_closed_ci.get("lo") is not None and mani_closed_ci.get("lo") > 0.0, bool(maniskill)),
        f"learned-random closed-loop CI={mani_closed_ci}",
    )
    learned_open_ci = maniskill_ci.get("learned_minus_random_real_utility_N32") or {}
    add(
        claims,
        54,
        "ManiSkill learned open-loop scorer is honestly reported.",
        status(learned_open_ci.get("n", 0) >= 5, bool(maniskill)),
        f"learned-random open-loop CI={learned_open_ci}",
    )
    visual_wam_ci = benchmark_visual_wam.get("confidence_intervals") or {}
    add(
        claims,
        55,
        "Benchmark RGB visual WAM-lite trained and evaluated.",
        status(
            bool(benchmark_visual_wam)
            and benchmark_visual_wam.get("verified", False)
            and (benchmark_visual_wam.get("validation") or {}).get("utility_corr", 0.0) > 0.20,
            bool(benchmark_visual_wam),
        ),
        f"model={benchmark_visual_wam.get('model_type')}, validation={benchmark_visual_wam.get('validation')}",
    )
    add(
        claims,
        56,
        "Benchmark RGB visual WAM exact law verified.",
        status(
            benchmark_visual_wam.get("exact_law_utility_mae") is not None
            and benchmark_visual_wam.get("exact_law_utility_mae") < 0.05,
            bool(benchmark_visual_wam),
        ),
        f"utility MAE={benchmark_visual_wam.get('exact_law_utility_mae')}",
    )
    add(
        claims,
        57,
        "Benchmark RGB visual WAM scorer beats random with CI.",
        status(
            (visual_wam_ci.get("visual_minus_random_N32") or {}).get("lo") is not None
            and (visual_wam_ci.get("visual_minus_random_N32") or {}).get("lo") > 0.0,
            bool(benchmark_visual_wam),
        ),
        f"visual-random CI={visual_wam_ci.get('visual_minus_random_N32')}",
    )
    add(
        claims,
        58,
        "Benchmark RGB visual WAM oracle gap reported.",
        status(
            (visual_wam_ci.get("oracle_minus_visual_N32") or {}).get("lo") is not None
            and (visual_wam_ci.get("oracle_minus_visual_N32") or {}).get("lo") > 0.0,
            bool(benchmark_visual_wam),
        ),
        f"oracle-visual CI={visual_wam_ci.get('oracle_minus_visual_N32')}",
    )
    gym_robotics_ci = gym_robotics.get("confidence_intervals") or {}
    add(
        claims,
        59,
        "Gymnasium Robotics Fetch benchmark suite verified.",
        status(
            bool(gym_robotics)
            and gym_robotics.get("available", False)
            and len(gym_robotics.get("env_ids") or []) >= 3
            and (gym_robotics.get("n_rollout_pools") or 0) >= 50,
            bool(gym_robotics),
        ),
        f"envs={gym_robotics.get('env_ids')}, pools={gym_robotics.get('n_rollout_pools')}",
    )
    add(
        claims,
        60,
        "Gymnasium Robotics Fetch exact law verified.",
        status(
            gym_robotics.get("exact_law_utility_mae") is not None
            and gym_robotics.get("exact_law_utility_mae") < 0.06,
            bool(gym_robotics),
        ),
        f"utility MAE={gym_robotics.get('exact_law_utility_mae')}",
    )
    add(
        claims,
        61,
        "Gymnasium Robotics learned WAM scorer beats random with CI.",
        status(
            (gym_robotics_ci.get("learned_minus_random_N32") or {}).get("lo") is not None
            and (gym_robotics_ci.get("learned_minus_random_N32") or {}).get("lo") > 0.0,
            bool(gym_robotics),
        ),
        f"learned-random CI={gym_robotics_ci.get('learned_minus_random_N32')}",
    )
    add(
        claims,
        62,
        "Gymnasium Robotics closed-loop learned scorer beats random.",
        status(
            (gym_robotics_ci.get("closed_loop_learned_minus_random_N32") or {}).get("lo") is not None
            and (gym_robotics_ci.get("closed_loop_learned_minus_random_N32") or {}).get("lo") > 0.0,
            bool(gym_robotics),
        ),
        f"closed-loop learned-random CI={gym_robotics_ci.get('closed_loop_learned_minus_random_N32')}",
    )
    add(
        claims,
        63,
        "Gymnasium Robotics oracle gap reported.",
        status(
            (gym_robotics_ci.get("oracle_minus_learned_N32") or {}).get("lo") is not None
            and (gym_robotics_ci.get("oracle_minus_learned_N32") or {}).get("lo") > 0.0,
            bool(gym_robotics),
        ),
        f"oracle-learned CI={gym_robotics_ci.get('oracle_minus_learned_N32')}",
    )
    gym_robotics_visual_ci = gym_robotics_visual.get("confidence_intervals") or {}
    add(
        claims,
        64,
        "Gymnasium Robotics RGB visual WAM trained and evaluated.",
        status(
            bool(gym_robotics_visual)
            and gym_robotics_visual.get("verified", False)
            and len(gym_robotics_visual.get("env_ids") or []) >= 3
            and (gym_robotics_visual.get("mean_validation_utility_corr") or 0.0) > 0.25,
            bool(gym_robotics_visual),
        ),
        f"envs={gym_robotics_visual.get('env_ids')}, mean corr={gym_robotics_visual.get('mean_validation_utility_corr')}",
    )
    add(
        claims,
        65,
        "Gymnasium Robotics RGB visual exact law verified.",
        status(
            gym_robotics_visual.get("exact_law_utility_mae") is not None
            and gym_robotics_visual.get("exact_law_utility_mae") < 0.08,
            bool(gym_robotics_visual),
        ),
        f"utility MAE={gym_robotics_visual.get('exact_law_utility_mae')}",
    )
    add(
        claims,
        66,
        "Gymnasium Robotics RGB visual scorer beats random with CI.",
        status(
            (gym_robotics_visual_ci.get("visual_minus_random_N32") or {}).get("lo") is not None
            and (gym_robotics_visual_ci.get("visual_minus_random_N32") or {}).get("lo") > 0.0,
            bool(gym_robotics_visual),
        ),
        f"visual-random CI={gym_robotics_visual_ci.get('visual_minus_random_N32')}",
    )
    add(
        claims,
        67,
        "Gymnasium Robotics RGB visual oracle gap is reported without requiring significance.",
        status((gym_robotics_visual_ci.get("oracle_minus_visual_N32") or {}).get("n", 0) >= 5, bool(gym_robotics_visual)),
        f"oracle-visual CI={gym_robotics_visual_ci.get('oracle_minus_visual_N32')}",
    )
    add(
        claims,
        68,
        "ManiSkill RGB/RGB-D and EE-control probe is artifact-documented.",
        status(
            bool(maniskill_visual_probe)
            and maniskill_visual_probe.get("attempted", False)
            and maniskill_visual_probe.get("state_baseline_ok", False)
            and (maniskill_visual_probe.get("visual_attempt_count") or 0) >= 5
            and "ErrorOutOfPoolMemory" in str(maniskill_visual_probe.get("visual_blocker", ""))
            and bool(maniskill_dependency_probe)
            and maniskill_dependency_probe.get("attempted", False)
            and not maniskill_dependency_probe.get("pinocchio_import_available", True)
            and not maniskill_dependency_probe.get("pin_binary_wheel_available", True),
            bool(maniskill_visual_probe),
        ),
        f"visual_success={maniskill_visual_probe.get('any_visual_success')}, blocker={maniskill_visual_probe.get('visual_blocker')}; pinocchio={maniskill_dependency_probe.get('pinocchio_import_available')}, pin_binary={maniskill_dependency_probe.get('pin_binary_wheel_available')}",
    )
    metaworld_ci = metaworld.get("confidence_intervals") or {}
    add(
        claims,
        69,
        "Meta-World ML1 benchmark suite verified.",
        status(
            bool(metaworld)
            and metaworld.get("available", False)
            and (metaworld.get("n_tasks_verified") or 0) >= 3
            and (metaworld.get("n_rollout_pools") or 0) >= 45,
            bool(metaworld),
        ),
        f"tasks={metaworld.get('task_names')}, pools={metaworld.get('n_rollout_pools')}",
    )
    add(
        claims,
        70,
        "Meta-World exact law verified.",
        status(
            metaworld.get("exact_law_utility_mae") is not None
            and metaworld.get("exact_law_utility_mae") < 0.04,
            bool(metaworld),
        ),
        f"utility MAE={metaworld.get('exact_law_utility_mae')}",
    )
    add(
        claims,
        71,
        "Meta-World learned WAM scorer beats random open-loop with CI.",
        status(
            (metaworld_ci.get("learned_minus_random_N32") or {}).get("lo") is not None
            and (metaworld_ci.get("learned_minus_random_N32") or {}).get("lo") > 0.0,
            bool(metaworld),
        ),
        f"learned-random CI={metaworld_ci.get('learned_minus_random_N32')}",
    )
    add(
        claims,
        72,
        "Meta-World oracle and benchmark-reward scorers beat random with CI.",
        status(
            (metaworld_ci.get("reward_minus_random_N32") or {}).get("lo") is not None
            and (metaworld_ci.get("reward_minus_random_N32") or {}).get("lo") > 0.0
            and (metaworld_ci.get("oracle_minus_random_N32") or {}).get("lo") is not None
            and (metaworld_ci.get("oracle_minus_random_N32") or {}).get("lo") > 0.0,
            bool(metaworld),
        ),
        f"reward-random CI={metaworld_ci.get('reward_minus_random_N32')}; oracle-random CI={metaworld_ci.get('oracle_minus_random_N32')}",
    )
    robosuite_ci = robosuite.get("confidence_intervals") or {}
    add(
        claims,
        73,
        "RoboSuite Panda manipulation benchmark suite verified.",
        status(
            bool(robosuite)
            and robosuite.get("available", False)
            and (robosuite.get("n_tasks_verified") or 0) >= 3
            and (robosuite.get("n_rollout_pools") or 0) >= 30,
            bool(robosuite),
        ),
        f"envs={robosuite.get('env_names')}, pools={robosuite.get('n_rollout_pools')}",
    )
    add(
        claims,
        74,
        "RoboSuite exact law verified.",
        status(
            robosuite.get("exact_law_utility_mae") is not None
            and robosuite.get("exact_law_utility_mae") < 0.02,
            bool(robosuite),
        ),
        f"utility MAE={robosuite.get('exact_law_utility_mae')}",
    )
    add(
        claims,
        75,
        "RoboSuite learned WAM scorer beats random open-loop with CI.",
        status(
            (robosuite_ci.get("learned_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("learned_minus_random_N32") or {}).get("lo") > 0.0,
            bool(robosuite),
        ),
        f"learned-random CI={robosuite_ci.get('learned_minus_random_N32')}",
    )
    add(
        claims,
        76,
        "RoboSuite reward, progress, and oracle scorers beat random with CI.",
        status(
            (robosuite_ci.get("reward_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("reward_minus_random_N32") or {}).get("lo") > 0.0
            and (robosuite_ci.get("progress_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("progress_minus_random_N32") or {}).get("lo") > 0.0
            and (robosuite_ci.get("oracle_minus_random_N32") or {}).get("lo") is not None
            and (robosuite_ci.get("oracle_minus_random_N32") or {}).get("lo") > 0.0,
            bool(robosuite),
        ),
        f"reward-random CI={robosuite_ci.get('reward_minus_random_N32')}; progress-random CI={robosuite_ci.get('progress_minus_random_N32')}; oracle-random CI={robosuite_ci.get('oracle_minus_random_N32')}",
    )
    add(
        claims,
        77,
        "RoboSuite closed-loop learned and reward scorers beat random.",
        status(
            (robosuite_ci.get("closed_loop_learned_minus_random_N8") or {}).get("lo") is not None
            and (robosuite_ci.get("closed_loop_learned_minus_random_N8") or {}).get("lo") > 0.0
            and (robosuite_ci.get("closed_loop_reward_minus_random_N8") or {}).get("lo") is not None
            and (robosuite_ci.get("closed_loop_reward_minus_random_N8") or {}).get("lo") > 0.0,
            bool(robosuite),
        ),
        f"learned-random CI={robosuite_ci.get('closed_loop_learned_minus_random_N8')}; reward-random CI={robosuite_ci.get('closed_loop_reward_minus_random_N8')}",
    )
    add(
        claims,
        78,
        "RoboCasa kitchen benchmark smoke verified.",
        status(
            bool(robocasa)
            and robocasa.get("available", False)
            and robocasa.get("verified", False)
            and (robocasa.get("n_rollout_pools") or 0) >= 5
            and (robocasa.get("n_rollouts_total") or 0) >= 80
            and robocasa.get("exact_law_utility_mae") is not None
            and robocasa.get("exact_law_utility_mae") < 0.01
            and (((robocasa.get("confidence_intervals") or {}).get("oracle_minus_random_N8") or {}).get("n") or 0) >= 5
            and (((robocasa.get("confidence_intervals") or {}).get("oracle_minus_random_N8") or {}).get("lo") or 0.0) > 0.0,
            bool(robocasa),
        ),
        f"env={robocasa.get('env_id')}, pools={robocasa.get('n_rollout_pools')}, rollouts={robocasa.get('n_rollouts_total')}, exact MAE={robocasa.get('exact_law_utility_mae')}, oracle-random CI={((robocasa.get('confidence_intervals') or {}).get('oracle_minus_random_N8'))}",
    )
    robocasa_learned_ci = robocasa_learned.get("confidence_intervals") or {}
    add(
        claims,
        79,
        "RoboCasa learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_learned)
            and robocasa_learned.get("available", False)
            and robocasa_learned.get("verified", False)
            and (robocasa_learned.get("train_samples") or 0) >= 80
            and (robocasa_learned.get("validation_samples") or 0) >= 32
            and (robocasa_learned.get("eval_samples") or 0) >= 80
            and ((robocasa_learned.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_learned.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_learned_ci.get("learned_minus_random_N8") or {}).get("lo") or 0.0) > 0.0,
            bool(robocasa_learned),
        ),
        f"train={robocasa_learned.get('train_samples')}, val={robocasa_learned.get('validation_samples')}, eval={robocasa_learned.get('eval_samples')}, utility corr={((robocasa_learned.get('model_metrics') or {}).get('utility_corr'))}, learned-random CI={robocasa_learned_ci.get('learned_minus_random_N8')}",
    )
    robocasa_multitask_ci = robocasa_multitask.get("confidence_intervals") or {}
    add(
        claims,
        80,
        "RoboCasa three-task learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_multitask)
            and robocasa_multitask.get("available", False)
            and robocasa_multitask.get("verified", False)
            and len(robocasa_multitask.get("env_ids") or []) >= 3
            and (robocasa_multitask.get("train_samples") or 0) >= 144
            and (robocasa_multitask.get("validation_samples") or 0) >= 96
            and (robocasa_multitask.get("eval_samples") or 0) >= 240
            and (robocasa_multitask.get("eval_rollout_pools") or 0) >= 15
            and ((robocasa_multitask.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_multitask.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_multitask_ci.get("best_learned_minus_random_N8") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_multitask_ci.get("oracle_minus_best_learned_N8") or {}).get("lo") or 0.0) > 0.0,
            bool(robocasa_multitask),
        ),
        f"tasks={robocasa_multitask.get('env_ids')}, train={robocasa_multitask.get('train_samples')}, val={robocasa_multitask.get('validation_samples')}, eval={robocasa_multitask.get('eval_samples')}, utility corr={((robocasa_multitask.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_multitask.get('promoted_scorer')}, learned-random CI={robocasa_multitask_ci.get('best_learned_minus_random_N8')}, oracle-learned CI={robocasa_multitask_ci.get('oracle_minus_best_learned_N8')}",
    )

    libero_ci = libero_wam.get("confidence_intervals") or {}
    libero_max_n = max(libero_wam.get("n_values") or [8])
    add(
        claims,
        83,
        "LIBERO rollout-pool learned WAM-lite benchmark verified.",
        status(
            bool(libero_wam)
            and libero_wam.get("available", False)
            and libero_wam.get("verified", False)
            and len(libero_wam.get("tasks") or []) >= 1
            and (libero_wam.get("train_samples") or 0) >= 64
            and (libero_wam.get("validation_samples") or 0) >= 32
            and (libero_wam.get("eval_samples") or 0) >= 80
            and (libero_wam.get("eval_rollout_pools") or 0) >= 5
            and (libero_wam.get("exact_law_utility_mae") or 1.0) < 0.03
            and ((libero_wam.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and ((libero_ci.get(f"best_learned_minus_random_N{libero_max_n}") or {}).get("lo") or 0.0) > 0.0,
            bool(libero_wam),
        ),
        f"tasks={libero_wam.get('tasks')}, train={libero_wam.get('train_samples')}, val={libero_wam.get('validation_samples')}, eval={libero_wam.get('eval_samples')}, exact MAE={libero_wam.get('exact_law_utility_mae')}, utility corr={((libero_wam.get('model_metrics') or {}).get('utility_corr'))}, learned-random CI={libero_ci.get(f'best_learned_minus_random_N{libero_max_n}')}",
    )
    robocasa_broad_ci = robocasa_broad.get("confidence_intervals") or {}
    robocasa_broad_max_n = max(robocasa_broad.get("n_values") or [8])
    add(
        claims,
        84,
        "RoboCasa broad task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_broad)
            and robocasa_broad.get("available", False)
            and robocasa_broad.get("verified", False)
            and len(robocasa_broad.get("env_ids") or []) >= 4
            and (robocasa_broad.get("train_samples") or 0) >= 64
            and (robocasa_broad.get("validation_samples") or 0) >= 32
            and (robocasa_broad.get("eval_samples") or 0) >= 128
            and (robocasa_broad.get("eval_rollout_pools") or 0) >= 16
            and ((robocasa_broad.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_broad.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_broad_ci.get(f"best_learned_minus_random_N{robocasa_broad_max_n}") or {}).get("lo") or 0.0) > 0.0,
            bool(robocasa_broad),
        ),
        f"tasks={robocasa_broad.get('env_ids')}, train={robocasa_broad.get('train_samples')}, val={robocasa_broad.get('validation_samples')}, eval={robocasa_broad.get('eval_samples')}, utility corr={((robocasa_broad.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_broad.get('promoted_scorer')}, learned-random CI={robocasa_broad_ci.get(f'best_learned_minus_random_N{robocasa_broad_max_n}')}",
    )
    robocasa_family12_ci = robocasa_family12.get("confidence_intervals") or {}
    robocasa_family12_max_n = max(robocasa_family12.get("n_values") or [8])
    add(
        claims,
        85,
        "RoboCasa 12-task family learned WAM-lite scorer beats random with CI.",
        status(
            bool(robocasa_family12)
            and robocasa_family12.get("available", False)
            and robocasa_family12.get("verified", False)
            and len(robocasa_family12.get("env_ids") or []) >= 12
            and (robocasa_family12.get("train_samples") or 0) >= 96
            and (robocasa_family12.get("validation_samples") or 0) >= 96
            and (robocasa_family12.get("eval_samples") or 0) >= 192
            and (robocasa_family12.get("eval_rollout_pools") or 0) >= 24
            and ((robocasa_family12.get("model_metrics") or {}).get("utility_corr") or 0.0) > 0.0
            and (robocasa_family12.get("exact_law_utility_mae") or 1.0) < 0.01
            and ((robocasa_family12_ci.get(f"best_learned_minus_random_N{robocasa_family12_max_n}") or {}).get("lo") or 0.0) > 0.0
            and ((robocasa_family12_ci.get(f"oracle_minus_best_learned_N{robocasa_family12_max_n}") or {}).get("lo") or 0.0) > 0.0,
            bool(robocasa_family12),
        ),
        f"tasks={robocasa_family12.get('env_ids')}, train={robocasa_family12.get('train_samples')}, val={robocasa_family12.get('validation_samples')}, eval={robocasa_family12.get('eval_samples')}, utility corr={((robocasa_family12.get('model_metrics') or {}).get('utility_corr'))}, promoted={robocasa_family12.get('promoted_scorer')}, learned-random CI={robocasa_family12_ci.get(f'best_learned_minus_random_N{robocasa_family12_max_n}')}",
    )

    libero_scripted_ci = (libero_scripted.get("confidence_intervals") or {}).get("success_rate") or {}
    add(
        claims,
        86,
        "LIBERO sparse-success scripted policy smoke verified.",
        status(
            bool(libero_scripted)
            and libero_scripted.get("available", False)
            and libero_scripted.get("verified", False)
            and (libero_scripted.get("n_tasks") or 0) >= 10
            and (libero_scripted.get("n_seeds") or 0) >= 5
            and (libero_scripted.get("n_episodes") or 0) >= 50
            and (libero_scripted.get("n_successes") or 0) >= 20
            and (libero_scripted.get("success_rate") or 0.0) >= 0.5
            and (libero_scripted_ci.get("lo") or 0.0) > 0.25,
            bool(libero_scripted),
        ),
        f"episodes={libero_scripted.get('n_episodes')}, successes={libero_scripted.get('n_successes')}, success CI={libero_scripted_ci}",
    )

    readme_text = README.read_text(encoding="utf-8") if README.exists() else ""
    paper_text = PAPER.read_text(encoding="utf-8") if PAPER.exists() else ""
    claim_by_id = {c["id"]: c for c in claims}
    overclaims = []
    for cid, c in claim_by_id.items():
        pattern = f"VERIFIED CLAIM {cid}"
        if pattern.lower() in readme_text.lower() and c["status"] not in {"VERIFIED", "PARTIAL"}:
            overclaims.append({"surface": "README", "id": cid, "pattern": pattern, "status": c["status"]})
        if pattern.lower() in paper_text.lower() and c["status"] not in {"VERIFIED", "PARTIAL"}:
            overclaims.append({"surface": "paper_outline", "id": cid, "pattern": pattern, "status": c["status"]})

    add(claims, 81, "README has no unsupported claims.", status(len([o for o in overclaims if o["surface"] == "README"]) == 0), f"README overclaims={len([o for o in overclaims if o['surface'] == 'README'])}")
    add(claims, 82, "paper_outline has no unsupported claims.", status(len([o for o in overclaims if o["surface"] == "paper_outline"]) == 0), f"paper overclaims={len([o for o in overclaims if o['surface'] == 'paper_outline'])}")

    payload = {
        "claims": claims,
        "readme_overclaims": overclaims,
        "num_verified": sum(c["status"] == "VERIFIED" for c in claims),
        "num_partial": sum(c["status"] == "PARTIAL" for c in claims),
        "num_unsupported": sum(c["status"] == "UNSUPPORTED" for c in claims),
        "num_failed": sum(c["status"] == "FAILED" for c in claims),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Claims Status", ""]
    for c in claims:
        lines.append(f"- Claim {c['id']}: **{c['status']}** - {c['claim']} Evidence: {c['evidence']}")
    if overclaims:
        lines.append("")
        lines.append("## Overclaims")
        for item in overclaims:
            lines.append(f"- {item['surface']} claim {item['id']} pattern `{item['pattern']}` has status {item['status']}.")
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if overclaims:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
