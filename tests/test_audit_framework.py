from __future__ import annotations

import numpy as np

from wam_inference_value.audit import audit_score_distribution, classify_profile, inference_value_profile, tail_alignment


def test_aligned_distribution_is_helpful_and_sample_more():
    utilities = np.linspace(0.0, 1.0, 200)
    scores = utilities.copy()
    audit = audit_score_distribution(scores, utilities, [1, 2, 4, 8, 16, 32, 64])

    assert audit["profile_class"] == "helpful"
    assert audit["alignment"]["alignment_status"] == "aligned"
    assert audit["decision"]["action"] in {"sample_more", "stop_early"}
    assert audit["profile"]["gain_last_minus_first"] > 0.35


def test_anti_aligned_distribution_blocks_high_n():
    utilities = np.linspace(0.0, 1.0, 200)
    scores = -utilities
    audit = audit_score_distribution(scores, utilities, [1, 2, 4, 8, 16, 32, 64])

    assert audit["profile_class"] == "harmful"
    assert audit["alignment"]["alignment_status"] == "anti_aligned"
    assert audit["decision"]["action"] == "block_high_n"
    assert audit["profile"]["gain_last_minus_first"] < -0.35


def test_random_scores_have_weak_tail_alignment():
    rng = np.random.default_rng(9)
    scores = rng.normal(size=400)
    utilities = rng.normal(size=400)
    alignment = tail_alignment(scores, utilities)

    assert alignment["alignment_status"] in {"weak", "aligned", "anti_aligned"}
    assert abs(alignment["score_real_rank_corr"]) < 0.18


def test_profile_classifies_saturation():
    scores = np.arange(100, dtype=float)
    utilities = np.ones(100)
    profile = inference_value_profile(scores, utilities, [1, 2, 4, 8])

    assert classify_profile(profile) == "saturating"
    assert profile["gain_last_minus_first"] == 0.0
