from wam_inference_value.narrative_consistency import NarrativeCheck, add_contains, markdown_series


def test_markdown_series_uses_final_and():
    assert markdown_series(["`0.1`"]) == "`0.1`"
    assert markdown_series(["`0.1`", "`0.2`"]) == "`0.1`, and `0.2`"
    assert markdown_series(["`0.1`", "`0.2`", "`0.3`"]) == "`0.1`, `0.2`, and `0.3`"


def test_add_contains_records_missing_snippet():
    checks: list[NarrativeCheck] = []

    add_contains(checks, "README", "actual text", "claim_number", "expected metric")

    assert checks == [
        NarrativeCheck(
            surface="README",
            name="claim_number",
            ok=False,
            expected="expected metric",
            detail="missing expected snippet",
        )
    ]


def test_add_contains_records_found_snippet():
    checks: list[NarrativeCheck] = []

    add_contains(checks, "README", "expected metric is here", "claim_number", "expected metric")

    assert checks[0].ok is True
    assert checks[0].detail == "found"
