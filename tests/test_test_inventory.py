from wam_inference_value.test_inventory import audit_pytest_collection, parse_pytest_collect_stdout


def sample_stdout(count: int = 94) -> str:
    nodeids = [
        "tests/test_abstract_claim_support.py::test_abstract_claim_support_accepts_exact_supported_headlines",
        "tests/test_claim_scope_audit.py::test_claim_scope_audit_accepts_scoped_benchmark_and_smoke_claims",
        "tests/test_command_result_consistency.py::test_command_result_consistency_accepts_current_report",
        "tests/test_model_artifact_integrity.py::test_model_artifact_integrity_accepts_loadable_npz",
        "tests/test_publication_scope.py::test_publication_scope_accepts_guarded_risk_mentions",
        "tests/test_script_contracts.py::test_ordered_subsequence_requires_order",
        "tests/test_theorem_binary.py::test_n1_equals_p",
    ]
    nodeids.extend(f"tests/test_extra.py::test_case_{idx}" for idx in range(count - len(nodeids)))
    return "\n".join(nodeids) + f"\n\n{count} tests collected in 1.23s\n"


def test_parse_pytest_collect_stdout_counts_nodeids_and_trailer():
    parsed = parse_pytest_collect_stdout(sample_stdout(95))

    assert parsed["n_nodeids"] == 95
    assert parsed["trailer_count"] == 95


def test_audit_pytest_collection_accepts_sane_inventory():
    payload = audit_pytest_collection(stdout=sample_stdout(95), returncode=0)

    assert payload["verified"] is True
    assert payload["n_tests"] == 95
    assert payload["n_issues"] == 0


def test_audit_pytest_collection_rejects_duplicate_and_trailer_mismatch():
    stdout = sample_stdout(95)
    first = stdout.splitlines()[0]
    bad_stdout = stdout.replace("95 tests collected", "97 tests collected") + first + "\n"

    payload = audit_pytest_collection(stdout=bad_stdout, returncode=0)

    failures = {check["name"] for check in payload["issues"]}
    assert "nodeids_unique" in failures
    assert "trailer_count_matches_nodeids" in failures
