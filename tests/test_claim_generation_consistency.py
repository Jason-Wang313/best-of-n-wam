import json
import sys
from pathlib import Path

from wam_inference_value.claim_generation_consistency import audit_claim_generation_consistency


def write_claim_files(results: Path, *, evidence: str = "value=1") -> None:
    payload = {
        "claims": [{"id": 1, "claim": "Stable.", "status": "VERIFIED", "evidence": evidence}],
        "overclaims": [],
        "num_verified": 1,
        "num_partial": 0,
        "num_unsupported": 0,
        "num_failed": 0,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "claims_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (results / "claims_status.md").write_text(f"# Claims Status\n\n- Claim 1: **VERIFIED** - Stable. Evidence: {evidence}\n", encoding="utf-8")


def write_generator(path: Path, *, mutate: bool) -> None:
    evidence = "value=2" if mutate else "value=1"
    counter_block = (
        "counter = results / 'counter.txt'\n"
        "count = int(counter.read_text(encoding='utf-8')) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count), encoding='utf-8')\n"
        "evidence = 'value=' + str(count)\n"
        if mutate
        else f"evidence = '{evidence}'\n"
    )
    path.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "results = Path(os.environ['WAM_RESULTS_DIR'])\n"
        f"{counter_block}"
        "payload = {'claims': [{'id': 1, 'claim': 'Stable.', 'status': 'VERIFIED', 'evidence': evidence}], 'overclaims': [], 'num_verified': 1, 'num_partial': 0, 'num_unsupported': 0, 'num_failed': 0}\n"
        "md = '# Claims Status\\n\\n- Claim 1: **VERIFIED** - Stable. Evidence: ' + payload['claims'][0]['evidence'] + '\\n'\n"
        "(results / 'claims_status.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')\n"
        "(results / 'claims_status.md').write_text(md, encoding='utf-8')\n"
        "print(md)\n",
        encoding="utf-8",
    )


def test_claim_generation_consistency_accepts_byte_stable_generator(tmp_path: Path):
    results = tmp_path / "results"
    generator = tmp_path / "stable_generator.py"
    write_claim_files(results)
    write_generator(generator, mutate=False)

    payload = audit_claim_generation_consistency(tmp_path, results, command=[sys.executable, str(generator)])

    assert payload["verified"] is True
    assert payload["n_issues"] == 0


def test_claim_generation_consistency_rejects_mutating_generator(tmp_path: Path):
    results = tmp_path / "results"
    generator = tmp_path / "mutating_generator.py"
    write_claim_files(results)
    write_generator(generator, mutate=True)

    payload = audit_claim_generation_consistency(tmp_path, results, command=[sys.executable, str(generator)])

    failures = {check["name"] for check in payload["issues"]}
    assert "claims_json_byte_stable" in failures
    assert "claims_md_byte_stable" in failures
