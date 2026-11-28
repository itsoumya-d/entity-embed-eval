"""Unit and regression tests for rubric-driven verify-and-repair loop."""

import json
import math
from pathlib import Path
import tempfile
import pytest

from entity_embed_eval import (
    load_rubric,
    verify_output,
    repair_output,
    VerificationScore,
    RepairResult,
)


def _make_unit_vector(dim: int = 64) -> list[float]:
    raw = [1.0] * dim
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def test_rubric_json_valid_spec():
    rubric = load_rubric()
    assert rubric.get("threshold") == 80
    checks = rubric.get("checks", [])
    assert len(checks) == 5
    total_weight = sum(c["weight"] for c in checks)
    assert total_weight == 100


def test_verify_clean_output():
    payload = {
        "id": "e_clean_101",
        "text": "Automated data pipeline transformation with distributed stream ingestion",
        "entity_type": "post",
        "tags": ["data", "streaming"],
        "dim": 64,
        "vector": _make_unit_vector(64),
        "similarity": 0.85,
    }
    score: VerificationScore = verify_output(payload)
    assert score.score == 100
    assert score.passed is True
    assert score.threshold == 80
    for c in score.checks:
        assert c.passed is True


def test_verify_broken_mechanical_fixture():
    # Broken: text has extra whitespace and control chars (-20),
    # entity_type is missing (-20),
    # tags is None (-20),
    # vector is not unit normalized (norm = 64.0) (-20),
    # score is 1.75 out of bounds (-20)
    payload = {
        "id": "e_broken_mech_01",
        "text": "  Unformatted text   with extra   spaces \x00 ",
        "dim": 64,
        "vector": [1.0] * 64,
        "similarity": 1.75,
    }
    score: VerificationScore = verify_output(payload)
    assert score.score < 80
    assert score.passed is False
    assert score.check_map["required_envelope"] is False
    assert score.check_map["text_formatting"] is False
    assert score.check_map["vector_unit_norm"] is False
    assert score.check_map["score_bounds"] is False


def test_repair_mechanical_fixture_passes():
    payload = {
        "id": "e_broken_mech_02",
        "text": "   Pipeline data ingestion staging table   \n",
        "dim": 64,
        "vector": [2.5] * 64,  # unnormalized
        "score": 2.4,          # out of bounds
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = Path(tmpdir) / "audit_ledger.jsonl"
        res: RepairResult = repair_output(payload, ledger_path=str(ledger_file), max_rounds=3)

        assert res.passed is True
        assert res.final_score >= 80
        assert res.rounds >= 1
        assert "entity_type" in res.repaired_fields
        assert "tags" in res.repaired_fields
        assert "text" in res.repaired_fields
        assert "vector" in res.repaired_fields
        assert "score" in res.repaired_fields

        # Check repaired payload integrity
        repaired = res.repaired_output
        assert repaired["entity_type"] == "post"
        assert repaired["tags"] == []
        assert repaired["text"] == "Pipeline data ingestion staging table"
        assert repaired["score"] == 1.0
        norm = math.sqrt(sum(x * x for x in repaired["vector"]))
        assert abs(norm - 1.0) <= 0.005

        # Check ledger existence and content
        assert ledger_file.exists()
        lines = ledger_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == res.rounds
        first_entry = json.loads(lines[0])
        assert first_entry["passed"] is True
        assert first_entry["score"] >= 80


def test_repair_non_mechanical_fault_reported():
    # Non-mechanical: empty text and zero vector cannot be fabricated or guessed
    payload = {
        "id": "e_non_mech_01",
        "text": "",
        "entity_type": "post",
        "tags": ["core"],
        "dim": 64,
        "vector": [0.0] * 64,
        "similarity": 0.5,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = Path(tmpdir) / "audit_ledger.jsonl"
        res: RepairResult = repair_output(payload, ledger_path=str(ledger_file), max_rounds=3)

        assert res.passed is False
        assert res.final_score < 80
        # Check non-mechanical faults are reported for human review without guessing
        assert len(res.unresolved_issues) > 0
        assert any("empty text" in issue.lower() or "missing" in issue.lower() for issue in res.unresolved_issues)
        assert any("zeros" in issue.lower() for issue in res.unresolved_issues)

        # Confirm original empty text was NOT fabricated/invented
        assert res.repaired_output["text"] == ""


def test_repair_max_rounds_enforced():
    # Payload with multiple non-mechanical faults that cannot reach 80 threshold
    payload = {
        "id": "e_stubborn_payload",
        "text": "",            # empty text (non-mechanical, cannot fabricate)
        "entity_type": "post",
        "tags": ["query"],
        "dim": 64,
        "vector": [0.0] * 64,  # zero vector (non-mechanical, cannot normalize)
    }
    res: RepairResult = repair_output(payload, max_rounds=3)
    assert res.rounds == 3
    assert res.passed is False
    assert res.final_score < 80
    assert len(res.unresolved_issues) >= 2
    assert any("empty text" in issue.lower() or "missing" in issue.lower() for issue in res.unresolved_issues)


def test_ledger_jsonl_output():
    payload = {
        "id": "e_ledger_test",
        "text": "  Unformatted streaming query  ",
        "dim": 64,
        "vector": [1.0] * 64,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = Path(tmpdir) / "audit_ledger.jsonl"
        res = repair_output(payload, ledger_path=str(ledger_file))
        assert ledger_file.exists()
        lines = [json.loads(line) for line in ledger_file.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == res.rounds
        for record in lines:
            assert "round" in record
            assert "timestamp" in record
            assert "score" in record
            assert "threshold" in record
            assert "passed" in record
            assert "repaired_fields" in record
            assert "unresolved_issues" in record
