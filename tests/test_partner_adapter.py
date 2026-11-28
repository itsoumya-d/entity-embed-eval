import pytest
from entity_embed_eval import Entity, RepresentationEngine, cosine_similarity
from entity_embed_eval.partner_adapter import FieldDiagnosticReport, PartnerSchemaAdapter


def test_partner_adapter_normal_flow():
    """Validates successful schema diagnosis and entity adaptation on standard partner records."""
    adapter = PartnerSchemaAdapter(default_entity_type="beneficiary_survey")
    records = [
        {"survey_id": "REC-001", "response_notes": "Beneficiary completed micro-grant training", "region": "District-A", "tags": ["training", "finance"]},
        {"survey_id": "REC-002", "response_notes": "Follow-up visit verified seed crop distribution", "region": "District-B", "tags": "agriculture,seed"}
    ]
    diag = adapter.diagnose_schema(records, required_fields=["survey_id", "response_notes"])
    assert diag.total_records == 2
    assert diag.valid_records == 2
    assert diag.completeness_score == 1.0
    assert diag.ready_for_ingestion is True
    assert len(diag.type_errors) == 0

    entities = adapter.adapt_records(
        records,
        field_mapping={"id": "survey_id", "text": "response_notes", "tags": "tags"}
    )
    assert len(entities) == 2
    assert isinstance(entities[0], Entity)
    assert entities[0].id == "REC-001"
    assert "District-A" in entities[0].metadata.get("region", "")
    assert "training" in entities[0].tags
    assert "seed" in entities[1].tags


def test_partner_adapter_empty_input():
    """Verifies that an empty record list yields zero counts without crashing."""
    adapter = PartnerSchemaAdapter()
    diag = adapter.diagnose_schema([])
    assert diag.total_records == 0
    assert diag.valid_records == 0
    assert diag.completeness_score == 0.0
    assert diag.ready_for_ingestion is False
    assert "No records provided" in diag.anomalies[0]


def test_partner_adapter_malformed_record_type_error():
    """Detects type mismatch when integer is provided where string is expected."""
    adapter = PartnerSchemaAdapter()
    malformed = [
        {"id": "V-101", "text": 12345},  # malformed: text is int
        {"id": 999, "text": "Valid text"}  # malformed: id is int
    ]
    diag = adapter.diagnose_schema(malformed, required_fields=["id", "text"], expected_types={"id": str, "text": str})
    assert diag.valid_records == 0
    assert len(diag.type_errors) >= 2
    assert any("expected str, got int" in err for err in diag.type_errors)
    assert diag.ready_for_ingestion is False


def test_partner_adapter_boundary_null_or_zero_values():
    """Checks boundary edge conditions with empty strings, null values, and whitespace."""
    adapter = PartnerSchemaAdapter()
    boundary_records = [
        {"id": "B-01", "text": "   "},  # whitespace text
        {"id": None, "text": "Valid text"},  # null id
        {"id": "B-03", "text": "Solid description"}
    ]
    diag = adapter.diagnose_schema(boundary_records)
    assert diag.total_records == 3
    assert diag.valid_records == 1
    assert "text" in diag.missing_fields
    assert "id" in diag.missing_fields
    assert diag.completeness_score < 1.0


def test_partner_adapter_missing_required_identifier_fails():
    """Fails with ValueError when required mapped identifier is missing from record."""
    adapter = PartnerSchemaAdapter()
    records = [{"unmapped_id": "999", "content": "Missing canonical id field"}]
    with pytest.raises(ValueError, match="Missing required identifier or text"):
        adapter.adapt_records(records, field_mapping={"id": "record_id", "text": "content"})


def test_partner_adapter_invalid_field_mapping_rejected():
    """Rejects empty or non-dict field mapping configuration."""
    adapter = PartnerSchemaAdapter()
    records = [{"id": "1", "text": "Sample"}]
    with pytest.raises(ValueError, match="field_mapping must be a non-empty dictionary"):
        adapter.adapt_records(records, field_mapping={})
    with pytest.raises(ValueError, match="must include mappings for both 'id' and 'text'"):
        adapter.adapt_records(records, field_mapping={"id": "id"})


def test_partner_adapter_property_idempotence():
    """Invariant property: adapting identical input multiple times produces identical representations."""
    adapter = PartnerSchemaAdapter()
    records = [
        {"uuid": "id-abc", "desc": "Health clinic supplies received", "category": "health"},
        {"uuid": "id-xyz", "desc": "Clean water borehole drilled", "category": "water"}
    ]
    mapping = {"id": "uuid", "text": "desc"}
    res1 = adapter.adapt_records(records, field_mapping=mapping)
    res2 = adapter.adapt_records(records, field_mapping=mapping)
    assert len(res1) == len(res2)
    for e1, e2 in zip(res1, res2):
        assert e1.id == e2.id
        assert e1.text == e2.text
        assert e1.metadata == e2.metadata

    # Invariant: downstream embeddings match exactly
    engine = RepresentationEngine(dim=64, calibrated=True)
    vec1 = engine.embed(res1[0])
    vec2 = engine.embed(res2[0])
    assert cosine_similarity(vec1, vec2) == pytest.approx(1.0, abs=1e-6)
