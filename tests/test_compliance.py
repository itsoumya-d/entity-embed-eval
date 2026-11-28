"""Unit tests for the RegulatoryComplianceAuditor capability."""

import json
import pytest
from pathlib import Path

from entity_embed_eval.core import Entity
from entity_embed_eval.compliance import (
    ComplianceDomain,
    ComplianceRule,
    RegulatoryComplianceAuditor,
    DEFAULT_COMPLIANCE_RULES,
)


def test_normal_compliant_entities():
    """1. The normal case: Clean medical research text with zero PHI passes compliance audit."""
    auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.HEALTHCARE_HIPAA)
    clean_entities = [
        Entity(id="doc_1", text="Patient shows improvement following standard immunotherapy protocol."),
        Entity(id="doc_2", text="Biomarker panel demonstrates elevated CD8+ T-cell infiltration in tumor bed."),
    ]

    report = auditor.audit_entities(clean_entities)
    assert report.is_compliant is True
    assert report.compliance_score == 1.0
    assert report.total_entities == 2
    assert report.clean_entities == 2
    assert report.violation_count == 0


def test_empty_input_handling():
    """2. Empty input case: Empty entity batch returns clean report and empty redaction tuple."""
    auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.HEALTHCARE_HIPAA)
    report = auditor.audit_entities([])

    assert report.is_compliant is True
    assert report.compliance_score == 1.0
    assert report.total_entities == 0
    assert report.clean_entities == 0
    assert len(report.violations) == 0

    redacted, count = auditor.sanitize_and_redact([])
    assert redacted == []
    assert count == 0


def test_malformed_input_handling():
    """3. Malformed input case: Non-list inputs, invalid entity items, or broken regex raise ValueError."""
    auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.HEALTHCARE_HIPAA)

    with pytest.raises(ValueError, match="Entities parameter must be a list"):
        auditor.audit_entities("invalid-string-input")  # type: ignore

    with pytest.raises(ValueError, match="must be an instance of Entity"):
        auditor.audit_entities([{"id": "doc_1", "text": "not an Entity"}])  # type: ignore

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        ComplianceRule(
            id="broken_rule",
            domain=ComplianceDomain.HEALTHCARE_HIPAA,
            name="Broken",
            pattern="[unclosed-regex-bracket",
        )


def test_boundary_score_and_severity():
    """4. Boundary value case: Dense violations are bounded in [0.0, 1.0] and penalties scale with severity."""
    auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.HEALTHCARE_HIPAA)

    # Critical violation: MRN + SSN + phone in a single entity
    dense_phi_entity = Entity(
        id="phi_dense",
        text="Patient MRN-8849201 with SSN 123-45-6789 and contact phone 415-555-0199 admitted today.",
    )

    report = auditor.audit_entities([dense_phi_entity])
    assert report.is_compliant is False
    assert 0.0 <= report.compliance_score < 1.0
    assert report.violation_count >= 3
    criticals = [v for v in report.violations if v.severity == "CRITICAL"]
    assert len(criticals) >= 2  # MRN and SSN are CRITICAL


def test_dependency_failure_unsupported_domain(tmp_path: Path):
    """5. Dependency failure case: Unsupported domain or invalid score threshold raises explicit error."""
    with pytest.raises(ValueError, match="Unsupported compliance domain 'invalid_sector'"):
        RegulatoryComplianceAuditor(domain="invalid_sector")

    with pytest.raises(ValueError, match="min_pass_score must be between 0.0 and 1.0"):
        RegulatoryComplianceAuditor(min_pass_score=1.5)

    # Test export with valid log
    auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.EDUCATION_FERPA)
    ferpa_entity = Entity(id="stu_1", text="Student BANNER-992011 achieved GPA: 3.85 in Fall semester.")
    report = auditor.audit_entities([ferpa_entity])

    log_path = tmp_path / "audit_log.json"
    exported = auditor.export_audit_log(report, filepath=log_path)
    assert log_path.exists()
    assert exported["domain"] == "education_ferpa"
    assert exported["is_compliant"] is False
    assert len(exported["violations"]) == 2


def test_redaction_and_idempotency_invariant():
    """6. Invariant property: Redaction produces 100% compliant entities, and re-redaction is idempotent."""
    auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.HEALTHCARE_HIPAA)

    entities = [
        Entity(id="e1", text="Record for MRN: 9948201A email test@clinic.org with SSN 987-65-4321."),
        Entity(id="e2", text="Unrelated technical documentation regarding vector search index."),
    ]

    # Pre-condition: entities contain violations
    initial_report = auditor.audit_entities(entities)
    assert initial_report.is_compliant is False
    assert initial_report.violation_count > 0

    # Act: Sanitize and redact
    redacted_entities, count = auditor.sanitize_and_redact(entities)
    assert count >= 2
    assert "[HIPAA_MRN_REDACTED]" in redacted_entities[0].text or "[HIPAA_SSN_REDACTED]" in redacted_entities[0].text

    # INVARIANT 1: Sanitized entities must be 100% compliant with zero violations
    post_audit = auditor.audit_entities(redacted_entities)
    assert post_audit.is_compliant is True
    assert post_audit.compliance_score == 1.0
    assert post_audit.violation_count == 0

    # INVARIANT 2: Re-sanitizing already sanitized entities is strictly idempotent (0 additional redactions)
    second_pass, second_count = auditor.sanitize_and_redact(redacted_entities)
    assert second_count == 0
    assert [e.text for e in second_pass] == [e.text for e in redacted_entities]
