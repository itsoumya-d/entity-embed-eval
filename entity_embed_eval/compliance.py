"""Regulatory compliance audit harness for evaluating sensitive data leakage in entity embeddings."""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .core import Entity


class ComplianceDomain(str, Enum):
    """Supported regulatory compliance frameworks."""
    HEALTHCARE_HIPAA = "healthcare_hipaa"
    EDUCATION_FERPA = "education_ferpa"
    FINANCIAL_GLBA = "financial_glba"
    GENERAL_PRIVACY = "general_privacy"


@dataclass
class ComplianceRule:
    """Detection rule for sensitive identifiers in RAG and entity retrieval pipelines."""
    id: str
    domain: ComplianceDomain
    name: str
    pattern: str
    severity: str = "HIGH"  # CRITICAL, HIGH, MEDIUM
    description: str = ""

    def __post_init__(self):
        if not self.id or not self.pattern:
            raise ValueError("Rule ID and pattern must be non-empty strings")
        if self.severity not in {"CRITICAL", "HIGH", "MEDIUM"}:
            raise ValueError(f"Invalid severity '{self.severity}'. Must be CRITICAL, HIGH, or MEDIUM")
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.pattern}': {e}") from e


# Standard patterns for regulatory compliance
DEFAULT_COMPLIANCE_RULES: List[ComplianceRule] = [
    # HIPAA Safe Harbor Identifiers
    ComplianceRule(
        id="hipaa_mrn",
        domain=ComplianceDomain.HEALTHCARE_HIPAA,
        name="Medical Record Number (MRN)",
        pattern=r"\b(?:MRN|MEDREC|PATIENT[\s_-]?ID)[:\s#\-]*[A-Z0-9]{6,12}\b",
        severity="CRITICAL",
        description="Protected health information: patient medical record identifier",
    ),
    ComplianceRule(
        id="hipaa_ssn",
        domain=ComplianceDomain.HEALTHCARE_HIPAA,
        name="Social Security Number (SSN)",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        severity="CRITICAL",
        description="Protected health information: national identity number",
    ),
    ComplianceRule(
        id="hipaa_phone",
        domain=ComplianceDomain.HEALTHCARE_HIPAA,
        name="Direct Telephone Number",
        pattern=r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        severity="HIGH",
        description="Protected health information: patient contact telephone",
    ),
    ComplianceRule(
        id="hipaa_email",
        domain=ComplianceDomain.HEALTHCARE_HIPAA,
        name="Personal Email Address",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        severity="HIGH",
        description="Protected health information: electronic mail address",
    ),
    # FERPA Educational Records
    ComplianceRule(
        id="ferpa_student_id",
        domain=ComplianceDomain.EDUCATION_FERPA,
        name="Student Identification Number",
        pattern=r"\b(?:STU|STUDENT|BANNER)(?:[\s_-]?ID)?[:\s#\-]*\d{6,10}\b",
        severity="CRITICAL",
        description="FERPA protected student educational record identifier",
    ),
    ComplianceRule(
        id="ferpa_gpa_grade",
        domain=ComplianceDomain.EDUCATION_FERPA,
        name="Restricted Grade Record",
        pattern=r"\b(?:GPA|GRADE|TRANSCRIPT)[:\s]*[0-4]\.\d{1,2}\b",
        severity="MEDIUM",
        description="FERPA protected individual student academic performance record",
    ),
]


@dataclass
class AuditViolation:
    """Recorded compliance fault during entity evaluation."""
    rule_id: str
    rule_name: str
    entity_id: str
    matched_snippet: str
    severity: str
    remediation: str


@dataclass
class ComplianceAuditReport:
    """Structured report returned by the regulatory compliance auditor."""
    domain: ComplianceDomain
    is_compliant: bool
    compliance_score: float
    total_entities: int
    clean_entities: int
    violations: List[AuditViolation] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class RegulatoryComplianceAuditor:
    """Evaluates entity representations against regulatory privacy standards."""

    def __init__(
        self,
        domain: Union[ComplianceDomain, str] = ComplianceDomain.HEALTHCARE_HIPAA,
        custom_rules: Optional[List[ComplianceRule]] = None,
        min_pass_score: float = 0.95,
    ) -> None:
        if isinstance(domain, str):
            try:
                self.domain = ComplianceDomain(domain.lower())
            except ValueError as e:
                valid = [d.value for d in ComplianceDomain]
                raise ValueError(f"Unsupported compliance domain '{domain}'. Must be one of {valid}") from e
        else:
            self.domain = domain

        if not (0.0 <= min_pass_score <= 1.0):
            raise ValueError(f"min_pass_score must be between 0.0 and 1.0, got {min_pass_score}")

        self.min_pass_score = min_pass_score
        rules = [r for r in DEFAULT_COMPLIANCE_RULES if r.domain == self.domain]
        if custom_rules:
            rules.extend(custom_rules)
        self.rules = rules

    def audit_entities(self, entities: List[Entity]) -> ComplianceAuditReport:
        """Audits an entity batch for regulatory violations."""
        if not isinstance(entities, list):
            raise ValueError("Entities parameter must be a list of Entity instances")

        if not entities:
            return ComplianceAuditReport(
                domain=self.domain,
                is_compliant=True,
                compliance_score=1.0,
                total_entities=0,
                clean_entities=0,
                violations=[],
            )

        violations: List[AuditViolation] = []
        clean_count = 0

        for entity in entities:
            if not isinstance(entity, Entity):
                raise ValueError("Every item in entities must be an instance of Entity")

            entity_has_violation = False
            for rule in self.rules:
                matches = list(re.finditer(rule.pattern, entity.text, re.IGNORECASE))
                for match in matches:
                    entity_has_violation = True
                    violations.append(
                        AuditViolation(
                            rule_id=rule.id,
                            rule_name=rule.name,
                            entity_id=entity.id,
                            matched_snippet=match.group(0),
                            severity=rule.severity,
                            remediation=f"Tokenize or redact {rule.name} prior to embedding indexing",
                        )
                    )

            if not entity_has_violation:
                clean_count += 1

        # Calculate weighted compliance score
        severity_penalties = {"CRITICAL": 0.40, "HIGH": 0.20, "MEDIUM": 0.10}
        total_penalty = sum(severity_penalties.get(v.severity, 0.15) for v in violations)
        score = max(0.0, round(1.0 - (total_penalty / max(1, len(entities))), 4))
        is_compliant = score >= self.min_pass_score and len(violations) == 0

        return ComplianceAuditReport(
            domain=self.domain,
            is_compliant=is_compliant,
            compliance_score=score,
            total_entities=len(entities),
            clean_entities=clean_count,
            violations=violations,
        )

    def sanitize_and_redact(self, entities: List[Entity]) -> Tuple[List[Entity], int]:
        """Redacts identified sensitive tokens into safe pseudonym placeholders."""
        if not entities:
            return [], 0

        redacted_entities: List[Entity] = []
        total_redactions = 0

        for entity in entities:
            text = entity.text
            for rule in self.rules:
                def replace_fn(m):
                    nonlocal total_redactions
                    total_redactions += 1
                    return f"[{rule.id.upper()}_REDACTED]"

                text = re.sub(rule.pattern, replace_fn, text, flags=re.IGNORECASE)

            redacted_entities.append(
                Entity(
                    id=entity.id,
                    text=text,
                    entity_type=entity.entity_type,
                    tags=list(entity.tags or []),
                    metadata=dict(entity.metadata or {}),
                )
            )

        return redacted_entities, total_redactions

    def export_audit_log(
        self, report: ComplianceAuditReport, filepath: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Serializes audit report to dictionary and optional JSON file."""
        data = {
            "domain": report.domain.value,
            "is_compliant": report.is_compliant,
            "compliance_score": report.compliance_score,
            "total_entities": report.total_entities,
            "clean_entities": report.clean_entities,
            "violation_count": report.violation_count,
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "entity_id": v.entity_id,
                    "matched_snippet": v.matched_snippet,
                    "severity": v.severity,
                    "remediation": v.remediation,
                }
                for v in report.violations
            ],
        }
        if filepath:
            Path(filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
