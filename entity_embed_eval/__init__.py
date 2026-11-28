"""Multimodal Entity Representation and Embedding Evaluation Engine."""

from .core import Entity, RepresentationEngine, cosine_similarity
from .evaluation import RelevancePair, BenchmarkResult, evaluate_embeddings
from .responsible_ai import (
    PolicyConfig,
    DeploymentCheckResult,
    DeploymentGateReport,
    ResponsibleAIGate,
)
from .compliance import (
    ComplianceDomain,
    ComplianceRule,
    AuditViolation,
    ComplianceAuditReport,
    RegulatoryComplianceAuditor,
    DEFAULT_COMPLIANCE_RULES,
)
from .verify_repair import (
    CheckResult,
    VerificationScore,
    RepairResult,
    load_rubric,
    verify_output,
    repair_output,
)

__all__ = [
    "Entity",
    "RepresentationEngine",
    "cosine_similarity",
    "RelevancePair",
    "BenchmarkResult",
    "evaluate_embeddings",
    "PolicyConfig",
    "DeploymentCheckResult",
    "DeploymentGateReport",
    "ResponsibleAIGate",
    "ComplianceDomain",
    "ComplianceRule",
    "AuditViolation",
    "ComplianceAuditReport",
    "RegulatoryComplianceAuditor",
    "DEFAULT_COMPLIANCE_RULES",
    "CheckResult",
    "VerificationScore",
    "RepairResult",
    "load_rubric",
    "verify_output",
    "repair_output",
]
