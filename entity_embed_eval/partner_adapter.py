"""Partner Data Schema Diagnostic and Entity Adaptation Engine.

Enables forward-deployed engineers to diagnose partner schema health,
detect field-level missingness, and adapt decentralized partner data structures
into canonical Entity representations for semantic embedding and evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .core import Entity


@dataclass
class FieldDiagnosticReport:
    """Diagnostic health summary of partner records."""
    total_records: int
    valid_records: int
    completeness_score: float
    missing_fields: Dict[str, int]
    type_errors: List[str]
    anomalies: List[str]
    ready_for_ingestion: bool


class PartnerSchemaAdapter:
    """Diagnoses partner data structures and adapts heterogenous records into Entity format."""

    def __init__(self, default_entity_type: str = "partner_record"):
        self.default_entity_type = default_entity_type

    def diagnose_schema(
        self,
        records: List[Dict[str, Any]],
        required_fields: Optional[List[str]] = None,
        expected_types: Optional[Dict[str, Type]] = None
    ) -> FieldDiagnosticReport:
        """Diagnoses data quality, missing keys, and type conformity in partner records."""
        if not records:
            return FieldDiagnosticReport(
                total_records=0,
                valid_records=0,
                completeness_score=0.0,
                missing_fields={},
                type_errors=[],
                anomalies=["No records provided for diagnostic evaluation."],
                ready_for_ingestion=False
            )

        required = required_fields or ["id", "text"]
        types = expected_types or {"id": str, "text": str}

        missing_counts: Dict[str, int] = {f: 0 for f in required}
        type_errors: List[str] = []
        anomalies: List[str] = []
        valid_count = 0

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                type_errors.append(f"Record index {idx} is malformed: expected dict, got {type(rec).__name__}")
                continue

            record_valid = True
            for f in required:
                val = rec.get(f)
                if val is None or (isinstance(val, str) and not val.strip()):
                    missing_counts[f] += 1
                    record_valid = False
                elif f in types and not isinstance(val, types[f]):
                    type_errors.append(
                        f"Record {idx} field '{f}' invalid type: expected {types[f].__name__}, got {type(val).__name__}"
                    )
                    record_valid = False

            if record_valid:
                valid_count += 1

        total = len(records)
        total_required_checks = total * len(required)
        missing_total = sum(missing_counts.values())
        completeness = max(0.0, (total_required_checks - missing_total) / total_required_checks) if total_required_checks else 0.0

        if completeness < 0.70:
            anomalies.append(f"High missingness detected: completeness score is only {completeness:.1%}")
        if type_errors:
            anomalies.append(f"Schema type discrepancies found: {len(type_errors)} type error(s)")

        ready = valid_count > 0 and len(type_errors) == 0 and completeness >= 0.80

        return FieldDiagnosticReport(
            total_records=total,
            valid_records=valid_count,
            completeness_score=round(completeness, 4),
            missing_fields={k: v for k, v in missing_counts.items() if v > 0},
            type_errors=type_errors,
            anomalies=anomalies,
            ready_for_ingestion=ready
        )

    def adapt_records(
        self,
        records: List[Dict[str, Any]],
        field_mapping: Dict[str, str],
        default_values: Optional[Dict[str, Any]] = None,
        drop_invalid: bool = False
    ) -> List[Entity]:
        """Adapts heterogenous partner fields into canonical Entity models."""
        if not field_mapping or not isinstance(field_mapping, dict):
            raise ValueError("field_mapping must be a non-empty dictionary mapping canonical fields (id, text, tags, metadata)")

        id_key = field_mapping.get("id")
        text_key = field_mapping.get("text")

        if not id_key or not text_key:
            raise ValueError("field_mapping must include mappings for both 'id' and 'text'")

        defaults = default_values or {}
        entities: List[Entity] = []

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                if drop_invalid:
                    continue
                raise TypeError(f"Record at index {idx} is malformed: expected dict, got {type(rec).__name__}")

            entity_id = rec.get(id_key, defaults.get("id"))
            text_val = rec.get(text_key, defaults.get("text"))

            if not entity_id or not text_val:
                if drop_invalid:
                    continue
                raise ValueError(f"Missing required identifier or text at record index {idx} (mapped keys '{id_key}', '{text_key}')")

            # Extract tags
            tags: List[str] = []
            tags_key = field_mapping.get("tags")
            if tags_key and tags_key in rec:
                raw_tags = rec[tags_key]
                if isinstance(raw_tags, list):
                    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
                elif isinstance(raw_tags, str):
                    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

            # Extract metadata
            metadata: Dict[str, str] = {}
            meta_key = field_mapping.get("metadata")
            if meta_key and isinstance(rec.get(meta_key), dict):
                metadata = {str(k): str(v) for k, v in rec[meta_key].items()}
            else:
                for k, v in rec.items():
                    if k not in (id_key, text_key, tags_key, meta_key) and v is not None:
                        metadata[str(k)] = str(v)

            entity_type = rec.get("entity_type", defaults.get("entity_type", self.default_entity_type))

            entities.append(Entity(
                id=str(entity_id),
                text=str(text_val),
                entity_type=str(entity_type),
                tags=tags,
                metadata=metadata
            ))

        return entities
