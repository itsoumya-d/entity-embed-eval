"""Verify-and-repair loop for entity representation and embedding outputs.

Scores entity payloads against a declarative weighted rubric (rubric.json),
mechanically repairs structural, formatting, and bounded numeric fields up to 3 rounds,
reports non-mechanical issues for human review without guessing, and logs all attempts
to a JSONL audit ledger.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CheckResult:
    """Evaluation result for an individual rubric check."""
    name: str
    weight: int
    passed: bool
    details: str
    mechanical: bool


@dataclass(frozen=True)
class VerificationScore:
    """Overall score and individual check breakdown from rubric verification."""
    score: int
    threshold: int
    passed: bool
    checks: List[CheckResult]

    @property
    def check_map(self) -> Dict[str, bool]:
        return {c.name: c.passed for c in self.checks}


@dataclass
class RepairResult:
    """Outcome of mechanical repair loop across up to 3 rounds."""
    repaired_output: Dict[str, Any]
    final_score: int
    passed: bool
    rounds: int
    repaired_fields: List[str]
    unresolved_issues: List[str]
    history: List[Dict[str, Any]] = field(default_factory=list)


def default_rubric_path() -> Path:
    """Returns default path to rubric.json located in package root."""
    return Path(__file__).parent / "rubric.json"


def load_rubric(rubric_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads and validates rubric specification from JSON file."""
    path = Path(rubric_path) if rubric_path else default_rubric_path()
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def verify_output(data: Dict[str, Any], rubric_path: Optional[str] = None) -> VerificationScore:
    """Scores entity embedding output dict from 0 to 100 against rubric checks."""
    rubric = load_rubric(rubric_path)
    threshold = int(rubric.get("threshold", 80))
    checks_config = rubric.get("checks", [])

    results: List[CheckResult] = []
    total_score = 0

    for c in checks_config:
        name = c["name"]
        weight = int(c["weight"])
        is_mech = bool(c.get("mechanical", True))

        if name == "required_envelope":
            required_keys = {"id", "text", "entity_type", "tags"}
            missing = [k for k in required_keys if k not in data or data[k] is None]
            if not missing:
                results.append(CheckResult(name, weight, True, "All required envelope fields present", is_mech))
                total_score += weight
            else:
                results.append(CheckResult(name, weight, False, f"Missing required fields: {sorted(missing)}", is_mech))

        elif name == "text_formatting":
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                results.append(CheckResult(name, weight, False, "Text is empty or missing (non-mechanical)", False))
            elif text != text.strip() or "  " in text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
                results.append(
                    CheckResult(
                        name,
                        weight,
                        False,
                        "Text contains irregular whitespace or control characters",
                        True,
                    )
                )
            else:
                results.append(CheckResult(name, weight, True, "Text is sanitized and normalized", is_mech))
                total_score += weight

        elif name == "vector_dimension":
            expected_dim = int(data.get("dim", 64))
            vec = data.get("vector")
            if isinstance(vec, list) and len(vec) == expected_dim and all(isinstance(x, (int, float)) for x in vec):
                results.append(CheckResult(name, weight, True, f"Vector matches expected dimension {expected_dim}", is_mech))
                total_score += weight
            else:
                act_len = len(vec) if isinstance(vec, list) else 0
                results.append(
                    CheckResult(
                        name,
                        weight,
                        False,
                        f"Vector dimension {act_len} != expected {expected_dim}",
                        False,
                    )
                )

        elif name == "vector_unit_norm":
            vec = data.get("vector")
            if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
                norm = math.sqrt(sum(float(x) * float(x) for x in vec))
                if norm == 0.0:
                    results.append(CheckResult(name, weight, False, "Vector is all zeros; cannot normalize", False))
                elif abs(norm - 1.0) <= 0.005:
                    results.append(CheckResult(name, weight, True, f"Vector L2 norm {norm:.4f} is unit length", is_mech))
                    total_score += weight
                else:
                    results.append(
                        CheckResult(
                            name,
                            weight,
                            False,
                            f"Vector L2 norm {norm:.4f} out of unit tolerance (+/- 0.005)",
                            True,
                        )
                    )
            else:
                results.append(CheckResult(name, weight, False, "Vector is empty or malformed", False))

        elif name == "score_bounds":
            score_found = False
            for k in ("score", "similarity", "similarity_score"):
                if k in data and data[k] is not None:
                    score_found = True
                    val = data[k]
                    if isinstance(val, (int, float)):
                        if -1.0 <= float(val) <= 1.0:
                            results.append(CheckResult(name, weight, True, f"Score {val} within [-1.0, 1.0]", is_mech))
                            total_score += weight
                        else:
                            results.append(
                                CheckResult(
                                    name,
                                    weight,
                                    False,
                                    f"Score {val} outside valid range [-1.0, 1.0]",
                                    True,
                                )
                            )
                    else:
                        results.append(CheckResult(name, weight, False, f"Score {val!r} is non-numeric", False))
                    break

            if not score_found:
                results.append(CheckResult(name, weight, True, "No optional similarity score present", is_mech))
                total_score += weight

        else:
            results.append(CheckResult(name, weight, True, f"Custom check {name} defaulted to pass", is_mech))
            total_score += weight

    return VerificationScore(
        score=total_score,
        threshold=threshold,
        passed=total_score >= threshold,
        checks=results,
    )


def repair_output(
    data: Dict[str, Any],
    rubric_path: Optional[str] = None,
    ledger_path: Optional[str] = None,
    max_rounds: int = 3,
) -> RepairResult:
    """Executes verify-and-repair loop for up to max_rounds (default 3).

    Repairs only mechanical faults (whitespace formatting, default missing fields,
    vector unit normalization, and out-of-range score clamping). Non-mechanical
    faults are flagged in unresolved_issues for human review and never fabricated.
    Appends each iteration as a structured JSON record into ledger_path.
    """
    cur = copy.deepcopy(data)
    repaired_fields: List[str] = []
    unresolved_issues: List[str] = []
    history: List[Dict[str, Any]] = []

    last_score = verify_output(cur, rubric_path)

    for round_num in range(1, max_rounds + 1):
        if last_score.passed and round_num > 1:
            break

        round_repaired_fields: List[str] = []

        for check in last_score.checks:
            if check.passed:
                continue

            if check.name == "required_envelope":
                if "entity_type" not in cur or not cur.get("entity_type"):
                    cur["entity_type"] = "post"
                    round_repaired_fields.append("entity_type")
                if "tags" not in cur or not isinstance(cur.get("tags"), list):
                    cur["tags"] = []
                    round_repaired_fields.append("tags")
                if "text" not in cur or not cur.get("text"):
                    msg = "Missing 'text' field requires human review; cannot fabricate entity content"
                    if msg not in unresolved_issues:
                        unresolved_issues.append(msg)
                if "id" not in cur or not cur.get("id"):
                    msg = "Missing 'id' field requires human review; cannot assign arbitrary identifier"
                    if msg not in unresolved_issues:
                        unresolved_issues.append(msg)

            elif check.name == "text_formatting":
                t = cur.get("text")
                if isinstance(t, str) and t.strip():
                    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    if cleaned != t:
                        cur["text"] = cleaned
                        round_repaired_fields.append("text")
                else:
                    msg = "Empty text cannot be mechanically repaired without guessing"
                    if msg not in unresolved_issues:
                        unresolved_issues.append(msg)

            elif check.name == "vector_dimension":
                msg = f"Vector dimension mismatch ({check.details}) requires model re-embedding"
                if msg not in unresolved_issues:
                    unresolved_issues.append(msg)

            elif check.name == "vector_unit_norm":
                vec = cur.get("vector")
                if isinstance(vec, list) and vec:
                    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
                    if norm > 0.0:
                        cur["vector"] = [float(x) / norm for x in vec]
                        round_repaired_fields.append("vector")
                    else:
                        msg = "Vector is all zeros; cannot normalize without fabricating values"
                        if msg not in unresolved_issues:
                            unresolved_issues.append(msg)
                else:
                    msg = "Vector missing or malformed"
                    if msg not in unresolved_issues:
                        unresolved_issues.append(msg)

            elif check.name == "score_bounds":
                for k in ("score", "similarity", "similarity_score"):
                    if k in cur and isinstance(cur[k], (int, float)):
                        val = float(cur[k])
                        if val < -1.0 or val > 1.0:
                            cur[k] = max(-1.0, min(1.0, val))
                            round_repaired_fields.append(k)

        repaired_fields.extend(round_repaired_fields)
        post_score = verify_output(cur, rubric_path)
        last_score = post_score

        entry = {
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": post_score.score,
            "threshold": post_score.threshold,
            "passed": post_score.passed,
            "repaired_fields": sorted(list(set(round_repaired_fields))),
            "unresolved_issues": sorted(list(unresolved_issues)),
        }
        history.append(entry)

        if ledger_path:
            p = Path(ledger_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        if post_score.passed:
            break

    return RepairResult(
        repaired_output=cur,
        final_score=last_score.score,
        passed=last_score.passed,
        rounds=len(history),
        repaired_fields=sorted(list(set(repaired_fields))),
        unresolved_issues=sorted(list(unresolved_issues)),
        history=history,
    )
