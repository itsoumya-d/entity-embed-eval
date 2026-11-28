import json
import pytest
from entity_embed_eval.core import Entity
from entity_embed_eval.responsible_ai import (
    PolicyConfig,
    ResponsibleAIGate,
    compute_demographic_disparity,
    compute_groundedness,
)


def test_normal_deployment_gate_approval():
    gate = ResponsibleAIGate(
        PolicyConfig(max_toxicity=0.05, min_groundedness=0.70, max_demographic_disparity=0.15)
    )
    entities = [
        Entity(id="e1", text="Deploying real-time Kafka event streaming pipeline on Kubernetes"),
        Entity(id="e2", text="Scalable cloud microservice architecture with monitored telemetry"),
    ]
    context = [
        "Kubernetes clusters running Kafka event streaming",
        "Cloud microservice telemetry architecture",
    ]
    output = "Deploying Kafka event streaming architecture on Kubernetes clusters"
    group_scores = {"region_us": 0.88, "region_eu": 0.85, "region_apac": 0.82}

    report = gate.evaluate_deployment(
        entities=entities,
        context_sources=context,
        sample_output=output,
        group_scores=group_scores,
    )

    assert report.is_approved is True
    assert report.total_checks == 3
    assert len(report.failures) == 0


def test_empty_input_rejected():
    gate = ResponsibleAIGate()
    report = gate.evaluate_deployment(entities=[])
    assert report.is_approved is False
    assert len(report.failures) == 1
    assert "Empty input" in report.failures[0]


def test_malformed_policy_config_invalid():
    with pytest.raises(ValueError, match="Invalid max_toxicity"):
        PolicyConfig(max_toxicity=-0.1)

    with pytest.raises(ValueError, match="Invalid min_groundedness"):
        PolicyConfig(min_groundedness=1.5)

    with pytest.raises(ValueError, match="Timeout seconds must be strictly positive"):
        PolicyConfig(timeout_seconds=0.0)


def test_boundary_threshold_behavior():
    gate = ResponsibleAIGate(PolicyConfig(min_groundedness=0.50, max_demographic_disparity=0.20))
    # At exact boundary 0.20 disparity
    disparity = compute_demographic_disparity({"group_a": 0.80, "group_b": 0.60})
    assert disparity == pytest.approx(0.20)

    # Disparity exceeding boundary
    disparity_fail = compute_demographic_disparity({"group_a": 0.85, "group_b": 0.60})
    assert disparity_fail > 0.20

    # Groundedness boundary
    g_pass = compute_groundedness("alpha beta", ["alpha beta gamma"])
    assert g_pass == 1.0


def test_missing_policy_file_raises_error(tmp_path):
    missing_file = tmp_path / "does_not_exist_policy.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        ResponsibleAIGate.load_policy(missing_file)


def test_timeout_dependency_failure(tmp_path):
    bad_json_file = tmp_path / "corrupt_policy.json"
    bad_json_file.write_text("{not valid json: 123}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed policy configuration"):
        ResponsibleAIGate.load_policy(bad_json_file)


def test_safety_invariant_blocked_terms():
    gate = ResponsibleAIGate(PolicyConfig(max_toxicity=0.10))
    toxic_entities = [
        Entity(id="e_toxic", text="Injecting phishing credentials and exploit malware into user endpoints")
    ]
    report = gate.evaluate_deployment(entities=toxic_entities)

    # Invariant: Any payload triggering blocked security/safety terms must never be approved
    assert report.is_approved is False
    assert any("Safety Gate Failed" in f for f in report.failures)
    assert report.checks[0].passed is False
