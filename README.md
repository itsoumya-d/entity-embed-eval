# entity-embed-eval

[![CI](https://github.com/itsoumya-d/entity-embed-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/itsoumya-d/entity-embed-eval/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Multimodal entity representation and offline embedding evaluation engine for high-throughput social discussions, community posts, and context-targeted creative ads.

---

## 1. Problem & Motivation

In high-throughput social platforms, user-generated posts and advertiser creative contain informal subculture slang, markdown formatting, and fast-evolving terminology. Standard off-the-shelf embedding models drift rapidly when applied to unstructured community interactions, leading to:

1. **Semantic drift** across conversational subtopics and creative placements.
2. **High false-positive retrieval** when cross-domain entity intent is uncalibrated.
3. **Slow ML experimentation cycles** due to lack of automated, reproducible offline relevance benchmarking.

`entity-embed-eval` provides a lightweight, deterministic representation and calibration engine with automated offline evaluation pipelines to measure relevance precision and drift.

---

## 2. Architecture & Design

```
Raw Entity (Post / Creative / Topic)
                │
                ▼
      [Text Normalization]       <-- URL masking, punctuation stripping, lowercasing
                │
                ▼
    [Subword & Feature Hash]    <-- Unigrams + character 3-grams + tag weighting
                │
                ▼
       [Tanh Calibration]       <-- Bounded non-linear activation to curb variance
                │
                ▼
        [L2 Normalization]      <-- Unit-norm dense representation vector (d=64)
                │
                ▼
    [Cosine Similarity & Eval]  <-- Multi-class benchmark against ground-truth pairs
```

### Key Components

- **RepresentationEngine**: Generates dense 64-dimensional semantic vectors using character n-gram hashing and entity-type-aware weighting.
- **Tanh Calibration**: Compresses extreme feature hash collisions and standardizes representation distributions across disparate domains.
- **Evaluation Suite**: Runs automated multi-class relevance benchmarks computing Precision, Recall, F1 score, and lift over uncalibrated baselines.
- **FastAPI Microservice**: Production-ready API for real-time embedding extraction, pairwise similarity calculation, and automated drift benchmarking.

---

## 3. Measured Benchmark Results

Evaluated on multi-class cross-domain entity pairs (Gaming, Cloud Infrastructure, Personal Finance, and cross-category mismatches):

| Metric | Uncalibrated Baseline | Calibrated Engine (`entity-embed-eval`) | Delta |
|:---|:---:|:---:|:---:|
| **Precision** | 0.7143 | **0.8889** | **+24.4%** |
| **Accuracy** | 0.7000 | **0.9000** | **+28.6%** |
| **F1 Score** | 0.8333 | **0.8889** | **+6.7%** |
| **Mean Pos. Sim** | 0.4512 | **0.5824** | +0.1312 |
| **Mean Neg. Sim** | 0.3810 | **0.1872** | -0.1938 |

The calibration layer widens the margin between relevant and irrelevant entity pairs, cutting cross-domain false positive ad impressions while boosting precision by **+24.4%**.

---

## 4. Quick Start

### Installation

```bash
git clone https://github.com/itsoumya-d/entity-embed-eval.git
cd entity-embed-eval
pip install -e ".[test]"
```

### Run Tests

```bash
pytest -v
```

### Python API Usage

```python
from entity_embed_eval import Entity, RepresentationEngine, cosine_similarity

engine = RepresentationEngine(dim=64, calibrated=True)

post = Entity(
    id="post_101",
    text="Configuring high-throughput Kafka streaming partitions on Kubernetes",
    tags=["kafka", "cloud", "devops"],
)

ad = Entity(
    id="ad_202",
    text="Managed enterprise event streaming and distributed Kafka clusters",
    entity_type="ad",
    tags=["kafka", "cloud"],
)

v_post = engine.embed(post)
v_ad = engine.embed(ad)

similarity = cosine_similarity(v_post, v_ad)
print(f"Semantic similarity: {similarity:.4f}")
# Output: Semantic similarity: 0.6512 (Relevant match)
```

### Run Evaluation Suite

```python
from entity_embed_eval.evaluation import (
    evaluate_embeddings,
    get_default_benchmark_dataset,
)

dataset = get_default_benchmark_dataset()
result = evaluate_embeddings(engine, dataset, similarity_threshold=0.35)

print(f"Precision: {result.precision}")
print(f"Improvement over baseline: +{result.improvement_pct}%")
```

### Responsible AI Deployment Checks

The `ResponsibleAIGate` embeds automated safety, hallucination drift, and demographic fairness parity gates directly into model deployment workflows.

```python
from entity_embed_eval import Entity, ResponsibleAIGate, PolicyConfig

# Configure deployment policy thresholds
policy = PolicyConfig(
    max_toxicity=0.05,
    min_groundedness=0.75,
    max_demographic_disparity=0.15,
)
gate = ResponsibleAIGate(policy=policy)

# Validate candidate entity outputs against policy guardrails
entities = [
    Entity(id="doc_1", text="Automated data lake orchestration with cloud storage"),
]
report = gate.evaluate_deployment(
    entities=entities,
    context_sources=["Cloud storage data lake orchestration manual"],
    sample_output="Automated data lake orchestration on cloud storage",
    group_scores={"tier_1": 0.90, "tier_2": 0.86},
)

if not report.is_approved:
    print(f"Deployment blocked: {report.failures}")
else:
    print("Deployment approved: all Responsible AI verification checks passed.")
```

**Scope & Limitations**:
- Safety verification performs lexical and pattern-based prohibited term scanning; domain-specific external classifier APIs should be chained for nuanced toxic sentiment.
- Groundedness evaluation measures lexical token alignment; semantic NLI models can be plugged in when cross-lingual generation is required.

### Regulatory Compliance Audit Gate

The `RegulatoryComplianceAuditor` evaluates entity representations and RAG contexts against regulatory compliance standards (such as healthcare HIPAA Safe Harbor 18-identifiers and educational FERPA record constraints) prior to embedding indexing or model context ingestion.

```python
from entity_embed_eval import Entity, RegulatoryComplianceAuditor, ComplianceDomain

# Initialize auditor for regulated domain
auditor = RegulatoryComplianceAuditor(domain=ComplianceDomain.HEALTHCARE_HIPAA)

entities = [
    Entity(id="e1", text="Patient protocol: follow up scheduled for clinical evaluation"),
    Entity(id="e2", text="Patient MRN-8472910 with SSN 123-45-6789 admitted for care"),
]

# Run pre-indexing compliance evaluation
report = auditor.audit_entities(entities)
if not report.is_compliant:
    print(f"Compliance violations detected: {report.violation_count}")
    # Deterministically tokenize and redact sensitive identifiers
    safe_entities, redaction_count = auditor.sanitize_and_redact(entities)
    print(f"Redacted {redaction_count} protected tokens; sanitized entities ready for embedding.")
```

**Scope & Limitations**:
- Built-in patterns cover standard structured identifiers (MRN, SSN, telephone numbers, emails, Banner/Student IDs, grade performance records).
- Unstructured narrative disclosures may require custom regex patterns or specialized Named Entity Recognition (NER) models for edge-case coverage.

### Verify-and-Repair Quality Loop

The `verify_repair` module evaluates entity payloads and representation vectors against a declarative weighted rubric (`rubric.json`). Faults are scored on a 0–100 scale against an 80-point pass threshold. Mechanical defects are deterministically repaired up to 3 rounds and logged to an immutable JSONL audit ledger. Non-mechanical defects (such as missing text or unnormalized zero vectors) are reported for human review without fabrication.

#### Rubric Specification

| Check Name | Weight | Mechanical | Description |
|:---|:---:|:---:|:---|
| `required_envelope` | 20 | Yes | Required fields (`id`, `text`, `entity_type`, `tags`) present |
| `text_formatting` | 20 | Yes | Text sanitized, stripped of irregular whitespace & control characters |
| `vector_dimension` | 20 | No | Embedding vector length matches configured dimension (64 elements) |
| `vector_unit_norm` | 20 | Yes | Vector L2 norm equals 1.0 within tolerance (+/- 0.005) |
| `score_bounds` | 20 | Yes | Cosine similarity or relevance score is within `[-1.0, 1.0]` |

#### Before & After Repair Example

```python
from entity_embed_eval import verify_output, repair_output

# Before: Broken raw fixture scoring 20/100 (Threshold: 80)
broken_payload = {
    "id": "e_raw_901",
    "text": "   Pipeline data ingestion staging table   \n",
    "dim": 64,
    "vector": [2.5] * 64,  # unnormalized (norm = 20.0)
    "score": 2.4,          # out-of-range score > 1.0
}

initial_score = verify_output(broken_payload)
print(f"Before Repair Score: {initial_score.score}/100 (Passed: {initial_score.passed})")
# Output: Before Repair Score: 20/100 (Passed: False)

# Run automated mechanical repair loop (logs to audit_ledger.jsonl)
result = repair_output(broken_payload, ledger_path="audit_ledger.jsonl", max_rounds=3)
print(f"After Repair Score: {result.final_score}/100 (Passed: {result.passed}) in {result.rounds} round(s)")
print(f"Repaired Fields: {result.repaired_fields}")
# Output: After Repair Score: 100/100 (Passed: True) in 1 round(s)
# Repaired Fields: ['entity_type', 'score', 'tags', 'text', 'vector']
```

---

## 5. License

MIT
