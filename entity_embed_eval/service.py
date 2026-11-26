"""FastAPI service exposing entity embedding extraction and evaluation endpoints."""

from typing import Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .core import Entity, RepresentationEngine, cosine_similarity
from .evaluation import (
    BenchmarkResult,
    RelevancePair,
    evaluate_embeddings,
    get_default_benchmark_dataset,
)

app = FastAPI(
    title="Entity Representation & Embedding Evaluation API",
    version="0.1.0",
    description="Microservice for extracting contextual entity representations and benchmarking relevance drift.",
)


class EntityPayload(BaseModel):
    id: str
    text: str
    entity_type: str = "post"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    id: str
    dim: int
    vector: List[float]


class SimilarityRequest(BaseModel):
    entity_a: EntityPayload
    entity_b: EntityPayload


class SimilarityResponse(BaseModel):
    similarity: float
    is_match: bool
    threshold: float


class EvaluationRequest(BaseModel):
    similarity_threshold: float = 0.35


engine = RepresentationEngine(dim=64, calibrated=True)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "entity-embed-eval", "version": "0.1.0"}


@app.post("/embed", response_model=EmbeddingResponse)
def embed_entity(payload: EntityPayload):
    e = Entity(
        id=payload.id,
        text=payload.text,
        entity_type=payload.entity_type,
        tags=payload.tags,
        metadata=payload.metadata,
    )
    vec = engine.embed(e)
    return EmbeddingResponse(id=e.id, dim=len(vec), vector=vec)


@app.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(req: SimilarityRequest):
    e_a = Entity(
        id=req.entity_a.id,
        text=req.entity_a.text,
        entity_type=req.entity_a.entity_type,
        tags=req.entity_a.tags,
    )
    e_b = Entity(
        id=req.entity_b.id,
        text=req.entity_b.text,
        entity_type=req.entity_b.entity_type,
        tags=req.entity_b.tags,
    )
    v_a = engine.embed(e_a)
    v_b = engine.embed(e_b)
    sim = cosine_similarity(v_a, v_b)
    threshold = 0.35
    return SimilarityResponse(
        similarity=round(sim, 4),
        is_match=sim >= threshold,
        threshold=threshold,
    )


@app.post("/evaluate")
def run_evaluation(req: Optional[EvaluationRequest] = None):
    threshold = req.similarity_threshold if req else 0.35
    dataset = get_default_benchmark_dataset()
    result = evaluate_embeddings(engine, dataset, similarity_threshold=threshold)
    return result
