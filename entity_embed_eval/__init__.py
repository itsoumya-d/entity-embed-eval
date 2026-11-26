"""Multimodal Entity Representation and Embedding Evaluation Engine."""

from .core import Entity, RepresentationEngine, cosine_similarity
from .evaluation import RelevancePair, BenchmarkResult, evaluate_embeddings

__all__ = [
    "Entity",
    "RepresentationEngine",
    "cosine_similarity",
    "RelevancePair",
    "BenchmarkResult",
    "evaluate_embeddings",
]
