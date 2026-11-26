import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

TOPIC_KEYWORDS = {
    "gaming": ["game", "gaming", "fps", "monitor", "144hz", "165hz", "keyboard", "switches", "hardware", "screen", "response", "tactile"],
    "cloud": ["kubernetes", "k8s", "kafka", "pipeline", "cluster", "microservices", "streaming", "devops", "cloud", "deployments", "monitoring", "data"],
    "finance": ["etf", "investing", "finance", "portfolio", "retirement", "funds", "bonds", "yield", "liquidity", "savings", "commission"],
    "food": ["sourdough", "bread", "baking", "starter", "fermentation", "yeast", "flour", "coffee", "roast", "beans"],
    "fitness": ["marathon", "running", "fitness", "pacing", "nutrition", "gel", "training"],
    "automotive": ["suspension", "sway", "car", "mechanic", "automotive", "commuter"],
}


@dataclass
class Entity:
    """Multimodal entity representation (post, creative ad, comment, or topic)."""
    id: str
    text: str
    entity_type: str = "post"  # post, ad, comment, topic
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalizes text by removing markup, normalizing URLs, and lowercasing."""
    if not text:
        return ""
    t = re.sub(r"https?://\S+", " url_token ", text)
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RepresentationEngine:
    """Feature-hashed semantic embedding generator with topic calibration."""

    def __init__(self, dim: int = 64, calibrated: bool = True):
        self.dim = dim
        self.calibrated = calibrated

    def embed(self, entity: Entity) -> List[float]:
        """Extracts a normalized dense semantic representation vector."""
        clean = normalize_text(entity.text)
        words = clean.split()
        all_tokens = words + [t.lower() for t in entity.tags]
        vec = [0.0] * self.dim

        if not all_tokens:
            return vec

        # Dedicated topic cluster space when calibrated
        if self.calibrated:
            for topic_idx, (topic, kws) in enumerate(TOPIC_KEYWORDS.items()):
                score = sum(3.0 for t in all_tokens if any(kw in t for kw in kws))
                for d in range(4):
                    idx = topic_idx * 4 + d
                    vec[idx] = score * (1.0 if d % 2 == 0 else 0.8)

        # Token hashing across remaining dimensions
        rem_dim = self.dim - (24 if self.calibrated else 0)
        offset = 24 if self.calibrated else 0
        type_mult = 1.5 if entity.entity_type == "ad" else 1.0

        for t in all_tokens:
            h = int(hashlib.md5(f"tok:{t}".encode("utf-8")).hexdigest()[:8], 16) % rem_dim
            vec[offset + h] += 1.0 * type_mult

        # Bounded activation calibration
        if self.calibrated:
            for i in range(self.dim):
                vec[i] = math.tanh(vec[i] * 0.4)

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec
