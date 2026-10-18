"""
app/services/nlq_qdrant.py
--------------------------
Initializes the NL query engine for Qdrant and runs queries.
Used by app/routes/query.py.
"""

from __future__ import annotations

from typing import Any

from qdrant_repository import QdrantRepository
from embeddings import Embedder
from nl_query_engine_qdrant import (
    AnthropicBackend,
    NaturalLanguageQdrantEngine,
    OllamaBackend,
)


def run_natural_query(
    repo: QdrantRepository,
    embedder: Embedder,
    query_text: str,
    config: dict,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build the NL query engine from config and run the query.
    Returns (results, filters_extracted_by_llm).
    """
    backend_name = config.get("LLM_BACKEND", "ollama").lower()

    if backend_name == "anthropic":
        backend = AnthropicBackend(model=config.get("ANTHROPIC_MODEL"))
    else:
        backend = OllamaBackend(
            model=config.get("OLLAMA_MODEL", "llama3"),
            host=config.get("OLLAMA_HOST", "http://localhost:11434"),
        )

    nlqe = NaturalLanguageQdrantEngine(
        repo=repo,
        backend=backend,
        embedder=embedder,
        bbox_padding=config.get("NLQ_BBOX_PADDING", 0.5),
        limit=config.get("NLQ_LIMIT", 10),
        score_threshold=config.get("NLQ_SCORE_THRESHOLD", 0.6),
    )

    results = nlqe.query(query_text)
    return results, nlqe._last_filters
