"""
embeddings.py
-------------
Embedding backends for GeoMeta.
Produces float vectors from text for storage in Qdrant.

Supported backends:
    OllamaEmbedder   — local models via Ollama (nomic-embed-text, mxbai-embed-large, ...)
    AnthropicEmbedder — via Anthropic (voyage-3 via voyageai, or stub for future support)

Usage:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    vector   = embedder.embed("sea surface temperature Mediterranean")
"""

from __future__ import annotations

import hashlib

from abc import ABC, abstractmethod

from dataclasses import dataclass

from metadata_dataclasses import DatasetMetadata, VariableMetadata



# ============================================================
# ABSTRACT INTERFACE
# ============================================================

class Embedder(ABC):
    """
    Interface that every embedding backend must implement.
    Returns a list of floats (the embedding vector) for a given text.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of the vectors produced by this embedder."""
        pass


# ============================================================
# BACKEND: OLLAMA (local)
# ============================================================

class OllamaEmbedder(Embedder):
    """
    Embedding backend using a locally running Ollama server.

    Recommended models:
        nomic-embed-text    274M params, 768-dim, fast
        mxbai-embed-large   670M params, 1024-dim, better quality

    Requirements:
        pip install ollama
        ollama pull nomic-embed-text
    """

    # Known vector sizes for common models.
    # If the model is not listed, the size is determined by a test call at init.
    _KNOWN_SIZES = {
        "nomic-embed-text":  768,
        "mxbai-embed-large": 1024,
        "all-minilm":        384,
    }

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")

        self._model = model
        self._host  = host
        self._size  = self._KNOWN_SIZES.get(model) or self._probe_size()

    def embed(self, text: str) -> list[float]:
        response = self._ollama.embeddings(model=self._model, prompt=text)
        return response["embedding"]

    @property
    def vector_size(self) -> int:
        return self._size

    def _probe_size(self) -> int:
        """Determine vector size by embedding a short test string."""
        return len(self.embed("test"))


# ============================================================
# METADATA TEXT BUILDER
# ============================================================


def make_point_id(logical_key: str) -> int:
    """
    Deterministic integer ID from a logical key string.
    Examples:
        make_point_id("dataset:abc123")
        make_point_id("variable:abc123:sst")
    """
    return int(hashlib.sha256(logical_key.encode()).hexdigest()[:16], 16)



MAX_DESC_CHARS = 4000


@dataclass
class EmbeddingDocument:
    """A single embeddable unit of text with its Qdrant point metadata."""
    point_id:      int
    text:          str
    doc_type:      str          # "dataset" or "variable"
    dataset_id:    str
    variable_name: str | None = None


def build_dataset_document(metadata: DatasetMetadata) -> EmbeddingDocument:
    """
    Dataset-level embedding: answers "what is this dataset about?"
    Includes title, description, source, institution — not variables.
    """
    parts = []

    #skip adding format to parts, as it should be filtered by LLM filters
    #and it causes false positives 
    #ex.: "find netcdf datasets containing sea surface temperature" would likely return all netcdf datasets, because the semantic search on dataset-level gives a high score

    for attr in ("title", "summary", "description", "comment",
                 "source", "institution"):
        val = metadata.global_attributes.get(attr)
        if isinstance(val, str) and val.strip():
            text = val.strip()
            if attr == "description" and len(text) > MAX_DESC_CHARS:
                text = text[:MAX_DESC_CHARS]
            parts.append(f"{attr}: {text}")

    return EmbeddingDocument(
        point_id=make_point_id(f"dataset:{metadata.dataset_id}"),
        text="\n".join(parts) or "unknown dataset",
        doc_type="dataset",
        dataset_id=metadata.dataset_id,
    )


def build_variable_documents(metadata: DatasetMetadata) -> list[EmbeddingDocument]:
    """
    Variable-level embeddings: answers "what does this dataset contain?"
    One document per variable.
    """
    docs = []
    for var in metadata.variables:
        parts = [f"Variable: {var.name}"]
        for cf_key in ("long_name", "standard_name"):
            val = var.attributes.get(cf_key)
            if isinstance(val, str) and val.strip():
                parts.append(f"{cf_key}: {val.strip()}")
        if "units" in var.attributes:
            parts.append(f"units: {var.attributes['units']}")

        docs.append(EmbeddingDocument(
            point_id=make_point_id(f"variable:{metadata.dataset_id}:{var.name}"),
            text="\n".join(parts),
            doc_type="variable",
            dataset_id=metadata.dataset_id,
            variable_name=var.name,
        ))
    return docs


def build_all_documents(metadata: DatasetMetadata) -> list[EmbeddingDocument]:
    return [build_dataset_document(metadata)] + build_variable_documents(metadata)