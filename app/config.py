"""
app/config.py
-------------
Central configuration — Qdrant version.
"""

from __future__ import annotations
import os


class Config:
    # Qdrant
    QDRANT_HOST       = os.getenv("QDRANT_HOST",       "localhost")
    QDRANT_PORT       = int(os.getenv("QDRANT_PORT",   "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "datasets")

    # Ollama
    OLLAMA_HOST        = os.getenv("OLLAMA_HOST",        "http://localhost:11434")
    OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "llama3")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Anthropic (optional cloud LLM)
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # LLM backend for NL queries: "ollama" or "anthropic"
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

    # NL query settings
    NLQ_BBOX_PADDING    = float(os.getenv("NLQ_BBOX_PADDING",    "0.5"))
    NLQ_LIMIT           = int(os.getenv("NLQ_LIMIT",             "10"))
    NLQ_SCORE_THRESHOLD = float(os.getenv("NLQ_SCORE_THRESHOLD", "0.6"))

    # Flask
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST  = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT  = int(os.getenv("FLASK_PORT", "5000"))