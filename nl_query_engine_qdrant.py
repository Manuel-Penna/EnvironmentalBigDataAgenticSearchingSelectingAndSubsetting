"""
nl_query_engine_qdrant.py
-------------------------
Natural language query engine backed by Qdrant.

Key difference from the MongoDB version:
- Variable matching is handled by semantic (vector) search, not by
  LLM-generated alias lists. The LLM only extracts structured filters
  (format, spatial bbox, temporal range).
- The user's full query is embedded and used for semantic search,
  with structured filters applied as Qdrant pre-filters.

Usage:
    embedder = OllamaEmbedder(model="nomic-embed-text")
    repo     = QdrantRepository(embedder=embedder)
    backend  = OllamaBackend(model="llama3")

    nlqe    = NaturalLanguageQdrantEngine(repo, backend, embedder)
    results = nlqe.query("NetCDF datasets with sea surface temperature
                          over the Mediterranean in 2020")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from qdrant_query_engine import QdrantQueryEngine
from qdrant_repository import QdrantRepository
from embeddings import Embedder
from datetime import date
from dateutil.relativedelta import relativedelta

# ============================================================
# SYSTEM PROMPT
# ============================================================

# Note: no "variables" field — variable matching is now done via
# semantic search, so the LLM only needs to extract structured filters.
BASE_SYSTEM_PROMPT = """
You are an assistant that extracts structured search filters from natural
language queries about geospatial scientific datasets (NetCDF and Zarr).

ALWAYS respond with a valid JSON object only, with no additional text,
no backticks, and no markdown.

Required schema:
{
  "format":   "netcdf" | "zarr" | null,
  "temporal": 
    {"type": "year",     "year": 2026}
  | {"type": "relative", "amount": 5, "unit": "years" | "months" | "days"}
  | {"type": "range",    "start": "YYYY-MM-DD" | null, "end": "YYYY-MM-DD" | null}
  | {"type": "before", "end": "YYYY-MM-DD"}
  | {"type": "after", "start": "YYYY-MM-DD"}
  | null,
  "spatial": {
    "type": "bbox",
    "min_lat": float, "max_lat": float,
    "min_lon": float, "max_lon": float
  } | {
    "type": "point",
    "lat": float, "lon": float
  } | null
}

Rules:
- If the user mentions a geographic region (e.g. "Mediterranean", "Italy",
  "North Pole"), convert it to approximate coordinates in the spatial field.
- If a filter is not specified, set the corresponding field to null.
- Do not specify the format unless the user explicitly mentions one.
- YYYY-MM-DD is ISO 8601 format.
- Variable names are handled separately by semantic search — do NOT
  include a variable field in your response.
"""


# ============================================================
# LLM BACKEND (same interface as before)
# ============================================================

class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        pass


class OllamaBackend(LLMBackend):
    """
    Backend for local models via Ollama.

    Requirements:
        pip install ollama
        ollama pull llama3
    """

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")
        self._model = model
        self._host  = host

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self._ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            format="json",
            options={"temperature": 0},
        )
        return response["message"]["content"]


class AnthropicBackend(LLMBackend):
    """
    Backend for Claude via the Anthropic API.

    Requirements:
        pip install anthropic
        export ANTHROPIC_API_KEY="sk-ant-..."
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            from anthropic import Anthropic
            self._client = Anthropic()
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        self._model = model

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


# ============================================================
# NATURAL LANGUAGE QUERY ENGINE (Qdrant)
# ============================================================

class NaturalLanguageQdrantEngine:
    """
    Translates natural language queries into Qdrant searches.

    The LLM extracts only structured filters (format, spatial, temporal).
    Variable/content matching is handled by semantic vector search,
    which is more robust than keyword alias expansion.

    Args:
        repo:         QdrantRepository instance.
        backend:      LLMBackend instance (Ollama or Anthropic).
        embedder:     Embedder instance for query vectorization.
        bbox_padding: Degrees added to LLM-generated bboxes (default 0.5).
        limit:        Max results per query (default 10).
        score_threshold: Minimum cosine similarity to include a result (default 0.3).
    """

    def __init__(
        self,
        repo: QdrantRepository,
        backend: LLMBackend,
        embedder: Embedder,
        bbox_padding: float = 0.5,
        limit: int = 10,
        score_threshold: float = 0.3,
    ):
        self._repo            = repo
        self._backend         = backend
        self._embedder        = embedder
        self._bbox_padding    = bbox_padding
        self._limit           = limit
        self._score_threshold = score_threshold
        self._last_filters: dict[str, Any] = {}

    def query(self, natural_language: str) -> list[dict[str, Any]]:
        """
        Run a natural language query.

        1. LLM extracts structured filters from the query text.
        2. The full query text is embedded.
        3. Qdrant semantic search is run with structured pre-filters.

        Returns a list of matching dataset payloads ordered by similarity.
        """
        filters = self._extract_filters(natural_language)
        self._last_filters = filters

        qe = QdrantQueryEngine(self._repo)
        self._apply_filters(qe, filters)

        self.print_debug() #debugging
        return qe.semantic_search(
            query_text=natural_language,
            embedder=self._embedder,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

    def print_debug(self) -> None:
        print("Filters extracted by the LLM:")
        print(json.dumps(self._last_filters, indent=2, ensure_ascii=False))

    # --------------------------------------------------------
    # PRIVATE
    # --------------------------------------------------------

    def _extract_filters(self, text: str) -> dict[str, Any]:
        system_prompt = BASE_SYSTEM_PROMPT
        user_message  = (
            f"{text}\n\n"
            "Respond with a JSON object only. No explanation, no text, just the JSON."
        )

        raw = self._backend.complete(system_prompt, user_message).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise RuntimeError(f"No JSON found in LLM response:\n{raw}")

        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in LLM response: {e}\n---\n{raw}")

    def _apply_filters(
        self, qe: QdrantQueryEngine, filters: dict[str, Any]
    ) -> None:
        if filters.get("format"):
            qe.filter_by_format(filters["format"])

        if filters.get("temporal"):
            t = filters["temporal"]
            t_start, t_end = self._resolve_temporal(t)
            if t_start or t_end:
                qe.filter_temporal_overlap(t_start, t_end)

        if filters.get("spatial"):
            s = filters["spatial"]
            p = self._bbox_padding
            if s.get("type") == "bbox":
                qe.filter_spatial_intersects(
                    min_lat=s["min_lat"] - p, max_lat=s["max_lat"] + p,
                    min_lon=s["min_lon"] - p, max_lon=s["max_lon"] + p,
                )
            elif s.get("type") == "point":
                qe.filter_spatial_intersects(
                    min_lat=s["lat"] - p, max_lat=s["lat"] + p,
                    min_lon=s["lon"] - p, max_lon=s["lon"] + p,
                )


    def _resolve_temporal(self, t: dict | None) -> tuple[str | None, str | None]:
        if not t:
            return None, None
        today = date.today()
        kind = t.get("type")

        if kind == "year" and t.get("year"):
            y = int(t["year"])
            return f"{y}-01-01", f"{y}-12-31"

        if kind == "relative" and t.get("amount") and t.get("unit") in ("years", "months", "days"):
            start = today - relativedelta(**{t["unit"]: int(t["amount"])})
            return start.isoformat(), today.isoformat()

        if kind == "range":
            return t.get("start"), t.get("end")

        if kind == "before":
            return None, t.get("end")

        if kind == "after":
            return t.get("start"), None

        return None, None