"""
qdrant_query_engine.py
----------------------
Replaces DatasetQueryEngine (MongoDB) with a Qdrant-backed query engine.

Supports two query modes:

1. STRUCTURED — filter-only, no vector search.
   Equivalent to the old DatasetQueryEngine.
   Use when the caller provides explicit filters (format, spatial, temporal).

2. SEMANTIC — vector search with optional structured pre-filtering.
   Use when a natural language query is available.
   The query is embedded and compared against dataset vectors.

The two modes can be combined: semantic search with structured pre-filters
narrows the candidate set before ranking by similarity.

Usage:
    # Structured only
    qe = QdrantQueryEngine(repo)
    results = (qe
        .filter_by_format("netcdf")
        .filter_spatial_intersects(30, 46, -6, 36)
        .filter_temporal_overlap("2020-01-01", "2020-12-31")
        .execute())

    # Semantic with pre-filters
    results = (qe
        .filter_by_format("netcdf")
        .semantic_search("sea surface temperature", embedder, limit=10))
"""

from __future__ import annotations

from typing import Any

from qdrant_repository import QdrantRepository, _build_filter
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embeddings import Embedder


class QdrantQueryEngine:
    """
    Chainable query builder for the Qdrant dataset collection.

    Mirrors the interface of the old DatasetQueryEngine so that
    route handlers require minimal changes.
    """

    def __init__(self, repo: QdrantRepository):
        self._repo        = repo
        self._fmt:        str | None   = None
        self._lat_min:    float | None = None
        self._lat_max:    float | None = None
        self._lon_min:    float | None = None
        self._lon_max:    float | None = None
        self._temp_start: str | None   = None
        self._temp_end:   str | None   = None
        self._variable:   str | None   = None
        self._variables:  list[str] | None = None

    def reset(self) -> QdrantQueryEngine:
        self.__init__(self._repo)
        return self

    # --------------------------------------------------------
    # FILTER METHODS (chainable)
    # --------------------------------------------------------

    def filter_by_format(self, fmt: str) -> QdrantQueryEngine:
        self._fmt = fmt.lower()
        return self

    def filter_by_variable(self, var_name: str) -> QdrantQueryEngine:
        """Exact variable name match (applied post-retrieval)."""
        self._variable = var_name
        return self

    def filter_by_variables(self, var_names: list[str]) -> QdrantQueryEngine:
        """Match any of the given variable names (applied post-retrieval)."""
        self._variables = var_names
        return self

    def filter_temporal_overlap(
        self, start: str, end: str
    ) -> QdrantQueryEngine:
        if start is not None:
            self._temp_start = str(start)
        if end is not None:
            self._temp_end   = str(end)
        return self

    def filter_spatial_intersects(
        self,
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
    ) -> QdrantQueryEngine:
        """Datasets whose bounding box intersects the given bbox."""
        self._lat_min = min_lat
        self._lat_max = max_lat
        self._lon_min = min_lon
        self._lon_max = max_lon
        return self

    def filter_spatial_contains_point(
        self, lat: float, lon: float
    ) -> QdrantQueryEngine:
        """Datasets whose bounding box contains the given point."""
        # Point containment is bbox intersection where min==max
        self._lat_min = lat
        self._lat_max = lat
        self._lon_min = lon
        self._lon_max = lon
        return self

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def execute(self, limit: int = 100) -> list[dict[str, Any]]:
        qdrant_filter = _build_filter(
            fmt=self._fmt,
            lat_min=self._lat_min,
            lat_max=self._lat_max,
            lon_min=self._lon_min,
            lon_max=self._lon_max,
            temporal_start=self._temp_start,
            temporal_end=self._temp_end,
        )

        # Always filter by type="dataset" to exclude variable points
        type_condition = FieldCondition(key="type", match=MatchValue(value="dataset"))
        if qdrant_filter:
            must = [type_condition] + list(qdrant_filter.must or [])
        else:
            must = [type_condition]

        points, _ = self._repo._client.scroll(
            collection_name=self._repo._collection,
            scroll_filter=Filter(must=must),
            limit=limit or 100,
            with_payload=True,
            with_vectors=False,
        )
        results = [self._repo._restore_dataset_payload(p.payload) for p in points]

        return self._apply_variable_filter(results)


    def semantic_search(
        self,
        query_text: str,
        embedder: Embedder,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Two-stage semantic search:
        1. Search dataset-level embeddings (what is the dataset about?)
        2. Search variable-level embeddings (what does it contain?)
        Merge results, deduplicate by dataset_id, fetch full payloads.
        """
        query_vector = embedder.embed(query_text)

        structured_filter = _build_filter(
            fmt=self._fmt,
            lat_min=self._lat_min,
            lat_max=self._lat_max,
            lon_min=self._lon_min,
            lon_max=self._lon_max,
            temporal_start=self._temp_start,
            temporal_end=self._temp_end,
        )

        # Stage 1 — dataset-level search
        dataset_results = self._repo.search_datasets(
            query_vector=query_vector,
            qdrant_filter=structured_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Stage 2 — variable-level search
        matched_variable_dataset_ids = self._repo.search_variables(
            query_vector=query_vector,
            qdrant_filter=structured_filter,
            limit=limit * 5,   # cast a wider net at variable level
            score_threshold=score_threshold,
        )

        # Merge: datasets from stage 1 first (ranked by similarity),
        # then any additional datasets found only through variables
        seen = {d["dataset_id"] for d in dataset_results}
        extra_ids = [
            did for did in matched_variable_dataset_ids
            if did not in seen
        ]

        extra_results = (
            self._repo.get_dataset_full_payloads(extra_ids)
            if extra_ids else []
        )

        merged = dataset_results + extra_results
        return self._apply_variable_filter(merged)[:limit]

    # --------------------------------------------------------
    # PRIVATE
    # --------------------------------------------------------

    def _apply_variable_filter(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply variable name filtering in Python.
        Qdrant doesn't efficiently support array-element matching
        on nested lists without a dedicated keyword index per element.
        """
        if self._variable:
            results = [
                r for r in results
                if any(v["name"] == self._variable
                       for v in r.get("variables", []))
            ]
        elif self._variables:
            var_set = set(self._variables)
            results = [
                r for r in results
                if any(v["name"] in var_set
                       for v in r.get("variables", []))
            ]
        return results

