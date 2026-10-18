"""
qdrant_repository.py
--------------------

Each dataset is stored as a Qdrant point:
    - id:      integer derived from the dataset_id hash
    - vector:  embedding of the dataset's textual metadata
    - payload: the full DatasetMetadata as a flat JSON dict,
               with spatial and temporal fields promoted to the
               top level for efficient Qdrant filtering.

Usage:
    embedder = OllamaEmbedder()
    repo     = QdrantRepository(embedder=embedder)
    repo.save_metadata(metadata)
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from qdrant_client import QdrantClient
from datetime import datetime, timezone

from metadata_dataclasses import DatasetMetadata
from embeddings import (
    Embedder, EmbeddingDocument,
    build_all_documents, build_dataset_document,
    make_point_id
)
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    FilterSelector, PointStruct, UpdateStatus,
    Distance, VectorParams, PayloadSchemaType,
    PointIdsList, Range, DatetimeRange
)

# Name of the Qdrant collection
COLLECTION_NAME = "datasets"


# ============================================================
# REPOSITORY
# ============================================================
class QdrantRepository:

    def __init__(
        self,
        embedder: Embedder,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "datasets",
    ):
        self._embedder   = embedder
        self._collection = collection_name
        self._client     = QdrantClient(host=host, port=port)
        self._ensure_collection()

    # --------------------------------------------------------
    # SAVE — upserts dataset point + all variable points
    # --------------------------------------------------------

    def save_metadata(self, metadata: DatasetMetadata) -> str:
        # Delete all existing points for this dataset first
        # (handles the case where variables changed since last ingestion)
        self._delete_by_dataset_id(metadata.dataset_id)

        documents = build_all_documents(metadata)
        points = []
        for doc in documents:
            vector  = self._embedder.embed(doc.text)
            payload = self._build_payload(doc, metadata)
            points.append(PointStruct(
                id=doc.point_id,
                vector=vector,
                payload=payload,
            ))

        result = self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

        if result.status == UpdateStatus.COMPLETED:
            n_vars = len(documents) - 1
            return (f"Dataset '{metadata.dataset_id}' saved "
                    f"({n_vars} variable embeddings).")
        raise RuntimeError(f"Qdrant upsert failed: {result.status}")

    # --------------------------------------------------------
    # DELETE — removes dataset point + all variable points
    # --------------------------------------------------------

    def delete(self, dataset_id: str) -> bool:
        # Check if the dataset point exists
        pid = make_point_id(f"dataset:{dataset_id}")
        existing = self._client.retrieve(
            collection_name=self._collection,
            ids=[pid],
            with_payload=False,
            with_vectors=False,
        )
        if not existing:
            return False
        self._delete_by_dataset_id(dataset_id)
        return True

    def _delete_by_dataset_id(self, dataset_id: str) -> None:
        """Delete all points (dataset + variables) for a given dataset_id."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(
                        key="dataset_id",
                        match=MatchValue(value=dataset_id),
                    )
                ])
            ),
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    def get_by_id(self, dataset_id: str) -> dict | None:
        pid = make_point_id(f"dataset:{dataset_id}")
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[pid],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return self._restore_dataset_payload(results[0].payload)

    def list_all(
        self,
        fmt: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return all dataset-level documents (type='dataset')."""
        must = [FieldCondition(key="type", match=MatchValue(value="dataset"))]
        if fmt:
            must.append(
                FieldCondition(key="format", match=MatchValue(value=fmt.lower()))
            )
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._restore_dataset_payload(p.payload) for p in points]

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search_datasets(
        self,
        query_vector: list[float],
        qdrant_filter: Filter | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Semantic search against dataset-level embeddings.
        Adds type='dataset' to whatever filter is passed.
        """
        type_condition = FieldCondition(
            key="type", match=MatchValue(value="dataset")
        )
        combined = self._combine_filter(qdrant_filter, type_condition)

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=combined,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [self._restore_dataset_payload(r.payload) for r in results]

    def search_variables(
        self,
        query_vector: list[float],
        qdrant_filter: Filter | None = None,
        limit: int = 50,
        score_threshold: float = 0.0,
    ) -> list[str]:
        """
        Semantic search against variable-level embeddings.
        Returns a list of dataset_ids whose variables matched.
        """
        type_condition = FieldCondition(
            key="type", match=MatchValue(value="variable")
        )
        combined = self._combine_filter(qdrant_filter, type_condition)

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=combined,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        # Return unique dataset_ids that had a matching variable
        seen = set()
        dataset_ids = []
        for r in results:
            did = r.payload.get("dataset_id")
            if did and did not in seen:
                seen.add(did)
                dataset_ids.append(did)
        return dataset_ids

    def get_dataset_full_payloads(
        self, dataset_ids: list[str]
    ) -> list[dict]:
        """Fetch full dataset payloads by a list of dataset_ids."""
        point_ids = [
            make_point_id(f"dataset:{did}") for did in dataset_ids
        ]
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return [self._restore_dataset_payload(r.payload) for r in results]

    def get_all_variable_names(self) -> list[str]:
        """Return all unique variable names across the catalog."""
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="variable"))
            ]),
            limit=100000,
            with_payload=True,
            with_vectors=False,
        )
        return list({
            p.payload["variable_name"]
            for p in points
            if p.payload.get("variable_name")
        })

    # --------------------------------------------------------
    # PAYLOAD HELPERS
    # --------------------------------------------------------

    def _build_payload(
        self,
        doc: EmbeddingDocument,
        metadata: DatasetMetadata,
    ) -> dict:
        """
        Dataset points get the full metadata payload.
        Variable points get a lightweight payload with just
        the fields needed for filtering and retrieval.
        """
        if doc.doc_type == "dataset":
            from dataclasses import asdict
            d = asdict(metadata)
            sp = d.get("spatial") or {}
            tp = d.get("temporal") or {}
            d["type"]           = "dataset"
            d["lat_min"]        = sp.get("lat_min")
            d["lat_max"]        = sp.get("lat_max")
            d["lon_min"]        = sp.get("lon_min")
            d["lon_max"]        = sp.get("lon_max")
            d["temporal_start"] = tp.get("start")
            d["temporal_end"]   = tp.get("end")
            return d
        else:
            # Variable point — lightweight, just enough for filtering
            # and for resolving back to the parent dataset
            return {
                "type":          "variable",
                "dataset_id":    doc.dataset_id,
                "variable_name": doc.variable_name,
                # Promote spatial/temporal from parent for pre-filtering
                # variable searches by region/time
                "lat_min":        self._get_spatial(metadata, "lat_min"),
                "lat_max":        self._get_spatial(metadata, "lat_max"),
                "lon_min":        self._get_spatial(metadata, "lon_min"),
                "lon_max":        self._get_spatial(metadata, "lon_max"),
                "temporal_start": metadata.temporal.start,
                "temporal_end":   metadata.temporal.end,
                "format":         metadata.format,
            }

    def _get_spatial(self, metadata: DatasetMetadata, field: str):
        return getattr(metadata.spatial, field, None)

    def _restore_dataset_payload(self, payload: dict) -> dict:
        flat_fields = ("lat_min", "lat_max", "lon_min", "lon_max",
                       "temporal_start", "temporal_end", "type")
        return {k: v for k, v in payload.items() if k not in flat_fields}

    def _combine_filter(
        self, existing: Filter | None, extra: FieldCondition
    ) -> Filter:
        must = [extra]
        if existing and existing.must:
            must.extend(existing.must)
        return Filter(must=must)

    # --------------------------------------------------------
    # COLLECTION SETUP
    # --------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedder.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        for field, schema in {
            "type":           PayloadSchemaType.KEYWORD,
            "dataset_id":     PayloadSchemaType.KEYWORD,
            "variable_name":  PayloadSchemaType.KEYWORD,
            "format":         PayloadSchemaType.KEYWORD,
            "temporal_start": PayloadSchemaType.DATETIME,
            "temporal_end":   PayloadSchemaType.DATETIME,
            "lat_min":        PayloadSchemaType.FLOAT,
            "lat_max":        PayloadSchemaType.FLOAT,
            "lon_min":        PayloadSchemaType.FLOAT,
            "lon_max":        PayloadSchemaType.FLOAT,
        }.items():
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=schema,
            )

def _build_filter(
    fmt: str | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    temporal_start: str | None = None,
    temporal_end: str | None = None,
) -> Filter | None:
    """Build a Qdrant Filter from structured parameters."""
    conditions = []

    if fmt:
        conditions.append(
            FieldCondition(key="format", match=MatchValue(value=fmt.lower()))
        )

    ts_start = temporal_start
    ts_end   = temporal_end

    if ts_end is not None:
        conditions.append(
            FieldCondition(key="temporal_start", range=DatetimeRange(lte=ts_end))
        )
    if ts_start is not None:
        conditions.append(
            FieldCondition(key="temporal_end", range=DatetimeRange(gte=ts_start))
        )

    # Spatial intersection (range query on 4 scalar fields)
    if lat_max is not None:
        conditions.append(FieldCondition(key="lat_min", range=Range(lte=lat_max)))
    if lat_min is not None:
        conditions.append(FieldCondition(key="lat_max", range=Range(gte=lat_min)))
    if lon_max is not None:
        conditions.append(FieldCondition(key="lon_min", range=Range(lte=lon_max)))
    if lon_min is not None:
        conditions.append(FieldCondition(key="lon_max", range=Range(gte=lon_min)))

    return Filter(must=conditions) if conditions else None

def _date_to_timestamp(date_str: str | None) -> float | None:
    """
    Convert an ISO 8601 date/datetime string to a Unix timestamp.
    Returns None if the input is None or cannot be parsed.
    """
    if not date_str:
        return None
    # Try progressively shorter formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",  # numpy datetime64 default
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(date_str[:len(fmt)], fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None