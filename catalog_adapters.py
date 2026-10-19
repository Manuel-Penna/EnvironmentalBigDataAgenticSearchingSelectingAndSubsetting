"""
Catalog access for the agent tools.

The tools never talk to Qdrant directly. They talk to a CatalogAdapter, which
has two methods and returns plain dicts shaped like `asdict(DatasetMetadata)`:

    {"dataset_id", "source", "format", "is_remote", "size_bytes",
     "global_attributes": {...}, "dimensions": {...},
     "spatial":  {"lat_min", "lat_max", "lon_min", "lon_max"},
     "temporal": {"start", "end"},
     "variables": [{"name", "dtype", "dimensions", "shape", "attributes",
                    "chunks", "statistics": {"min","max","mean","count","nan_count",...}}],
     "warnings": [...]}

Search results additionally carry "score" (float | None) and
"matched_via" ("dataset" | "variable").

Why an adapter: the tools (and their tests) don't depend on Qdrant being up,
and the same tools can be benchmarked against different catalogs.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class CatalogAdapter(Protocol):
    def search(self, query: str | None, *, format: str | None = None,
               bbox: tuple[float, float, float, float] | None = None,
               time: tuple[str | None, str | None] | None = None,
               limit: int = 5) -> list[dict]:
        """Datasets matching the semantic `query` AND the structured filters.

        bbox  = (lat_min, lat_max, lon_min, lon_max): dataset must INTERSECT it.
        time  = (start, end), either may be None: dataset period must OVERLAP it.
        query may be None: filters only.
        """
        ...

    def get(self, dataset_id: str) -> dict | None:
        """Full record for one dataset, or None."""
        ...


# --------------------------------------------------------------------------
# In-memory catalog: for tests and development without Qdrant
# --------------------------------------------------------------------------
def _norm_lon(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def _lon_intervals(lon_min: float, lon_max: float) -> list[tuple[float, float]]:
    """Split a west->east longitude range into intervals inside [-180, 180]."""
    span = lon_max - lon_min
    if span < 0:
        span += 360.0
    if span >= 360.0:
        return [(-180.0, 180.0)]
    a = _norm_lon(lon_min)
    b = a + span
    return [(a, b)] if b <= 180.0 else [(a, 180.0), (-180.0, b - 360.0)]


def _bbox_intersects(sp: dict, bbox: tuple[float, float, float, float]) -> bool:
    if any(sp.get(k) is None for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
        return False  # no spatial metadata -> cannot match a spatial filter
    lat_min, lat_max, lon_min, lon_max = bbox
    if sp["lat_max"] < lat_min or sp["lat_min"] > lat_max:
        return False
    ds_iv = (sp["lon_min"], sp["lon_max"])
    return any(not (ds_iv[1] < a or ds_iv[0] > b)
               for a, b in _lon_intervals(lon_min, lon_max))


def _time_overlaps(tp: dict, time: tuple[str | None, str | None]) -> bool:
    if not tp.get("start") or not tp.get("end"):
        return False  # no temporal metadata -> excluded when a time filter is active
    start, end = time
    ds_start, ds_end = pd.Timestamp(tp["start"]), pd.Timestamp(tp["end"])
    if start is not None and ds_end < pd.Timestamp(start):
        return False
    if end is not None and ds_start > pd.Timestamp(end):
        return False
    return True


class InMemoryCatalog:
    """Naive keyword-overlap catalog over a list of record dicts."""

    def __init__(self, records: list[dict]):
        self._records = {r["dataset_id"]: r for r in records}

    def get(self, dataset_id: str) -> dict | None:
        return self._records.get(dataset_id)

    def search(self, query, *, format=None, bbox=None, time=None, limit=5):
        tokens = {t for t in (query or "").lower().replace(",", " ").split() if len(t) > 2}
        hits = []
        for rec in self._records.values():
            if format and rec.get("format") != format:
                continue
            if bbox and not _bbox_intersects(rec.get("spatial") or {}, bbox):
                continue
            if time and not _time_overlaps(rec.get("temporal") or {}, time):
                continue
            if not tokens:
                hits.append({**rec, "score": None, "matched_via": "dataset"})
                continue
            ds_text = " ".join(str(v) for v in (rec.get("global_attributes") or {}).values()).lower()
            var_text = " ".join(
                " ".join(str(x) for x in [v["name"], *(v.get("attributes") or {}).values()])
                for v in rec.get("variables", [])).lower()
            ds_score = sum(t in ds_text for t in tokens) / len(tokens)
            var_score = sum(t in var_text for t in tokens) / len(tokens)
            score = max(ds_score, var_score)
            if score > 0:
                hits.append({**rec, "score": round(score, 3),
                             "matched_via": "dataset" if ds_score >= var_score else "variable"})
        hits.sort(key=lambda r: r["score"] or 0, reverse=True)
        return hits[:limit]


# --------------------------------------------------------------------------
# Qdrant catalog: adapter over QdrantQueryEngine / QdrantRepository
# --------------------------------------------------------------------------
def _time_bound(value: str | None, end: bool) -> str | None:
    """Turn a date-only bound into a full RFC 3339 instant so it behaves inclusively.

    "2020-12-31" as an END becomes 2020-12-31T23:59:59Z (a bare date means
    midnight at the START of that day, which would miss datasets starting later
    that day). As a START it becomes T00:00:00Z. Bounds that already carry a
    time are passed through unchanged. "2020" / "2020-06" are expanded to the
    whole year / month.
    """
    if value is None:
        return None
    s = str(value).strip()
    if "T" in s or " " in s:
        return s
    try:
        p = pd.Period(s)
        t = p.end_time if end else p.start_time
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: BLE001
        return s


class QdrantCatalog:
    """CatalogAdapter over the existing Qdrant layer.

    repo      QdrantRepository
    embedder  the same embedder used at ingestion (never mix models)
    score_threshold  similarity cut-off for semantic search; default is the
                     NLQ_SCORE_THRESHOLD environment variable, else 0.6

    Differences from calling QdrantQueryEngine directly:
      * a fresh engine per query (the engine accumulates filter state);
      * the request bbox is split into intervals inside [-180, 180] so boxes
        crossing the antimeridian and requests in 0-360 convention work (the
        engine's single-range longitude filter would silently miss them);
      * date-only time bounds are made inclusive (see _time_bound);
      * an empty query means "filters only" (engine.execute()).

    Not available from the engine: similarity scores. Records are returned as
    the engine produced them; "score" / "matched_via" are simply absent.
    """

    def __init__(self, repo: Any, embedder: Any, score_threshold: float | None = None):
        import os
        self._repo = repo
        self._embedder = embedder
        self._score_threshold = (float(os.environ.get("NLQ_SCORE_THRESHOLD", "0.6"))
                                 if score_threshold is None else score_threshold)

    def search(self, query, *, format=None, bbox=None, time=None, limit=5):
        import logging
        from qdrant_query_engine import QdrantQueryEngine  # lazy: keeps this module importable without Qdrant

        log = logging.getLogger(__name__)
        intervals: list[tuple[float, float, float, float] | None] = [None]
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            intervals = [(lat_min, lat_max, a, b) for a, b in _lon_intervals(lon_min, lon_max)]
        start, end = (None, None) if not time else (_time_bound(time[0], False),
                                                    _time_bound(time[1], True))
        text = (query or "").strip()

        merged: list[dict] = []
        seen: set[str] = set()
        for iv in intervals:
            engine = QdrantQueryEngine(self._repo)
            if format:
                engine.filter_by_format(format)
            if iv:
                engine.filter_spatial_intersects(*iv)
            if start is not None or end is not None:
                engine.filter_temporal_overlap(start, end)
            log.debug("catalog search: query=%r format=%r bbox_interval=%r time=%r",
                      text, format, iv, (start, end))
            if text:
                hits = engine.semantic_search(text, self._embedder, limit=limit,
                                              score_threshold=self._score_threshold)
            else:
                hits = engine.execute(limit=limit)
            for h in hits:
                if h["dataset_id"] not in seen:
                    seen.add(h["dataset_id"])
                    merged.append(h)
        return merged[:limit]

    def get(self, dataset_id):
        return self._repo.get_by_id(dataset_id)
