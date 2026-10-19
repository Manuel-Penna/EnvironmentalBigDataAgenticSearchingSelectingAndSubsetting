"""Run with:  pytest -q test_qdrant_catalog.py

Uses the project's real QdrantRepository / QdrantQueryEngine against Qdrant's
in-memory mode (no server needed) and a fake embedder (no Ollama needed).
Skipped automatically if qdrant-client or those modules are unavailable.
"""
import hashlib
import re

import pytest

pytest.importorskip("qdrant_client")
qr = pytest.importorskip("qdrant_repository")
pytest.importorskip("qdrant_query_engine")

from qdrant_client import QdrantClient  # noqa: E402

from agent_tools import ToolContext, build_registry  # noqa: E402
from catalog_adapters import QdrantCatalog, _time_bound  # noqa: E402
from metadata_dataclasses import (DatasetMetadata, SpatialMetadata,  # noqa: E402
                                  TemporalMetadata, VariableMetadata,
                                  VariableStatistics)


class FakeEmbedder:
    """Bag-of-words hashing embedder: shared words -> high cosine similarity."""
    vector_size = 128

    def embed(self, text):
        v = [0.0] * self.vector_size
        for w in re.findall(r"[a-z]+", text.lower()):
            if len(w) > 2:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.vector_size] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]


def make_md(did, title, fmt, spatial, temporal, var, long_name):
    return DatasetMetadata(
        dataset_id=did, source=f"/data/{did}.zarr", format=fmt, is_remote=False,
        size_bytes=None, global_attributes={"title": title, "summary": f"{title} data"},
        dimensions={"time": 10, "lat": 5, "lon": 5},
        spatial=SpatialMetadata(*spatial), temporal=TemporalMetadata(*temporal),
        variables=[VariableMetadata(
            name=var, dtype="float32", dimensions=["time", "lat", "lon"], shape=[10, 5, 5],
            attributes={"long_name": long_name, "units": "K"}, chunks=None,
            statistics=VariableStatistics(mean=1.0, std=1.0, min=0.0, max=2.0,
                                          median=1.0, count=240, nan_count=10))],
        warnings=[])


NS = ""  # seconds precision; see test_nanosecond_timestamps for the 9-digit case
DATASETS = [
    make_md("med", "Mediterranean sea surface temperature", "zarr",
            (30, 46, -6, 36), (f"2000-01-01T00:00:00{NS}", f"2001-12-31T00:00:00{NS}"),
            "sst", "sea surface temperature"),
    make_md("pac", "Pacific ocean wind speed", "zarr",
            (-10, 10, 150, 180), (f"2000-01-01T12:00:00{NS}", f"2000-02-19T00:00:00{NS}"),
            "wspd", "wind speed"),
    make_md("glob", "global ocean sea surface temperature reanalysis", "netcdf",
            (-90, 90, -180, 180), (f"2010-01-01T00:00:00{NS}", f"2020-12-31T00:00:00{NS}"),
            "analysed_sst", "analysed sea surface temperature"),
    make_md("notime", "Mediterranean bathymetry elevation", "zarr",
            (30, 46, -6, 36), (None, None), "elevation", "sea floor depth"),
]


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(qr, "QdrantClient", lambda host=None, port=None: QdrantClient(":memory:"))
    emb = FakeEmbedder()
    repo = qr.QdrantRepository(embedder=emb)
    for md in DATASETS:
        repo.save_metadata(md)
    return QdrantCatalog(repo, emb, score_threshold=0.3)


def ids(hits):
    return {h["dataset_id"] for h in hits}


# ---------------------------------------------------------------- semantic
def test_semantic_query_finds_by_topic(catalog):
    got = ids(catalog.search("sea surface temperature", limit=10))
    assert {"med", "glob"} <= got and "pac" not in got


def test_variable_level_match(catalog):
    assert "pac" in ids(catalog.search("wind speed", limit=10))


def test_empty_query_means_filters_only(catalog):
    assert ids(catalog.search(None, limit=10)) == {"med", "pac", "glob", "notime"}
    assert ids(catalog.search("   ", format="netcdf")) == {"glob"}


# ----------------------------------------------------------------- spatial
def test_bbox_intersection(catalog):
    assert ids(catalog.search(None, bbox=(30, 40, 0, 10), limit=10)) == {"med", "glob", "notime"}


def test_bbox_crossing_antimeridian_is_split(catalog):
    got = ids(catalog.search(None, bbox=(-5, 5, 170, -170), limit=10))
    assert got == {"pac", "glob"}  # raw single-range filter would return only 'glob'


def test_bbox_in_0_360_convention_is_normalised(catalog):
    assert ids(catalog.search(None, bbox=(-5, 5, 190, 200), limit=10)) == {"glob"}


# ---------------------------------------------------------------- temporal
def test_time_filters_and_datasets_without_time_are_excluded(catalog):
    assert ids(catalog.search(None, time=("2005-01-01", None), limit=10)) == {"glob"}
    assert ids(catalog.search(None, time=(None, "2001-01-01"), limit=10)) == {"med", "pac"}


def test_date_only_end_is_inclusive(catalog):
    # pac starts at 12:00 on 2000-01-01: a bare-date end (midnight) would miss it
    assert "pac" in ids(catalog.search(None, time=(None, "2000-01-01"), limit=10))
    assert "med" in ids(catalog.search(None, time=("2001-12-31", "2001-12-31"), limit=10))


@pytest.mark.xfail(strict=False, reason=(
    "In-memory Qdrant silently excludes datetimes with 9 fractional digits, which is "
    "what str(numpy datetime64[ns]) produces. A real server may accept them: verify "
    "against yours, or truncate to seconds at ingestion."))
def test_nanosecond_timestamps(catalog, monkeypatch):
    md = make_md("ns", "Nanosecond dataset", "zarr", (0, 1, 0, 1),
                 ("2000-01-01T00:00:00.000000000", "2000-12-31T00:00:00.000000000"),
                 "x", "x")
    catalog._repo.save_metadata(md)
    assert "ns" in ids(catalog.search(None, time=("2000-06-01", "2000-06-30"), limit=10))


# ---------------------------------------------------------------- combined
def test_query_plus_filters(catalog):
    got = ids(catalog.search("sea surface temperature", format="zarr",
                             bbox=(30, 40, 0, 10), time=("2000-06-01", "2000-06-30"), limit=10))
    assert got == {"med"}


def test_limit_is_respected(catalog):
    assert len(catalog.search(None, limit=2)) == 2


# --------------------------------------------------------------------- get
def test_get_returns_full_record(catalog):
    rec = catalog.get("med")
    assert rec["source"] == "/data/med.zarr" and rec["variables"][0]["statistics"]["max"] == 2.0
    assert "lat_min" not in rec and "type" not in rec  # flat filter fields stripped
    assert catalog.get("nope") is None


# ------------------------------------------------------------- time helper
@pytest.mark.parametrize("value, end, expected", [
    ("2020-12-31", True, "2020-12-31T23:59:59Z"),
    ("2020-12-31", False, "2020-12-31T00:00:00Z"),
    ("2020", True, "2020-12-31T23:59:59Z"),
    ("2020-06", False, "2020-06-01T00:00:00Z"),
    ("2020-06-01T08:30:00", True, "2020-06-01T08:30:00"),
    (None, True, None),
])
def test_time_bound(value, end, expected):
    assert _time_bound(value, end) == expected


# ---------------------------------------------------- through the agent tools
def test_agent_tools_over_qdrant_catalog(catalog, tmp_path):
    reg = build_registry(ToolContext(catalog=catalog, output_dir=tmp_path / "o"))
    r = reg.call("search_catalog", {"query": "sea surface temperature", "format": "zarr",
                                    "time": {"start": "2000-06-01", "end": "2000-06-30"}})
    assert r["ok"] and [h["dataset_id"] for h in r["results"]] == ["med"]
    assert r["results"][0]["variables"] == ["sst (sea surface temperature, K)"]
    d = reg.call("get_dataset", {"dataset_id": "med"})
    assert d["ok"] and d["variables"][0]["stats"]["nan_fraction"] == round(10 / 250, 4)
