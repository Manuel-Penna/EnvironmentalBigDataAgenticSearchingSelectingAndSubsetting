"""
app/routes/query.py  (Qdrant version)
--------------------------------------
Structured and natural language query endpoints backed by Qdrant.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from app.services.nlq_qdrant import run_natural_query
from qdrant_query_engine import QdrantQueryEngine

query_bp = Blueprint("query", __name__)


# ── POST /query/structured ────────────────────────────────────

@query_bp.post("/structured")
def structured_query():
    """
    Query with explicit filters. No vector search.

    Body (JSON) — all optional:
        {
            "format":   "netcdf" | "zarr",
            "variable": "temperature",
            "temporal": { "start": "2020-01-01", "end": "2020-12-31" },
            "spatial":  { "type": "bbox",
                          "min_lat": 30, "max_lat": 46,
                          "min_lon": -6, "max_lon": 36 },
            "limit":    20
        }
    """
    body  = request.get_json(silent=True) or {}
    limit = int(body.get("limit", 100))

    try:
        qe = QdrantQueryEngine(current_app.qdrant_repo)

        if body.get("format"):
            qe.filter_by_format(body["format"])

        if body.get("variable"):
            qe.filter_by_variable(body["variable"])

        if body.get("temporal"):
            t = body["temporal"]
            if t.get("start") and t.get("end"):
                qe.filter_temporal_overlap(t["start"], t["end"])

        if body.get("spatial"):
            s = body["spatial"]
            if s.get("type") == "bbox":
                qe.filter_spatial_intersects(
                    min_lat=s["min_lat"], max_lat=s["max_lat"],
                    min_lon=s["min_lon"], max_lon=s["max_lon"],
                )
            elif s.get("type") == "point":
                qe.filter_spatial_contains_point(
                    lat=s["lat"], lon=s["lon"]
                )

        results = qe.execute(limit=limit)
        return jsonify({"count": len(results), "datasets": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── POST /query/natural ───────────────────────────────────────

@query_bp.post("/natural")
def natural_query():
    """
    Natural language query — LLM extracts structured filters,
    Qdrant semantic search handles variable/content matching.

    Body (JSON):
        { "query": "NetCDF datasets with SST over the Mediterranean in 2020" }

    Returns:
        {
            "count":          N,
            "datasets":       [...],
            "filters":        {...},   (structured filters from LLM)
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    query_text = body.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "Field 'query' is required."}), 400

    try:
        results, filters = run_natural_query(
            repo=current_app.qdrant_repo,
            embedder=current_app.embedder,
            query_text=query_text,
            config=current_app.config,
        )
        return jsonify({
            "count":   len(results),
            "datasets": results,
            "filters":  filters,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
