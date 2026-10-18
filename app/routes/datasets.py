"""
app/routes/datasets.py  (Qdrant version)
-----------------------------------------
CRUD endpoints — same interface as before, backed by QdrantRepository.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from extractors import NetCDFExtractor, ZarrExtractor

datasets_bp = Blueprint("datasets", __name__)

_extractors = {
    "netcdf": NetCDFExtractor(),
    "zarr":   ZarrExtractor(),
}


# ── POST /datasets/extract ────────────────────────────────────

@datasets_bp.post("/extract")
def extract():
    """
    Extract metadata from a file and save it to Qdrant.

    Body (JSON):
        { "source": "/path/to/file.nc", "format": "netcdf" | "zarr" }

    Returns:
        201  { "message": "...", "dataset_id": "..." }
        400  { "error": "..." }
        500  { "error": "..." }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    source = body.get("source")
    fmt    = body.get("format")

    if not source:
        return jsonify({"error": "Field 'source' is required."}), 400
    if fmt not in _extractors:
        return jsonify({"error": "Field 'format' must be 'netcdf' or 'zarr'."}), 400

    try:
        metadata = _extractors[fmt].extract(source)
        message  = current_app.qdrant_repo.save_metadata(metadata)
        return jsonify({"message": message, "dataset_id": metadata.dataset_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── GET /datasets/ ────────────────────────────────────────────

@datasets_bp.get("/")
def list_datasets():
    """
    List all datasets with optional filters.

    Query params (optional): format, variable, limit
    """
    fmt      = request.args.get("format")
    limit    = int(request.args.get("limit", 100))

    try:
        results = current_app.qdrant_repo.list_all(
            fmt=fmt, limit=limit
        )
        return jsonify({"count": len(results), "datasets": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── GET /datasets/<dataset_id> ────────────────────────────────

@datasets_bp.get("/<dataset_id>")
def get_dataset(dataset_id: str):
    try:
        doc = current_app.qdrant_repo.get_by_id(dataset_id)
        if doc is None:
            return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404
        return jsonify(doc), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DELETE /datasets/<dataset_id> ─────────────────────────────

@datasets_bp.delete("/<dataset_id>")
def remove_dataset(dataset_id: str):
    try:
        deleted = current_app.qdrant_repo.delete(dataset_id)
        if not deleted:
            return jsonify({"error": f"Dataset '{dataset_id}' not found."}), 404
        return jsonify({"message": f"Dataset '{dataset_id}' deleted."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
