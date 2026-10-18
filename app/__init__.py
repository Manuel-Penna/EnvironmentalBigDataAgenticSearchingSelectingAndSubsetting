"""
app/__init__.py
---------------
Flask application factory — Qdrant version.
"""

from __future__ import annotations

from flask import Flask

from app.config import Config
from embeddings import OllamaEmbedder
from qdrant_repository import QdrantRepository


def create_app(config: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)

    # --- Embedder and Qdrant repository ---
    embedder = OllamaEmbedder(
        model=config.OLLAMA_EMBED_MODEL,
        host=config.OLLAMA_HOST,
    )
    app.embedder   = embedder
    app.qdrant_repo = QdrantRepository(
        embedder=embedder,
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        collection_name=config.QDRANT_COLLECTION,
    )

    # --- Register blueprints ---
    from app.routes.datasets import datasets_bp
    from app.routes.query    import query_bp
    from app.routes.frontend import frontend_bp

    app.register_blueprint(datasets_bp, url_prefix="/datasets")
    app.register_blueprint(query_bp,    url_prefix="/query")
    app.register_blueprint(frontend_bp)

    # --- Health check ---
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
