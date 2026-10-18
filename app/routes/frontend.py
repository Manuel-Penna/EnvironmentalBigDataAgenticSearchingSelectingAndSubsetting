"""
app/routes/frontend.py
----------------------
Serves the single-page frontend application.
"""

from flask import Blueprint, render_template

frontend_bp = Blueprint("frontend", __name__)


@frontend_bp.get("/")
def index():
    return render_template("index.html")
