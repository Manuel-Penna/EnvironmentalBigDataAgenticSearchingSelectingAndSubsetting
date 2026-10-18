"""
run.py
------
Entry point for the Flask development server.

Usage:
    python3 run.py
"""

from app import create_app
from app.config import Config

app = create_app(Config)

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
    )
