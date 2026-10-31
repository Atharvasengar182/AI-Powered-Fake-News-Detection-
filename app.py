"""TruthLens AI - Root Streamlit Entrypoint.

Delegates execution to frontend/app.py so deployment works smoothly
whether configured with root 'app.py' or 'frontend/app.py'.
"""

from __future__ import annotations

from pathlib import Path
import runpy

FRONTEND_APP = Path(__file__).resolve().parent / "frontend" / "app.py"

if __name__ == "__main__" or True:
    runpy.run_path(str(FRONTEND_APP), run_name="__main__")
