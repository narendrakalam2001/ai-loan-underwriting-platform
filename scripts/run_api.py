# ============================================================
# RUN API — runner script
# ============================================================
# Usage:
#   python scripts/run_api.py
#   python scripts/run_api.py --port 8080 --reload
# ============================================================

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)   # so relative model/log paths resolve correctly

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the underwriting platform's FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("serving.api:app", host=args.host, port=args.port, reload=args.reload)
