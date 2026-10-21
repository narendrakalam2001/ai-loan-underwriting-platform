# ============================================================
# RUN DASHBOARD — runner script
# ============================================================
# Usage:
#   python scripts/run_dashboard.py            # ops monitoring dashboard
#   python scripts/run_dashboard.py --portal   # user-facing portal instead
# ============================================================

import sys
import os
import argparse
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Streamlit app for the underwriting platform")
    parser.add_argument("--portal", action="store_true",
                         help="Run the User Portal (frontend/app.py) instead of the ops dashboard")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    target = "frontend/app.py" if args.portal else "monitoring/dashboard.py"
    port = args.port or (8502 if args.portal else 8501)

    cmd = [
        sys.executable, "-m", "streamlit", "run", target,
        "--server.port", str(port), "--server.address", "0.0.0.0",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT)
