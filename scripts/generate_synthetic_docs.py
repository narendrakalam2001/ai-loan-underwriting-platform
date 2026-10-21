# ============================================================
# GENERATE SYNTHETIC BANK STATEMENTS — runner script
# ============================================================
# Usage:
#   python scripts/generate_synthetic_docs.py            # 1000 statements (config default)
#   python scripts/generate_synthetic_docs.py --n 50      # custom count
# ============================================================

import sys
import os
import argparse
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.synthetic_data_gen import generate_synthetic_statements
from src.config import SYNTHETIC_STATEMENT_COUNT, SYNTHETIC_STATEMENTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic bank statement PDFs")
    parser.add_argument("--n", type=int, default=SYNTHETIC_STATEMENT_COUNT,
                         help=f"Number of statements to generate (default: {SYNTHETIC_STATEMENT_COUNT})")
    parser.add_argument("--out-dir", type=str, default=SYNTHETIC_STATEMENTS_DIR)
    args = parser.parse_args()

    ids = generate_synthetic_statements(n=args.n, out_dir=args.out_dir)
    print(f"\nGenerated {len(ids)} synthetic bank statements → {args.out_dir}")
