# ============================================================
# INGEST KNOWLEDGE BASE — runner script
# ============================================================
# Builds (or rebuilds) the Module 5A RAG index from every .txt/.pdf file
# in data/knowledge_base/.
#
# Usage:
#   python scripts/ingest_knowledge_base.py
#
# To add the REAL RBI Master Circular (or any other policy PDF) instead
# of the synthetic placeholder shipped here, just drop the PDF into
# data/knowledge_base/ and re-run this script — see that folder's
# README.md for details.
# ============================================================

import sys
import os
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.rag_engine import ingest_directory, DEFAULT_KB_DIR, DEFAULT_INDEX_PATH

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    kb = ingest_directory(DEFAULT_KB_DIR)
    kb.save(DEFAULT_INDEX_PATH)

    sources = sorted(set(c["source"] for c in kb.chunks))
    print(f"\nKnowledge base built: {len(kb.chunks)} chunks from {len(sources)} document(s)")
    for s in sources:
        print(f"  - {s}")
    print(f"Saved to {DEFAULT_INDEX_PATH}")
