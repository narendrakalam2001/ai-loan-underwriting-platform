# ============================================================
# DOCKERFILE — AI Loan Underwriting Platform (API)
# ============================================================
# Multi-stage build:
#   Stage 1 (builder) — install all Python dependencies
#   Stage 2 (runtime) — lean final image
#
# Module 5: the knowledge base (data/knowledge_base/) is copied in and the
# TF-IDF index is BUILT AT IMAGE BUILD TIME, so the chatbot's FAQ tool works
# on first start. Set GEMINI_API_KEY at runtime (never bake it into the image);
# without it the platform uses its offline template/rules paths.
# ============================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# build-essential + libgomp1 needed by XGBoost / LightGBM / CatBoost;
# mupdf deps needed by PyMuPDF's wheel at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt && \
    python -m spacy download en_core_web_sm || true


# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source (only what the API needs at runtime)
COPY src/          ./src/
COPY serving/      ./serving/
COPY services/     ./services/
COPY models/       ./models/
COPY data/raw/     ./data/raw/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY scripts/ingest_knowledge_base.py ./scripts/ingest_knowledge_base.py

# logs/ is created here instead of COPY'd: an empty logs/ folder is not tracked by git,
# so `COPY logs/` breaks builds from a fresh clone.
RUN mkdir -p data/document_cache logs/audit_trail \
    && python scripts/ingest_knowledge_base.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
