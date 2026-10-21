# ============================================================
# UNDERWRITING API — FastAPI Serving Layer
# ============================================================
# Endpoints:
#   POST /underwrite         ← main endpoint (Modules 1+2+3 → 4)
#   POST /upload-document     ← bank statement PDF upload/parse
#   GET  /health
#   GET  /model-info
#   GET  /application/{id}    ← audit trail lookup
#   POST /chat                ← Module 5D loan assistant (bounded in-memory sessions)
#   POST /explain             ← Module 5B explanation (Gemini or template)
#   GET  /application/{id}/letter?language=en|hi  ← Module 5C decision-letter PDF
# ============================================================

import os
import io
import time
import uuid
import json
import re
import logging
import threading
from collections import OrderedDict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal

from src.config import CREDIT_MODEL_SUBDIR, FRAUD_MODEL_SUBDIR, SYNTHETIC_STATEMENTS_DIR
from src.model_loader import load_latest_model
from src.document_parser import parse_bank_statement
from src.decision_engine import make_final_decision, load_audit_trail
from src.evaluation import explain_single_prediction
from src.chatbot import LoanAssistantChatbot
from src.llm_explainer import generate_explanation
from src.doc_generator import generate_decision_letter
from services.prediction_service import (
    predict_credit, predict_fraud, prepare_credit_features,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Loan Underwriting Platform API",
    description="Combines Credit Risk Scoring, Fraud Detection and Document "
                 "Intelligence into one underwriting decision.",
    version="1.0.0",
)

DOCUMENT_CACHE_DIR = os.path.join("data", "document_cache")
os.makedirs(DOCUMENT_CACHE_DIR, exist_ok=True)

# ── Load both models on startup ───────────────────────────────
try:
    credit_model, credit_threshold = load_latest_model(CREDIT_MODEL_SUBDIR)
    logger.info("Credit risk model loaded successfully")
except Exception as e:
    logger.error("Credit model loading failed: %s", e)
    credit_model, credit_threshold = None, 0.5

try:
    fraud_model, fraud_threshold = load_latest_model(FRAUD_MODEL_SUBDIR)
    logger.info("Fraud model loaded successfully")
except Exception as e:
    logger.error("Fraud model loading failed: %s", e)
    fraud_model, fraud_threshold = None, 0.5


# ============================================================
# INPUT SCHEMAS
# ============================================================

class ApplicantInput(BaseModel):
    # ── Personal details ──────────────────────────────────────
    full_name:      Optional[str] = None
    age:            float
    employment_type: Optional[str] = "Salaried"
    years_employment: Optional[float] = 2.0

    # ── Financial details (feed the Credit Risk model — Kaggle's "Give Me
    #    Some Credit" dataset: a genuine 90+ day delinquency label, not the
    #    marketing-offer-acceptance label the platform originally shipped
    #    with). `monthly_income` is an ACTUAL ₹ amount, not '000s — the old
    #    schema's ambiguous unit was itself a source of bugs. ─────────────
    monthly_income:  float                 # actual ₹ per month
    dependents:      int = 0
    existing_emis:   float = 0.0           # ₹/month — also drives debt_ratio
    credit_utilization_pct: float = 20.0   # % of available credit limit in use (can exceed 100)
    open_credit_lines: int = 2             # open credit cards / loans
    real_estate_loans: int = 0             # mortgages / property-backed loans
    times_30_59_days_late: int = 0         # self-declared late-payment history
    times_60_89_days_late: int = 0         # (a production system would pull
    times_90_days_late:    int = 0         #  these from a credit bureau)

    # ── Loan request ───────────────────────────────────────────
    loan_amount_requested: float = 500000.0
    loan_purpose: Optional[str] = "Personal"
    tenure_months: int = 36

    # ── Optional link to a previously uploaded bank statement ──
    document_id: Optional[str] = None


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Loan Underwriting Platform API is live 🚀",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "credit_model_loaded": credit_model is not None,
        "fraud_model_loaded":  fraud_model is not None,
    }


@app.get("/model-info")
def model_info():
    info = {}
    for name, model_dir in (("credit_risk", CREDIT_MODEL_SUBDIR), ("fraud", FRAUD_MODEL_SUBDIR)):
        registry_path = os.path.join(model_dir, "latest_model.json")
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                info[name] = json.load(f)
        else:
            info[name] = {"error": "Model registry not found"}
    return info


# ── Document upload endpoint (Module 3) ─────────────────────────

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a bank-statement PDF, parses it with document_parser.py,
    caches the extraction under a document_id, and returns both.
    The User Portal calls this before /underwrite and passes the
    returned document_id through.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    document_id = f"DOC{uuid.uuid4().hex[:10].upper()}"
    temp_path = os.path.join(DOCUMENT_CACHE_DIR, f"{document_id}.pdf")

    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        result = parse_bank_statement(temp_path)
    except Exception as e:
        logger.exception("Document parsing failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not parse document: {e}")

    result_dict = result.to_dict()
    result_dict["document_id"] = document_id

    with open(os.path.join(DOCUMENT_CACHE_DIR, f"{document_id}.json"), "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info("Document parsed | id=%s | txns=%d | warnings=%d",
                document_id, result.n_transactions_parsed, len(result.warnings))

    return result_dict


def _load_cached_document(document_id: str) -> Optional[dict]:
    path = os.path.join(DOCUMENT_CACHE_DIR, f"{document_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Main underwriting endpoint (Modules 1 + 2 + 3 → 4) ──────────

@app.post("/underwrite")
def underwrite(applicant: ApplicantInput):
    if credit_model is None or fraud_model is None:
        raise HTTPException(status_code=503, detail="One or more models failed to load")

    start = time.time()
    input_data = applicant.model_dump() if hasattr(applicant, "model_dump") else applicant.dict()

    # ── Module 1 — Credit Risk ────────────────────────────────
    credit_result = predict_credit(credit_model, input_data, credit_threshold)

    # ── Module 2 — Fraud Detection (checked before credit is
    #    finalized, per the platform's decision hierarchy) ─────
    fraud_input = dict(input_data)
    fraud_input["Amount"] = input_data["loan_amount_requested"]
    fraud_result = predict_fraud(fraud_model, fraud_input, fraud_threshold)

    # ── Module 3 — Document Intelligence (optional) ────────────
    doc_result = _load_cached_document(applicant.document_id) if applicant.document_id else None
    if applicant.document_id and doc_result is None:
        logger.warning("document_id '%s' not found in cache — proceeding without it", applicant.document_id)

    # ── SHAP top drivers for the explanation ──────────────────
    try:
        df = prepare_credit_features(input_data)
        shap_pairs = explain_single_prediction(credit_model, df, df, top_k=5)
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        shap_pairs = []

    # ── Module 4 — Unified Decision ────────────────────────────
    decision = make_final_decision(
        applicant_id=None,
        credit_result=credit_result,
        fraud_result=fraud_result,
        doc_result=doc_result,
        shap_pairs=shap_pairs,
        application=input_data,      # decision_engine keeps only name/amount/tenure/purpose
    )

    decision["latency_seconds"] = round(time.time() - start, 4)
    return decision


@app.get("/application/{application_id}")
def get_application(application_id: str):
    trail = load_audit_trail(application_id)
    if trail is None:
        raise HTTPException(status_code=404, detail=f"No audit trail found for '{application_id}'")
    return trail


# ============================================================
# MODULE 5 — chat, explanation, decision letter
# ============================================================

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

MAX_CHAT_SESSIONS = 500
_chat_sessions: "OrderedDict[str, LoanAssistantChatbot]" = OrderedDict()
_chat_lock = threading.Lock()


def _get_chatbot(session_id: str) -> LoanAssistantChatbot:
    """Bounded in-memory session store: at most MAX_CHAT_SESSIONS bots, oldest evicted first
    (least-recently-used, since every request moves its session to the end). Memory is lost on
    restart and is per-process — fine for a demo, not for multi-worker production."""
    with _chat_lock:
        bot = _chat_sessions.get(session_id)
        if bot is None:
            bot = LoanAssistantChatbot()
            _chat_sessions[session_id] = bot
            while len(_chat_sessions) > MAX_CHAT_SESSIONS:
                _chat_sessions.popitem(last=False)
        else:
            _chat_sessions.move_to_end(session_id)
        return bot


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., max_length=2000)    # the bot itself rejects > 500 chars politely
    language: Literal["en", "hi"] = "en"


class ExplainRequest(BaseModel):
    application_id: str
    language: Literal["en", "hi"] = "en"


def _trail_or_error(application_id: str) -> dict:
    if not _ID_RE.match(application_id or ""):
        raise HTTPException(status_code=400, detail="Invalid application ID format")
    trail = load_audit_trail(application_id)
    if trail is None:
        raise HTTPException(status_code=404, detail=f"No audit trail found for '{application_id}'")
    return trail


@app.post("/chat")
def chat(req: ChatRequest):
    if not _ID_RE.match(req.session_id):
        raise HTTPException(status_code=400, detail="session_id may only contain letters, digits, '_' and '-'")
    bot = _get_chatbot(req.session_id)
    try:
        return bot.chat(req.message, req.language)
    except Exception as e:                       # the chatbot is designed not to raise; belt and braces
        logger.exception("Chatbot failure: %s", e)
        raise HTTPException(status_code=500, detail="The assistant hit an internal error")


@app.post("/explain")
def explain(req: ExplainRequest):
    trail = _trail_or_error(req.application_id)
    return generate_explanation(trail, trail.get("application"), req.language)


@app.get("/application/{application_id}/letter")
def get_decision_letter(application_id: str, language: Literal["en", "hi"] = Query("en")):
    trail = _trail_or_error(application_id)
    letter = generate_decision_letter(trail, trail.get("application"), language)
    headers = {
        "Content-Disposition": f'attachment; filename="{application_id}_decision_letter.pdf"',
        "X-Language-Used": letter["language_used"],      # "en" if a Hindi font was unavailable
        "X-Text-Source": letter["text_source"],
    }
    return Response(content=letter["pdf"], media_type="application/pdf", headers=headers)