# ============================================================
# CONFIGURATION — AI Loan Underwriting Platform
# ============================================================
# Central config for all 4 modules:
#   Module 1 — Credit Risk Scoring Engine
#   Module 2 — Fraud Detection Engine
#   Module 3 — Document Intelligence Parser
#   Module 4 — Unified Decision Engine
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS       = -1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS            = 5
RANDOM_SEARCH_ITERS = 20

# ── Feature selection ────────────────────────────────────────
SELECT_K     = 10
SELECT_K_MAX = 20

# ── Outlier clipping (shared Clipper transformer) ────────────
CLIP_FOLD = 1.5

# ── Feature engineering thresholds ───────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 10

# ============================================================
# MODULE 1 — CREDIT RISK ENGINE
# ============================================================

# ── Risk bands (probability of default → tier) ───────────────
RISK_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

# ── Business rule thresholds ─────────────────────────────────
MAX_INCOME_RATIO_RULE = 10.0
MIN_INCOME_RULE       = 5000    # ₹/month — compared against the applicant's REAL ₹ income

# ── Income scale: ₹ (portal/API input) → model units (GMSC training data) ─────────────
# The credit model is trained on Kaggle "Give Me Some Credit", whose MonthlyIncome is
# dollar-scale (median ~5,400; p10 ~2,000; p90 ~11,700). The portal collects ₹. Feeding
# ₹45,000 straight in makes an ordinary salary look like a top-decile income to the model.
#
#   model_income = rupee_income / INCOME_INR_PER_MODEL_UNIT
#
# DEFAULT 5.0 = "income-rank matching": it assumes a typical Indian monthly income of about
# ₹27,000 corresponds to GMSC's median of ~5,400 (so ₹45,000 -> 9,000, upper-middle).
# THAT IS AN ASSUMPTION, not a measured fact — replace it with your own figure. Reference points:
#       ~5      income-rank matching (default, assumes median Indian income ~₹27k)
#       ~20-22  PPP exchange rate (₹ per US$, purchasing-power basis) -> ₹45k = ~2,100 (GMSC p10)
#       ~83-88  market exchange rate -> almost every Indian salary lands below GMSC p10
# Override without editing code:  set INCOME_INR_PER_MODEL_UNIT=21
# Changing it needs NO retraining: the model is trained in GMSC units; only live input is scaled.
INCOME_INR_PER_MODEL_UNIT = float(os.getenv("INCOME_INR_PER_MODEL_UNIT", "5.0"))
if INCOME_INR_PER_MODEL_UNIT <= 0:
    raise ValueError("INCOME_INR_PER_MODEL_UNIT must be > 0")

# The same ₹5,000 floor expressed in the model's units (used by credit_engine, whose rows are
# in model units both in training/evaluation and — after scaling — at serving time).
MIN_INCOME_RULE_MODEL_UNITS = MIN_INCOME_RULE / INCOME_INR_PER_MODEL_UNIT
FAMILY_SIZE_RULE      = 5

# NOTE: These replaced the old Bank-Personal-Loan-Modelling-era rules
# when the credit dataset was swapped to Kaggle's "Give Me Some Credit" —
# a genuine default-risk dataset (target = SeriousDlqin2yrs, real 90+ day
# delinquency) instead of a marketing-offer-acceptance dataset. See
# credit_engine.py. MIN_INCOME_RULE above is reused with a new (₹/month,
# not ₹'000) value; MAX_INCOME_RATIO_RULE / FAMILY_SIZE_RULE above are now
# unused by credit_engine.py but left defined in case other code references
# them.
HIGH_DEBT_RATIO_RULE       = 1.0     # DebtRatio > 100% of income → DECLINE_RULE (hard overleverage)
SEVERE_DELINQUENCY_RULE    = 3       # 3+ instances of 90-day-late payments → DECLINE_RULE
HIGH_DEPENDENTS_RULE       = 6       # NumberOfDependents >= this → REVIEW_DEPENDENTS

# ── 3-tier underwriting decision thresholds ──────────────────
# Applied to the *calibrated* default-probability score.
CREDIT_APPROVE_MAX        = 0.30   # prob < this  → APPROVE
CREDIT_MANUAL_REVIEW_MAX  = 0.60   # this <= prob < DECLINE → MANUAL_REVIEW
                                    # prob >= CREDIT_MANUAL_REVIEW_MAX → DECLINE

# ============================================================
# MODULE 2 — FRAUD DETECTION ENGINE
# ============================================================

FRAUD_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

HIGH_AMOUNT_RULE  = 5000     # Amount > this → BLOCK_RULE (hard rule) — tuned for CARD-TRANSACTION amounts
LOW_AMOUNT_RULE   = 10       # Amount < this → potential card-testing flag (card-transaction context)
REVIEW_PROB_RATIO = 0.6      # prob >= threshold * this ratio → REVIEW

# Loan-underwriting context uses a DIFFERENT amount scale than card
# transactions (₹ lakhs, not ₹ hundreds) — see fraud_engine.score_transaction()
# and services/prediction_service.predict_fraud() for how these are applied
# instead of HIGH_AMOUNT_RULE/LOW_AMOUNT_RULE when scoring a loan application.
LOAN_HIGH_AMOUNT_RULE = 5_000_000   # > ₹50 lakh requested → flag for manual review of the ask itself
LOAN_LOW_AMOUNT_RULE  = 5_000       # < ₹5,000 requested → implausibly small, possible test/spam application

# ============================================================
# SHARED — DRIFT / MONITORING / CHALLENGER GATES
# ============================================================

# ── PSI drift thresholds ─────────────────────────────────────
PSI_MODERATE = 0.10   # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH     = 0.20   # PSI >= 0.20 → critical drift, retrain

# ── Score monitoring alert ───────────────────────────────────
SCORE_MEAN_ALERT = 0.35

# ── Champion-Challenger promotion gates (3-gate system) ──────
MIN_F1_IMPROVEMENT     = 0.005
MIN_ROCAUC_THRESHOLD   = 0.95   # fallback floor for an engine with no entry below
# Gate 2 is PER ENGINE. A single 0.95 floor suits fraud (ROC-AUC ~0.95) but would
# reject every credit challenger forever: Give-Me-Some-Credit models top out
# around 0.86, so after the first auto-promoted model no credit retrain could win.
# Keys are the project labels used by the training pipeline.
MIN_ROCAUC_BY_ENGINE = {"credit_risk": 0.80, "fraud": 0.90}
MAX_GENERALIZATION_GAP = 0.10

# ============================================================
# MODULE 3 — DOCUMENT INTELLIGENCE PARSER
# ============================================================

# ── Synthetic bank statement generation ──────────────────────
SYNTHETIC_STATEMENT_COUNT   = 1000
SYNTHETIC_STATEMENT_MONTHS  = 6          # months of history per statement
SYNTHETIC_SEED              = RANDOM_STATE

# ── Employer / salary detection regex keywords ───────────────
SALARY_KEYWORDS = ["SALARY", "SAL CREDIT", "PAYROLL", "SALARY CREDIT"]
EMI_KEYWORDS    = ["EMI", "LOAN INSTALLMENT", "LOAN EMI", "AUTO DEBIT-EMI", "NACH EMI"]

# ── Account stability classification (coefficient of variation
#    of month-end balances → Good / Moderate / Poor) ─────────
BALANCE_STABILITY_GOOD_CV     = 0.25   # CV <= this        → "Good"
BALANCE_STABILITY_MODERATE_CV = 0.50   # this < CV <= 0.50 → "Moderate"
                                        # CV > 0.50         → "Poor"

# ============================================================
# MODULE 4 — UNIFIED DECISION ENGINE
# ============================================================

# ── Cost-sensitive business impact (₹) ───────────────────────
COST_WRONG_APPROVAL   = 45000   # avg loss when a bad loan is approved
COST_WRONG_REJECTION  = 1200    # opportunity cost of rejecting a good applicant
COST_MANUAL_REVIEW    = 150     # cost of routing an application to manual underwriter
COST_MANUAL_UNDERWRITER_PER_APP = 3500   # baseline fully-manual cost per application

# ── Confidence score weighting for final decision ────────────
DECISION_WEIGHTS = {
    "credit_risk":      0.55,
    "fraud_risk":       0.30,
    "document_quality": 0.15,
}

# ============================================================
# PATHS
# ============================================================

BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR              = "models"
CREDIT_MODEL_SUBDIR    = os.path.join(MODEL_DIR, "credit_risk")
FRAUD_MODEL_SUBDIR     = os.path.join(MODEL_DIR, "fraud")

DATA_DIR               = "data"
RAW_DATA_DIR           = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR     = os.path.join(DATA_DIR, "processed")
SYNTHETIC_STATEMENTS_DIR = os.path.join(DATA_DIR, "synthetic_statements")

LOGS_DIR               = "logs"
AUDIT_TRAIL_DIR        = os.path.join(LOGS_DIR, "audit_trail")
TESTS_DIR              = "tests"
SERVING_DIR            = "serving"
DOCS_DIR               = "docs"

for _d in (
    MODEL_DIR, CREDIT_MODEL_SUBDIR, FRAUD_MODEL_SUBDIR,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_STATEMENTS_DIR,
    LOGS_DIR, AUDIT_TRAIL_DIR,
):
    os.makedirs(_d, exist_ok=True)
