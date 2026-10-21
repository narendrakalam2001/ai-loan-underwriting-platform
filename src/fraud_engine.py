# ============================================================
# FRAUD ENGINE — Module 2 (Fraud Detection Engine)
# ============================================================
# Decision hierarchy (matches real payment-risk systems — rules
# are checked BEFORE the ML model):
#   1. Hard rule  — Amount > HIGH_AMOUNT_RULE       → BLOCK_RULE
#   2. ML model   — prob >= threshold               → BLOCK_MODEL
#   3. Borderline — prob >= threshold * ratio        → REVIEW
#   4. else                                          → APPROVE
# ============================================================

import pandas as pd
import logging

from src.config import FRAUD_BANDS, HIGH_AMOUNT_RULE, LOW_AMOUNT_RULE, REVIEW_PROB_RATIO

logger = logging.getLogger(__name__)


# ============================================================
# FRAUD BAND — probability → LOW / MEDIUM / HIGH
# ============================================================

def get_fraud_band(prob: float) -> str:
    for band, (low, high) in FRAUD_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# ROW-LEVEL DECISIONS (batch — used during training/evaluation)
# ============================================================

def fraud_engine(transaction_df: pd.DataFrame, probs, threshold: float) -> list:
    decisions = []

    for idx, (_, row) in enumerate(transaction_df.iterrows()):
        p = probs[idx]
        amount = row.get("Amount_original", row.get("Amount", 0))

        if amount > HIGH_AMOUNT_RULE:
            decisions.append("BLOCK_RULE")
        elif p >= threshold:
            decisions.append("BLOCK_MODEL")
        elif p >= threshold * REVIEW_PROB_RATIO:
            decisions.append("REVIEW")
        else:
            decisions.append("APPROVE")

    return decisions


# ============================================================
# SINGLE-TRANSACTION SCORING (used by API / Decision Engine)
# ============================================================

def score_transaction(row: dict, prob: float, threshold: float,
                       high_amount_rule: float = HIGH_AMOUNT_RULE,
                       low_amount_rule: float = LOW_AMOUNT_RULE) -> dict:
    """
    Returns a structured fraud verdict for one application/transaction.
    In the loan-underwriting context this runs on the applicant's
    declared financial profile + document-derived signals treated as
    a single "transaction" to be screened before credit scoring even
    runs — matching Module 2's spec: "Flag suspicious applications
    before credit check."

    `high_amount_rule` / `low_amount_rule` default to the card-transaction
    thresholds this model was trained on. When scoring a LOAN APPLICATION
    (₹ lakhs, not ₹ hundreds), pass config.LOAN_HIGH_AMOUNT_RULE /
    LOAN_LOW_AMOUNT_RULE instead — see services/prediction_service.py.
    """
    fraud_band = get_fraud_band(prob)
    amount = row.get("Amount_original", row.get("Amount", 0))

    rule_triggered = None
    if amount > high_amount_rule:
        verdict, rule_triggered = "BLOCK", "HIGH_AMOUNT_RULE"
    elif amount < low_amount_rule:
        verdict, rule_triggered = "REVIEW", "CARD_TESTING_PATTERN"
    elif prob >= threshold:
        verdict = "BLOCK"
    elif prob >= threshold * REVIEW_PROB_RATIO:
        verdict = "REVIEW"
    else:
        verdict = "APPROVE"

    return {
        "module":          "fraud_detection",
        "fraud_probability": round(float(prob), 4),
        "fraud_band":      fraud_band,
        "verdict":         verdict,
        "rule_triggered":  rule_triggered,
    }
