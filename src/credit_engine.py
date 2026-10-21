# ============================================================
# CREDIT ENGINE — Module 1 (Credit Risk Scoring Engine)
# ============================================================
# Fintech-grade 3-tier decision system:
#   APPROVE        → low predicted default risk, auto-approve
#   MANUAL_REVIEW  → borderline, routed to human underwriter
#   DECLINE        → high predicted default risk, auto-decline
#
# Decision hierarchy (hard business rules override the ML score,
# matching how real underwriting systems are built):
#   1. Hard rules   (debt ratio, severe delinquency history, income
#      floor, dependents count)
#   2. ML model probability vs the 3-tier thresholds in config.py
#
# Rules were redefined when the credit dataset was swapped from the
# Bank Personal Loan Modelling dataset (target = marketing-offer
# acceptance) to Kaggle's "Give Me Some Credit" (target =
# serious_dlqin2yrs, a genuine 90+ day delinquency label).
# ============================================================

import pandas as pd
import logging

from src.config import (
    RISK_BANDS, HIGH_DEBT_RATIO_RULE, SEVERE_DELINQUENCY_RULE,
    MIN_INCOME_RULE_MODEL_UNITS, HIGH_DEPENDENTS_RULE,
    CREDIT_APPROVE_MAX, CREDIT_MANUAL_REVIEW_MAX,
)

logger = logging.getLogger(__name__)


# ============================================================
# RISK BAND — probability → LOW / MEDIUM / HIGH
# ============================================================

def get_risk_band(prob: float) -> str:
    for band, (low, high) in RISK_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# ROW-LEVEL DECISIONS (batch — used during training/evaluation)
# ============================================================

def credit_engine(applicant_df: pd.DataFrame, probs, threshold: float) -> list:
    """
    Rule priority:
      1. debt_ratio > HIGH_DEBT_RATIO_RULE              → DECLINE_RULE
      2. times_90_days_late >= SEVERE_DELINQUENCY_RULE  → DECLINE_RULE
      3. monthly_income < MIN_INCOME_RULE_MODEL_UNITS   → REVIEW_INCOME
         (rows are in MODEL income units; the ₹5,000 floor is converted — see config.py)
      4. dependents >= HIGH_DEPENDENTS_RULE              → REVIEW_DEPENDENTS
      5. prob >= CREDIT_MANUAL_REVIEW_MAX                → DECLINE_MODEL
      6. prob >= CREDIT_APPROVE_MAX                      → REVIEW_MODEL
      7. else                                            → APPROVE
    """
    decisions = []

    for idx, (_, row) in enumerate(applicant_df.iterrows()):
        p = probs[idx]

        if "debt_ratio" in row and row["debt_ratio"] > HIGH_DEBT_RATIO_RULE:
            decisions.append("DECLINE_RULE")
            continue

        if "times_90_days_late" in row and row["times_90_days_late"] >= SEVERE_DELINQUENCY_RULE:
            decisions.append("DECLINE_RULE")
            continue

        if "monthly_income" in row and row["monthly_income"] < MIN_INCOME_RULE_MODEL_UNITS:
            decisions.append("REVIEW_INCOME")
            continue

        if "dependents" in row and row["dependents"] >= HIGH_DEPENDENTS_RULE:
            decisions.append("REVIEW_DEPENDENTS")
            continue

        if p >= CREDIT_MANUAL_REVIEW_MAX:
            decisions.append("DECLINE_MODEL")
        elif p >= CREDIT_APPROVE_MAX:
            decisions.append("REVIEW_MODEL")
        else:
            decisions.append("APPROVE")

    return decisions


# ============================================================
# SINGLE-APPLICANT SCORING (used by the API / Decision Engine)
# ============================================================

def score_applicant(row: dict, prob: float, threshold: float = None) -> dict:
    """
    Returns a structured credit-risk verdict for one applicant, collapsed
    to the platform-wide 3-tier vocabulary (APPROVE / MANUAL_REVIEW / DECLINE)
    so Module 4 (decision_engine.py) can combine it with fraud + document
    signals without needing to know this module's internal rule names.
    """
    risk_band = get_risk_band(prob)
    rule_triggered = None

    debt_ratio = row.get("debt_ratio", 0)
    severe_late = row.get("times_90_days_late", 0)

    if debt_ratio > HIGH_DEBT_RATIO_RULE:
        verdict, rule_triggered = "DECLINE", "HIGH_DEBT_RATIO"
    elif severe_late >= SEVERE_DELINQUENCY_RULE:
        verdict, rule_triggered = "DECLINE", "SEVERE_DELINQUENCY_HISTORY"
    elif row.get("monthly_income", float("inf")) < MIN_INCOME_RULE_MODEL_UNITS:
        verdict, rule_triggered = "MANUAL_REVIEW", "LOW_INCOME"
    elif row.get("dependents", 0) >= HIGH_DEPENDENTS_RULE:
        verdict, rule_triggered = "MANUAL_REVIEW", "MANY_DEPENDENTS"
    elif prob >= CREDIT_MANUAL_REVIEW_MAX:
        verdict = "DECLINE"
    elif prob >= CREDIT_APPROVE_MAX:
        verdict = "MANUAL_REVIEW"
    else:
        verdict = "APPROVE"

    return {
        "module":           "credit_risk",
        "default_probability": round(float(prob), 4),
        "risk_band":        risk_band,
        "verdict":          verdict,
        "rule_triggered":   rule_triggered,
    }
