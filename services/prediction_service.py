# ============================================================
# PREDICTION SERVICE — AI Loan Underwriting Platform
# ============================================================
# Bridges raw API/portal input → engineered features → model
# probability → engine-level verdict, for BOTH Module 1 (credit)
# and Module 2 (fraud). serving/api.py calls these functions;
# they contain no FastAPI-specific code so they're also unit-
# testable and reusable from frontend/app.py directly if needed.
# ============================================================

import random
import logging
import numpy as np
import pandas as pd

from src.data_loader import add_credit_engineered_features, add_fraud_engineered_features
from src.credit_engine import score_applicant as credit_score_applicant
from src.fraud_engine import score_transaction as fraud_score_transaction

from src.config import INCOME_INR_PER_MODEL_UNIT

logger = logging.getLogger(__name__)


# ============================================================
# MODULE 1 — CREDIT RISK
# ============================================================

def prepare_credit_features(input_data: dict) -> pd.DataFrame:
    """
    Raw applicant dict → engineered feature DataFrame (mirrors training).

    The API/portal collect applicant-facing fields (e.g. "existing EMIs in
    ₹/month", "credit utilization %") rather than the raw Kaggle "Give Me
    Some Credit" column names, since no applicant self-reports a
    `RevolvingUtilizationOfUnsecuredLines` value. This function derives the
    technical GMSC-schema columns the trained model actually expects:

        debt_ratio             = existing_emis / monthly_income   (from the real ₹ values)
        monthly_income         = ₹ income / config.INCOME_INR_PER_MODEL_UNIT  (model trained in GMSC units)
        revolving_utilization  = credit_utilization_pct / 100

    If the input already contains the technical column names directly
    (e.g. when called from training/evaluation code with a real dataset
    row), those values are used as-is rather than being overwritten.
    """
    data = dict(input_data)

    monthly_income = float(data.get("monthly_income", 0) or 0)

    if "debt_ratio" not in data:
        existing_emis = float(data.get("existing_emis", 0) or 0)
        data["debt_ratio"] = existing_emis / monthly_income if monthly_income > 0 else 0.0

    # ₹ → model units. debt_ratio above is already computed from the REAL ₹ figures (a ratio is
    # unit-free), so scaling the income afterwards cannot distort it.
    if "monthly_income" in data:
        data["monthly_income"] = monthly_income / INCOME_INR_PER_MODEL_UNIT

    if "revolving_utilization" not in data:
        util_pct = float(data.get("credit_utilization_pct", 0) or 0)
        data["revolving_utilization"] = util_pct / 100.0

    # Model was trained on features that include validate_credit_data()'s
    # missingness flags (monthly_income_was_missing / dependents_was_missing / late_count_was_sentinel).
    # A live API call always supplies a real value for both, so both flags
    # are always 0 here — but the columns must still exist or the trained
    # ColumnTransformer will error on a missing expected feature.
    data.setdefault("monthly_income_was_missing", 0)
    data.setdefault("dependents_was_missing", 0)
    data.setdefault("late_count_was_sentinel", 0)   # 96/98 placeholder codes never occur in live input

    df = pd.DataFrame([data])
    df = add_credit_engineered_features(df)
    return df


def predict_credit(model, input_data: dict, threshold: float) -> dict:
    df = prepare_credit_features(input_data)

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        logger.error("Credit predict_proba failed: %s", e)
        prob = 0.5

    row = df.iloc[0].to_dict()
    result = credit_score_applicant(row, prob, threshold)

    logger.info("Credit prediction | prob=%.4f band=%s verdict=%s",
                result["default_probability"], result["risk_band"], result["verdict"])
    return result


# ============================================================
# MODULE 2 — FRAUD DETECTION
# ============================================================
#
# IMPORTANT / KNOWN LIMITATION (documented, not hidden):
# The fraud model is trained on the classic anonymized card-
# transaction dataset (V1-V28 PCA components + Time + Amount).
# A loan *application* has no natural equivalent of those PCA
# components. Exactly like the reference Fraud Detection project's
# own API (`serving/fraud_api.py`, which fills V1-V28 with
# `random.uniform(-5, 5)`), this service fills the missing
# components with random draws so the pretrained model can be
# called at all. This means the fraud score in this platform is a
# STRUCTURAL DEMONSTRATION of "run a fraud model before the credit
# check", not a real fraud signal for loan applications — a
# production system would instead train Module 2 on genuine
# application-level fraud signals (device fingerprint, velocity of
# applications, IP/geo mismatch, document forgery signals, etc.).
# The HIGH_AMOUNT_RULE hard rule (checked before the ML score) is
# real and meaningful: an applicant requesting an implausibly large
# loan amount is a genuine, model-independent red flag.
# ============================================================

def prepare_fraud_features(input_data: dict, feature_order: list = None) -> pd.DataFrame:
    df = pd.DataFrame([input_data])

    # Seed deterministically from the applicant's own data so identical
    # requests always get identical (reproducible) fraud scores, rather
    # than a different random draw on every call — important for the
    # /application/{id} audit trail to be explainable on lookup.
    seed_material = "|".join(f"{k}={v}" for k, v in sorted(input_data.items()))
    rng = random.Random(seed_material)

    for i in range(1, 29):
        col = f"V{i}"
        if col not in df.columns:
            df[col] = rng.uniform(-5, 5)

    if "Time" not in df.columns:
        df["Time"] = 0.0
    if "Amount" not in df.columns:
        df["Amount"] = input_data.get("loan_amount_requested", 0.0)

    df = add_fraud_engineered_features(df)

    if feature_order:
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0.0

    return df


def predict_fraud(model, input_data: dict, threshold: float, loan_context: bool = True) -> dict:
    """
    `loan_context=True` (the default, used by serving/api.py) applies the
    LOAN_HIGH_AMOUNT_RULE / LOAN_LOW_AMOUNT_RULE thresholds from config
    (₹ lakhs) instead of the card-transaction thresholds the model was
    trained on (₹ hundreds) — see the module docstring above.
    """
    df = prepare_fraud_features(input_data)

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        logger.error("Fraud predict_proba failed: %s", e)
        prob = 0.0

    row = df.iloc[0].to_dict()

    if loan_context:
        from src.config import LOAN_HIGH_AMOUNT_RULE, LOAN_LOW_AMOUNT_RULE
        result = fraud_score_transaction(row, prob, threshold,
                                          high_amount_rule=LOAN_HIGH_AMOUNT_RULE,
                                          low_amount_rule=LOAN_LOW_AMOUNT_RULE)
    else:
        result = fraud_score_transaction(row, prob, threshold)

    logger.info("Fraud prediction | prob=%.4f band=%s verdict=%s",
                result["fraud_probability"], result["fraud_band"], result["verdict"])
    return result
