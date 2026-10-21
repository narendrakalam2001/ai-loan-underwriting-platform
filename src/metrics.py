# ============================================================
# METRICS — AI Loan Underwriting Platform
# ============================================================
# Shared metric primitives used by both Module 1 (Credit Risk)
# and Module 2 (Fraud Detection) evaluation pipelines.
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import COST_WRONG_APPROVAL, COST_WRONG_REJECTION, COST_MANUAL_REVIEW

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
    y_true:           np.ndarray,
    y_prob:           np.ndarray,
    target_recall:    float = None,
    target_precision: float = None,
) -> float:
    """Default: maximize F1. Optional: constrain on recall or precision first."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    if target_recall is not None:
        idxs = np.where(recall >= target_recall)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(precision[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    if target_precision is not None:
        idxs = np.where(precision >= target_precision)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(recall[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    logger.info("Best threshold: %.4f | precision=%.4f recall=%.4f f1=%.4f",
                best_thr, precision[best_idx], recall[best_idx], f1_scores[best_idx])

    return float(best_thr)


# ============================================================
# PSI — Population Stability Index (edge-based, correct)
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.10   → no significant shift (stable)
    PSI 0.10-0.20 → moderate shift (monitor closely)
    PSI > 0.20   → major shift (retrain recommended)

    Correct approach: compute quantile bin EDGES from `expected` only,
    then bin BOTH distributions using those SAME edges. Ranking both
    distributions independently is a common bug — it makes both uniform
    and PSI always ~0 regardless of actual shift.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.percentile(expected, quantiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)
        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# KS STATISTIC
# ============================================================

def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ============================================================
# RECALL @ K / LIFT @ K
# ============================================================

def recall_at_k(y_true, y_prob, k: float = 0.05) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    top_n = int(len(df) * k)
    return float(df.iloc[:top_n]["y"].sum() / (df["y"].sum() + 1e-9))


def lift_at_k(y_true, y_prob, k: float = 0.05) -> float:
    base = np.mean(y_true)
    return float(recall_at_k(y_true, y_prob, k) / (base + 1e-9))


# ============================================================
# COST-SENSITIVE EVALUATION — CREDIT RISK
# ============================================================

def cost_sensitive_evaluation_credit(
    X_test,
    y_true,
    y_pred,
    loan_amount_col: str = "monthly_income",
    loss_given_default: float = 0.6,
    review_cost_per_case: float = 50.0,
) -> dict:
    """
    Expected Credit Loss style: ECL = PD x LGD x EAD.
      - False negative (missed bad loan) → real ₹ loss
      - False positive (good loan flagged) → analyst review cost
    """
    df = X_test.copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred

    fn_mask     = (df["y_true"] == 1) & (df["y_pred"] == 0)
    fn_exposure = df.loc[fn_mask, loan_amount_col].sum() if loan_amount_col in df.columns else fn_mask.sum()
    fn_loss     = float(fn_exposure * loss_given_default)

    fp_mask = (df["y_true"] == 0) & (df["y_pred"] == 1)
    fp_cost = float(fp_mask.sum() * review_cost_per_case)

    total_loss = fn_loss + fp_cost

    result = {
        "false_negative_count": int(fn_mask.sum()),
        "false_positive_count": int(fp_mask.sum()),
        "estimated_fn_loss":    round(fn_loss, 2),
        "review_cost":          round(fp_cost, 2),
        "total_estimated_loss": round(total_loss, 2),
    }
    logger.info("Credit cost eval | FN=%d FP=%d FN_loss=%.2f review_cost=%.2f total=%.2f",
                *result.values())
    return result


# ============================================================
# COST-SENSITIVE EVALUATION — FRAUD
# ============================================================

def cost_sensitive_evaluation_fraud(
    X_test,
    y_true,
    y_pred,
    amount_col: str = "Amount_original",
    review_cost_per_case: float = 15.0,
) -> dict:
    """
    Fraud cost model:
      - False negative (missed fraud) → full transaction amount lost
      - False positive (genuine txn blocked/reviewed) → fixed review/friction cost
    """
    df = X_test.copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred

    fn_mask     = (df["y_true"] == 1) & (df["y_pred"] == 0)
    fn_exposure = df.loc[fn_mask, amount_col].sum() if amount_col in df.columns else fn_mask.sum()
    fn_loss     = float(fn_exposure)

    fp_mask = (df["y_true"] == 0) & (df["y_pred"] == 1)
    fp_cost = float(fp_mask.sum() * review_cost_per_case)

    total_loss = fn_loss + fp_cost

    result = {
        "false_negative_count": int(fn_mask.sum()),
        "false_positive_count": int(fp_mask.sum()),
        "estimated_fn_loss":    round(fn_loss, 2),
        "review_cost":          round(fp_cost, 2),
        "total_estimated_loss": round(total_loss, 2),
    }
    logger.info("Fraud cost eval | FN=%d FP=%d FN_loss=%.2f review_cost=%.2f total=%.2f",
                *result.values())
    return result


# ============================================================
# UNIFIED DECISION-ENGINE COST MODEL (Module 4)
# ============================================================

def underwriting_cost_summary(n_approve: int, n_review: int, n_decline: int,
                                bad_rate_in_approved: float = 0.0) -> dict:
    """
    Business-impact rollup used in the README / dashboard KPIs:
    compares the platform's automated cost to a fully-manual underwriting
    process (every application reviewed by a human).
    """
    n_total = n_approve + n_review + n_decline
    automated_cost = (
        n_review * COST_MANUAL_REVIEW
        + n_approve * bad_rate_in_approved * COST_WRONG_APPROVAL
    )
    manual_cost = n_total * 3500  # baseline fully-manual underwriter cost per application
    savings = manual_cost - automated_cost

    return {
        "n_total":            n_total,
        "n_approve":          n_approve,
        "n_review":           n_review,
        "n_decline":          n_decline,
        "automated_cost_inr": round(automated_cost, 2),
        "manual_cost_inr":    round(manual_cost, 2),
        "estimated_savings_inr": round(savings, 2),
        "savings_pct":        round(100 * savings / manual_cost, 2) if manual_cost else 0.0,
    }


# ============================================================
# DRIFT REPORT — feature mean shift (quick alternative to PSI)
# ============================================================

def simple_drift_report(X_ref: "pd.DataFrame", X_new: "pd.DataFrame", top_n: int = 10):
    diffs = (X_ref.mean() - X_new.mean()).abs()
    rel = (diffs / (X_ref.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)
