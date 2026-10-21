# ============================================================
# DECISION ENGINE — Module 4 (Unified Decision + Explainability)
# ============================================================
# Combines:
#   Module 1 — credit_engine.score_applicant()   output
#   Module 2 — fraud_engine.score_transaction()  output
#   Module 3 — document_parser.parse_bank_statement() output
#
# Into ONE final underwriting decision with:
#   - final verdict        (APPROVE / MANUAL_REVIEW / DECLINE)
#   - confidence score     (weighted per config.DECISION_WEIGHTS)
#   - plain-English explanation (SHAP-aware when available)
#   - improvement suggestions (when DECLINE / MANUAL_REVIEW)
#   - full audit-trail JSON (for regulatory compliance)
#
# Decision hierarchy (fraud is checked FIRST — matches Module 2's
# spec: "Flag suspicious applications before credit check"):
#   1. Fraud BLOCK           → final DECLINE (fraud always wins)
#   2. Fraud REVIEW          → final verdict can't be better than MANUAL_REVIEW
#   3. Document extraction has warnings / low confidence → floor at MANUAL_REVIEW
#   4. Credit verdict        → otherwise drives the outcome
# ============================================================

import os
import re
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from src.config import DECISION_WEIGHTS, AUDIT_TRAIL_DIR

logger = logging.getLogger(__name__)

VERDICT_RANK = {"APPROVE": 0, "MANUAL_REVIEW": 1, "DECLINE": 2}


def _worse(a: str, b: str) -> str:
    """Returns whichever of two verdicts is more restrictive."""
    return a if VERDICT_RANK[a] >= VERDICT_RANK[b] else b


def _document_quality_score(doc_result: Optional[Dict]) -> float:
    """
    Maps document_parser output to a 0-1 confidence score used in the
    weighted final confidence. No document / many warnings → low score.
    """
    if doc_result is None:
        return 0.3   # applicant hasn't uploaded a statement yet

    warnings = doc_result.get("warnings", [])
    stability_map = {"Good": 1.0, "Moderate": 0.6, "Poor": 0.25}

    income_score = stability_map.get(doc_result.get("income_stability", "Poor"), 0.25)
    balance_score = stability_map.get(doc_result.get("account_stability", "Poor"), 0.25)
    warning_penalty = min(0.5, 0.15 * len(warnings))

    score = (income_score + balance_score) / 2 - warning_penalty
    return round(max(0.0, min(1.0, score)), 3)


def _build_explanation(
    final_verdict: str,
    credit_result: Dict,
    fraud_result: Dict,
    doc_result: Optional[Dict],
    shap_pairs: Optional[list] = None,
) -> str:
    """
    Builds the plain-English explanation shown on Screen 3 of the User
    Portal, e.g.:
      "Loan declined because: high existing obligations detected in bank
       statement + credit score below threshold + unusual transaction
       pattern flagged"
    """
    reasons = []

    if fraud_result["verdict"] in ("BLOCK", "REVIEW"):
        if fraud_result.get("rule_triggered"):
            reasons.append(f"fraud check flagged {fraud_result['rule_triggered'].replace('_', ' ').lower()}")
        else:
            reasons.append("unusual transaction pattern flagged by the fraud model")

    if credit_result.get("rule_triggered"):
        reasons.append(credit_result["rule_triggered"].replace("_", " ").lower())
    elif credit_result["verdict"] in ("DECLINE", "MANUAL_REVIEW"):
        reasons.append(f"credit risk score in the {credit_result['risk_band'].lower()} band")

    if doc_result is not None:
        if doc_result.get("income_stability") == "Poor":
            reasons.append("unstable income detected in bank statement")
        if doc_result.get("account_stability") == "Poor":
            reasons.append("volatile account balance trend")
        if doc_result.get("existing_monthly_emi", 0) > 0 and credit_result["verdict"] != "APPROVE":
            reasons.append(f"existing EMI obligations of ₹{doc_result['existing_monthly_emi']:,.0f}/month detected in bank statement")

    if shap_pairs:
        top_feature = shap_pairs[0]["feature"].replace("_", " ")
        reasons.append(f"'{top_feature}' was the strongest driver of this decision")

    if final_verdict == "APPROVE":
        if not reasons:
            reasons = ["strong income stability", "low existing obligations", "clean fraud check"]
        prefix = "Approved because: "
    elif final_verdict == "MANUAL_REVIEW":
        if not reasons:
            reasons = ["borderline risk profile requiring analyst judgment"]
        prefix = "Routed to manual review because: "
    else:
        if not reasons:
            reasons = ["overall risk profile exceeded the platform's automated approval threshold"]
        prefix = "Declined because: "

    return prefix + " + ".join(reasons)


def _improvement_suggestions(credit_result: Dict, doc_result: Optional[Dict]) -> list:
    suggestions = []

    rule = credit_result.get("rule_triggered")
    if rule == "HIGH_DEBT_RATIO":
        suggestions.append("Reduce existing debt so monthly obligations stay well below your income")
    if rule == "SEVERE_DELINQUENCY_HISTORY":
        suggestions.append("Clear any overdue payments and keep every payment on time for 12-24 months before reapplying")
    if rule == "LOW_INCOME":
        suggestions.append("Apply with a co-applicant or after an income increase")
    if rule == "MANY_DEPENDENTS":
        suggestions.append("Apply with a co-applicant so household income is shared across fewer dependents")
    if doc_result and doc_result.get("existing_monthly_emi", 0) > 0:
        suggestions.append("Reduce existing EMIs before reapplying")
    if doc_result and doc_result.get("income_stability") in ("Moderate", "Poor"):
        suggestions.append("Maintain 6 months of stable salary credits before reapplying")
    if credit_result["risk_band"] in ("MEDIUM", "HIGH"):
        suggestions.append("Wait 6 months to improve credit history, then reapply")

    if not suggestions:
        suggestions.append("Provide additional supporting documents if you believe this decision is incorrect")

    return suggestions


# Only these four fields are ever stored from the applicant's request. They are exactly
# what the decision letter (offer + EMI page) and the LLM explainer need. Income, bureau
# numbers and everything else stay out of the stored summary on purpose.
APPLICATION_SUMMARY_FIELDS = ("full_name", "loan_amount_requested", "tenure_months", "loan_purpose")


def _application_summary(application: Optional[Dict]) -> Dict:
    if not application:
        return {}
    return {k: application[k] for k in APPLICATION_SUMMARY_FIELDS if application.get(k) is not None}


def make_final_decision(
    applicant_id:  Optional[str],
    credit_result: Dict,     # from credit_engine.score_applicant()
    fraud_result:  Dict,     # from fraud_engine.score_transaction()
    doc_result:    Optional[Dict] = None,   # from document_parser.parse_bank_statement().to_dict()
    shap_pairs:    Optional[list] = None,   # from evaluation.explain_single_prediction()
    application:   Optional[Dict] = None,   # minimal request summary (see _application_summary)
) -> Dict:
    """
    Module 4 — the single entry point the FastAPI /underwrite endpoint
    calls once Modules 1-3 have produced their individual outputs.
    """
    application_id = applicant_id or f"APP{uuid.uuid4().hex[:10].upper()}"

    # ── 1. Fraud gate — always checked first ────────────────────
    if fraud_result["verdict"] == "BLOCK":
        final_verdict = "DECLINE"
    elif fraud_result["verdict"] == "REVIEW":
        final_verdict = _worse("MANUAL_REVIEW", credit_result["verdict"])
    else:
        final_verdict = credit_result["verdict"]

    # ── 2. Document-confidence floor ────────────────────────────
    doc_quality = _document_quality_score(doc_result)
    if doc_result is not None and len(doc_result.get("warnings", [])) >= 2:
        final_verdict = _worse(final_verdict, "MANUAL_REVIEW")

    # ── 3. Weighted confidence score ─────────────────────────────
    credit_confidence = 1.0 - credit_result["default_probability"]
    fraud_confidence = 1.0 - fraud_result["fraud_probability"]

    confidence = (
        DECISION_WEIGHTS["credit_risk"] * credit_confidence
        + DECISION_WEIGHTS["fraud_risk"] * fraud_confidence
        + DECISION_WEIGHTS["document_quality"] * doc_quality
    )
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    explanation = _build_explanation(final_verdict, credit_result, fraud_result, doc_result, shap_pairs)
    suggestions = _improvement_suggestions(credit_result, doc_result) if final_verdict != "APPROVE" else []

    result = {
        "application_id":  application_id,
        "final_verdict":   final_verdict,
        "confidence_score": confidence,
        "explanation":     explanation,
        "improvement_suggestions": suggestions,
        "module_outputs": {
            "credit_risk":      credit_result,
            "fraud_detection":  fraud_result,
            "document_parser":  doc_result,
        },
        "shap_top_drivers": shap_pairs or [],
        "decided_at":       datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    summary = _application_summary(application)
    if summary:
        result["application"] = summary

    _write_audit_trail(result)
    logger.info("Application %s → %s (confidence=%.4f)", application_id, final_verdict, confidence)

    return result


def _write_audit_trail(result: Dict):
    """Every decision is persisted as its own JSON file — required for
    regulatory compliance (RBI audit trail requirement referenced in Module 5A)."""
    os.makedirs(AUDIT_TRAIL_DIR, exist_ok=True)
    path = os.path.join(AUDIT_TRAIL_DIR, f"{result['application_id']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Audit trail saved → %s", path)


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def load_audit_trail(application_id: str) -> Optional[Dict]:
    # application_id comes from outside (API path param, chatbot message), so
    # it must never be allowed to contain path separators or ".." — on
    # Windows a backslash would otherwise let "..\\..\\secret" escape the
    # audit-trail folder. Real IDs are "APP" + 10 hex chars; the test suite
    # uses short alphanumeric/underscore IDs, all of which pass this check.
    if not isinstance(application_id, str) or not _SAFE_ID.match(application_id):
        return None
    path = os.path.join(AUDIT_TRAIL_DIR, f"{application_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
