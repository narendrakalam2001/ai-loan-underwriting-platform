# ============================================================
# PYTEST — decision_engine (Module 4, unified decision)
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest

from src.decision_engine import make_final_decision, load_audit_trail, _worse


@pytest.fixture(autouse=True)
def isolated_audit_trail(tmp_path, monkeypatch):
    """
    Every test writes its audit-trail JSONs into a throw-away temp directory
    instead of the project's real logs/audit_trail/ folder, so running pytest
    can never delete real application decisions made through the API/portal.
    """
    import src.decision_engine as decision_engine
    monkeypatch.setattr(decision_engine, "AUDIT_TRAIL_DIR", str(tmp_path))
    yield


def _credit(verdict="APPROVE", prob=0.05, band="LOW", rule=None):
    return {"module": "credit_risk", "default_probability": prob, "risk_band": band,
            "verdict": verdict, "rule_triggered": rule}


def _fraud(verdict="APPROVE", prob=0.02, band="LOW", rule=None):
    return {"module": "fraud_detection", "fraud_probability": prob, "fraud_band": band,
            "verdict": verdict, "rule_triggered": rule}


class TestWorse:

    def test_decline_beats_everything(self):
        assert _worse("APPROVE", "DECLINE") == "DECLINE"
        assert _worse("DECLINE", "MANUAL_REVIEW") == "DECLINE"

    def test_manual_review_beats_approve(self):
        assert _worse("APPROVE", "MANUAL_REVIEW") == "MANUAL_REVIEW"

    def test_approve_is_weakest(self):
        assert _worse("APPROVE", "APPROVE") == "APPROVE"


class TestMakeFinalDecision:

    def test_clean_profile_approves(self):
        result = make_final_decision("TEST001", _credit("APPROVE"), _fraud("APPROVE"))
        assert result["final_verdict"] == "APPROVE"
        assert "Approved" in result["explanation"]

    def test_fraud_block_forces_decline_even_if_credit_approves(self):
        result = make_final_decision("TEST002", _credit("APPROVE"), _fraud("BLOCK", rule="HIGH_AMOUNT_RULE"))
        assert result["final_verdict"] == "DECLINE"

    def test_fraud_review_floors_at_manual_review(self):
        result = make_final_decision("TEST003", _credit("APPROVE"), _fraud("REVIEW"))
        assert result["final_verdict"] == "MANUAL_REVIEW"

    def test_credit_decline_propagates(self):
        result = make_final_decision("TEST004", _credit("DECLINE", prob=0.9, band="HIGH"), _fraud("APPROVE"))
        assert result["final_verdict"] == "DECLINE"

    def test_document_warnings_floor_at_manual_review(self):
        doc = {"income_stability": "Poor", "account_stability": "Poor",
               "warnings": ["w1", "w2"], "existing_monthly_emi": 0}
        result = make_final_decision("TEST005", _credit("APPROVE"), _fraud("APPROVE"), doc_result=doc)
        assert result["final_verdict"] == "MANUAL_REVIEW"

    def test_confidence_score_in_valid_range(self):
        result = make_final_decision("TEST006", _credit("APPROVE"), _fraud("APPROVE"))
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_decline_has_improvement_suggestions(self):
        result = make_final_decision("TEST007", _credit("DECLINE", prob=0.9, band="HIGH"), _fraud("APPROVE"))
        assert len(result["improvement_suggestions"]) > 0

    def test_approve_has_no_improvement_suggestions(self):
        result = make_final_decision("TEST008", _credit("APPROVE"), _fraud("APPROVE"))
        assert result["improvement_suggestions"] == []

    def test_audit_trail_is_written_and_loadable(self):
        result = make_final_decision("TEST009", _credit("APPROVE"), _fraud("APPROVE"))
        loaded = load_audit_trail("TEST009")
        assert loaded is not None
        assert loaded["application_id"] == "TEST009"

    def test_load_audit_trail_returns_none_for_missing_id(self):
        assert load_audit_trail("DOES_NOT_EXIST_123") is None

    def test_application_id_auto_generated_when_none_given(self):
        result = make_final_decision(None, _credit("APPROVE"), _fraud("APPROVE"))
        assert result["application_id"].startswith("APP")
