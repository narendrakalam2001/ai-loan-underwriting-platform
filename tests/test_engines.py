# ============================================================
# PYTEST — credit_engine + fraud_engine
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.credit_engine import get_risk_band, score_applicant
from src.fraud_engine import get_fraud_band, score_transaction


class TestCreditEngine:

    def test_get_risk_band_low(self):
        assert get_risk_band(0.10) == "LOW"

    def test_get_risk_band_medium(self):
        assert get_risk_band(0.45) == "MEDIUM"

    def test_get_risk_band_high(self):
        assert get_risk_band(0.75) == "HIGH"

    def test_score_applicant_approve(self):
        row = {"monthly_income": 50000.0, "debt_ratio": 0.2, "times_90_days_late": 0, "dependents": 1}
        result = score_applicant(row, prob=0.05)
        assert result["verdict"] == "APPROVE"
        assert result["risk_band"] == "LOW"

    def test_score_applicant_decline_rule_on_debt_ratio(self):
        row = {"monthly_income": 20000.0, "debt_ratio": 1.5, "times_90_days_late": 0, "dependents": 1}
        result = score_applicant(row, prob=0.05)
        assert result["verdict"] == "DECLINE"
        assert result["rule_triggered"] == "HIGH_DEBT_RATIO"

    def test_score_applicant_decline_rule_on_severe_delinquency(self):
        row = {"monthly_income": 50000.0, "debt_ratio": 0.2, "times_90_days_late": 3, "dependents": 1}
        result = score_applicant(row, prob=0.05)
        assert result["verdict"] == "DECLINE"
        assert result["rule_triggered"] == "SEVERE_DELINQUENCY_HISTORY"

    def test_score_applicant_review_on_low_income(self):
        row = {"monthly_income": 900.0, "debt_ratio": 0.2, "times_90_days_late": 0, "dependents": 1}   # model units (< Rs.5,000 / 5)
        result = score_applicant(row, prob=0.05)
        assert result["verdict"] == "MANUAL_REVIEW"
        assert result["rule_triggered"] == "LOW_INCOME"

    def test_score_applicant_review_on_many_dependents(self):
        row = {"monthly_income": 50000.0, "debt_ratio": 0.2, "times_90_days_late": 0, "dependents": 7}
        result = score_applicant(row, prob=0.05)
        assert result["verdict"] == "MANUAL_REVIEW"
        assert result["rule_triggered"] == "MANY_DEPENDENTS"

    def test_score_applicant_review_via_model(self):
        row = {"monthly_income": 50000.0, "debt_ratio": 0.3, "times_90_days_late": 0, "dependents": 1}
        result = score_applicant(row, prob=0.45)
        assert result["verdict"] == "MANUAL_REVIEW"
        assert result["rule_triggered"] is None

    def test_score_applicant_decline_via_model(self):
        row = {"monthly_income": 50000.0, "debt_ratio": 0.3, "times_90_days_late": 0, "dependents": 1}
        result = score_applicant(row, prob=0.9)
        assert result["verdict"] == "DECLINE"
        assert result["rule_triggered"] is None

    def test_output_keys_complete(self):
        row = {"monthly_income": 50000.0, "debt_ratio": 0.3, "times_90_days_late": 0, "dependents": 1}
        result = score_applicant(row, prob=0.2)
        for key in ("module", "default_probability", "risk_band", "verdict", "rule_triggered"):
            assert key in result


class TestFraudEngine:

    def test_get_fraud_band_low(self):
        assert get_fraud_band(0.10) == "LOW"

    def test_get_fraud_band_high(self):
        assert get_fraud_band(0.75) == "HIGH"

    def test_high_amount_hard_rule_blocks(self):
        row = {"Amount_original": 10000}
        result = score_transaction(row, prob=0.01, threshold=0.5)
        assert result["verdict"] == "BLOCK"
        assert result["rule_triggered"] == "HIGH_AMOUNT_RULE"

    def test_low_amount_flags_review(self):
        row = {"Amount_original": 2}
        result = score_transaction(row, prob=0.01, threshold=0.5)
        assert result["verdict"] == "REVIEW"
        assert result["rule_triggered"] == "CARD_TESTING_PATTERN"

    def test_high_prob_blocks_via_model(self):
        row = {"Amount_original": 100}
        result = score_transaction(row, prob=0.9, threshold=0.5)
        assert result["verdict"] == "BLOCK"
        assert result["rule_triggered"] is None

    def test_low_prob_approves(self):
        row = {"Amount_original": 100}
        result = score_transaction(row, prob=0.05, threshold=0.5)
        assert result["verdict"] == "APPROVE"

    def test_custom_thresholds_override_defaults(self):
        """Loan-context thresholds (much larger) must not trigger on a normal loan amount."""
        row = {"Amount_original": 500_000}
        result = score_transaction(row, prob=0.05, threshold=0.5,
                                    high_amount_rule=5_000_000, low_amount_rule=5_000)
        assert result["rule_triggered"] is None
        assert result["verdict"] == "APPROVE"

    def test_custom_thresholds_still_catch_extreme_amount(self):
        row = {"Amount_original": 50_000_000}
        result = score_transaction(row, prob=0.05, threshold=0.5,
                                    high_amount_rule=5_000_000, low_amount_rule=5_000)
        assert result["verdict"] == "BLOCK"
        assert result["rule_triggered"] == "HIGH_AMOUNT_RULE"
