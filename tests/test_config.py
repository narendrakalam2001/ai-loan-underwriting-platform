# ============================================================
# PYTEST — config.py sanity checks
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    RISK_BANDS, FRAUD_BANDS, PSI_MODERATE, PSI_HIGH,
    MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP,
    MAX_INCOME_RATIO_RULE, MIN_INCOME_RULE, FAMILY_SIZE_RULE,
    HIGH_DEBT_RATIO_RULE, SEVERE_DELINQUENCY_RULE, HIGH_DEPENDENTS_RULE,
    HIGH_AMOUNT_RULE, LOW_AMOUNT_RULE, LOAN_HIGH_AMOUNT_RULE, LOAN_LOW_AMOUNT_RULE,
    CREDIT_APPROVE_MAX, CREDIT_MANUAL_REVIEW_MAX,
    DECISION_WEIGHTS,
)


class TestConfigThresholds:

    def test_psi_thresholds_ordered(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_reasonable(self):
        assert 0.0 < MIN_F1_IMPROVEMENT < 0.1
        assert 0.5 < MIN_ROCAUC_THRESHOLD < 1.0
        assert 0.0 < MAX_GENERALIZATION_GAP < 1.0

    def test_risk_band_boundaries_valid(self):
        for band, (low, high) in RISK_BANDS.items():
            assert 0.0 <= low <= 1.0
            assert 0.0 <= high <= 1.1

    def test_fraud_band_boundaries_valid(self):
        for band, (low, high) in FRAUD_BANDS.items():
            assert 0.0 <= low <= 1.0
            assert 0.0 <= high <= 1.1

    def test_business_rule_thresholds_positive(self):
        assert MAX_INCOME_RATIO_RULE > 0
        assert MIN_INCOME_RULE > 0
        assert FAMILY_SIZE_RULE > 0

    def test_gmsc_credit_rule_thresholds_positive(self):
        """Rules used by credit_engine.py after the dataset swap to Give Me Some Credit."""
        assert HIGH_DEBT_RATIO_RULE > 0
        assert SEVERE_DELINQUENCY_RULE > 0
        assert HIGH_DEPENDENTS_RULE > 0

    def test_credit_tier_thresholds_ordered(self):
        assert CREDIT_APPROVE_MAX < CREDIT_MANUAL_REVIEW_MAX

    def test_loan_amount_rules_much_larger_than_card_rules(self):
        """LOAN_* thresholds must be scaled for ₹ loan amounts, not card transactions."""
        assert LOAN_HIGH_AMOUNT_RULE > HIGH_AMOUNT_RULE * 100
        assert LOAN_LOW_AMOUNT_RULE > LOW_AMOUNT_RULE

    def test_decision_weights_sum_to_one(self):
        assert abs(sum(DECISION_WEIGHTS.values()) - 1.0) < 1e-6
