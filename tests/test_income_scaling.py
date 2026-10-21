# ============================================================
# PYTEST — ₹ → model-unit income scaling (config.INCOME_INR_PER_MODEL_UNIT)
# The model is trained on dollar-scale GMSC income; live input is in ₹.
# ============================================================

import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest

from src import config
from src.credit_engine import score_applicant, credit_engine
from src.chatbot import eligibility_checker
import services.prediction_service as ps
import pandas as pd

BASE = {"age": 35, "existing_emis": 0.0, "credit_utilization_pct": 20.0, "dependents": 1, "open_credit_lines": 3,
        "real_estate_loans": 0, "times_30_59_days_late": 0, "times_60_89_days_late": 0, "times_90_days_late": 0}


def _features(income, emis=0.0):
    return ps.prepare_credit_features(dict(BASE, monthly_income=income, existing_emis=emis))


def test_default_factor_and_derived_floor():
    assert config.INCOME_INR_PER_MODEL_UNIT == 5.0
    assert config.MIN_INCOME_RULE_MODEL_UNITS == pytest.approx(config.MIN_INCOME_RULE / 5.0)


def test_rupee_income_is_converted_to_model_units():
    f = _features(50000.0)
    assert f.loc[0, "monthly_income"] == pytest.approx(10000.0)
    assert f.loc[0, "income_per_dependent"] == pytest.approx(5000.0)          # derived feature uses scaled income


def test_debt_ratio_is_computed_from_real_rupees_and_is_unit_free():
    assert _features(50000.0, emis=5000.0).loc[0, "debt_ratio"] == pytest.approx(0.1)
    assert _features(50000.0, emis=60000.0).loc[0, "debt_ratio"] == pytest.approx(1.2)


def test_factor_can_be_changed_at_runtime(monkeypatch):
    monkeypatch.setattr(ps, "INCOME_INR_PER_MODEL_UNIT", 20.0)
    assert _features(50000.0).loc[0, "monthly_income"] == pytest.approx(2500.0)


@pytest.mark.parametrize("income,expect_review", [(3000, True), (4999, True), (5000, False), (7000, False)])
def test_low_income_rule_still_means_rs_5000_at_serving(income, expect_review):
    row = _features(float(income)).iloc[0].to_dict()
    rule = score_applicant(row, prob=0.05)["rule_triggered"]
    assert (rule == "LOW_INCOME") is expect_review


@pytest.mark.parametrize("income", [3000, 4999, 5000, 7000, 50000])
def test_engine_and_chatbot_eligibility_agree_on_the_income_floor(income):
    engine_review = score_applicant(_features(float(income)).iloc[0].to_dict(), prob=0.05)["rule_triggered"] == "LOW_INCOME"
    chatbot_review = eligibility_checker(float(income), 0)["indicative_outcome"] == "LIKELY_MANUAL_REVIEW"
    assert engine_review == chatbot_review


def test_batch_evaluation_uses_model_unit_floor():
    df = pd.DataFrame({"monthly_income": [900.0, 1000.0, 5400.0], "debt_ratio": [0.2] * 3,
                       "times_90_days_late": [0] * 3, "dependents": [1] * 3})
    assert credit_engine(df, [0.05] * 3, 0.5) == ["REVIEW_INCOME", "APPROVE", "APPROVE"]


def test_env_override_and_validation():
    code = "from src import config; print(config.INCOME_INR_PER_MODEL_UNIT, config.MIN_INCOME_RULE_MODEL_UNITS)"
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                         env={**os.environ, "INCOME_INR_PER_MODEL_UNIT": "20"})
    assert out.stdout.split() == ["20.0", "250.0"]
    bad = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                         env={**os.environ, "INCOME_INR_PER_MODEL_UNIT": "0"})
    assert bad.returncode != 0 and "must be > 0" in bad.stderr
