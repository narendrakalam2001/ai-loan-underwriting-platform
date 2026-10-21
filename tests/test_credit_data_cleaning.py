# ============================================================
# PYTEST — real-GMSC data-quality handling (regression tests)
# Findings from the real cs-training.csv (scripts/inspect_credit_data.py):
#   * 269 rows carry 96/98 placeholder codes in all 3 days-past-due columns
#   * where MonthlyIncome is missing, DebtRatio holds an absolute amount
#     (median ~1,159) and made `debt_ratio > 1` auto-decline ~19% of the file
# ============================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.data_loader import validate_credit_data, add_credit_engineered_features, LATE_SENTINEL_CODES
from src.credit_engine import score_applicant
from services.prediction_service import prepare_credit_features


def _raw(n=200, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "serious_dlqin2yrs": (rng.rand(n) < 0.07).astype(int),
        "revolving_utilization": rng.rand(n),
        "age": rng.randint(21, 80, n),
        "times_30_59_days_late": rng.choice([0, 0, 0, 1, 2], n),
        "debt_ratio": rng.rand(n) * 0.6 + 0.05,
        "monthly_income": rng.randint(2000, 12000, n).astype(float),
        "open_credit_lines": rng.randint(1, 12, n),
        "times_90_days_late": rng.choice([0, 0, 0, 1], n),
        "real_estate_loans": rng.randint(0, 3, n),
        "times_60_89_days_late": rng.choice([0, 0, 1], n),
        "dependents": rng.randint(0, 4, n).astype(float),
    })
    return df


def test_sentinel_codes_are_flagged_and_zeroed_in_all_three_columns():
    df = _raw()
    late = ["times_30_59_days_late", "times_60_89_days_late", "times_90_days_late"]
    df.loc[[3, 4], late] = 98
    df.loc[[7], late] = 96
    out = validate_credit_data(df)
    assert out["late_count_was_sentinel"].sum() == 3
    assert (out[late].max() < 90).all()
    assert (out.loc[out["late_count_was_sentinel"] == 1, late] == 0).all().all()


def test_genuine_large_counts_are_not_touched():
    df = _raw()
    df.loc[10, "times_90_days_late"] = 11
    df.loc[11, "times_30_59_days_late"] = 5
    out = validate_credit_data(df)
    assert out["late_count_was_sentinel"].sum() == 0
    assert 11 in out["times_90_days_late"].values and 5 in out["times_30_59_days_late"].values


def test_sentinel_rows_no_longer_trigger_severe_delinquency_rule():
    df = _raw()
    df.loc[3, ["times_30_59_days_late", "times_60_89_days_late", "times_90_days_late"]] = 98
    out = add_credit_engineered_features(validate_credit_data(df))
    row = out.iloc[3].to_dict()
    assert row["late_count_was_sentinel"] == 1
    assert score_applicant(row, prob=0.05)["rule_triggered"] != "SEVERE_DELINQUENCY_HISTORY"


def test_real_severe_delinquency_still_declines():
    df = _raw()
    df.loc[5, "times_90_days_late"] = 4
    out = add_credit_engineered_features(validate_credit_data(df))
    assert score_applicant(out.iloc[5].to_dict(), prob=0.05)["rule_triggered"] == "SEVERE_DELINQUENCY_HISTORY"


def test_debt_ratio_is_not_a_ratio_when_income_missing():
    df = _raw()
    df.loc[20:29, "monthly_income"] = np.nan
    df.loc[20:29, "debt_ratio"] = 1159.0                        # absolute amount, as in the real file
    df.loc[40, "debt_ratio"] = 2.5                              # genuine over-leverage, income present
    valid_median = df.loc[df["monthly_income"].notna(), "debt_ratio"].median()
    out = add_credit_engineered_features(validate_credit_data(df))
    assert (out.loc[20:29, "debt_ratio"] <= 1.0).all()
    assert out.loc[20:29, "debt_ratio"].iloc[0] == pytest.approx(valid_median)
    assert out.loc[20:29, "monthly_income_was_missing"].eq(1).all()
    assert out.loc[40, "debt_ratio"] == 2.5                     # real over-leverage is kept...
    assert score_applicant(out.loc[40].to_dict(), prob=0.05)["rule_triggered"] == "HIGH_DEBT_RATIO"  # ...and still declines
    assert score_applicant(out.loc[20].to_dict(), prob=0.05)["rule_triggered"] != "HIGH_DEBT_RATIO"


def test_dataset_without_sentinels_or_missing_income_is_unchanged():
    df = _raw()
    before = df[["debt_ratio", "times_90_days_late"]].copy()
    out = validate_credit_data(df)
    assert out["late_count_was_sentinel"].sum() == 0
    pd.testing.assert_frame_equal(out[["debt_ratio", "times_90_days_late"]], before)


def test_all_income_missing_does_not_crash():
    df = _raw()
    df["monthly_income"] = np.nan
    df["monthly_income"] = df["monthly_income"].astype(float)
    out = validate_credit_data(df)                                # no valid median available → leave as is, no crash
    assert out["monthly_income_was_missing"].eq(1).all()


# what serving/api.py's ApplicantInput always supplies (with its defaults)
API_INPUT = {"age": 35, "monthly_income": 50000.0, "existing_emis": 5000.0, "credit_utilization_pct": 20.0,
             "dependents": 1, "open_credit_lines": 3, "real_estate_loans": 0,
             "times_30_59_days_late": 0, "times_60_89_days_late": 0, "times_90_days_late": 0}


def test_serving_features_include_the_new_flag_set_to_zero():
    feats = prepare_credit_features(dict(API_INPUT))
    assert feats.loc[0, "late_count_was_sentinel"] == 0
    assert feats.loc[0, "debt_ratio"] == pytest.approx(0.1)


def test_training_and_serving_share_the_flag_columns():
    train_cols = set(add_credit_engineered_features(validate_credit_data(_raw())).columns)
    serve_cols = set(prepare_credit_features(dict(API_INPUT)).columns)
    for flag in ("monthly_income_was_missing", "dependents_was_missing", "late_count_was_sentinel"):
        assert flag in train_cols and flag in serve_cols


def test_sentinel_codes_constant():
    assert set(LATE_SENTINEL_CODES) == {96, 98}
