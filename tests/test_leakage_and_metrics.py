# ============================================================
# PYTEST — Leakage check + shared metrics
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.leakage_check import detect_leakage
from src.metrics import tune_threshold, psi, recall_at_k, lift_at_k, ks_statistic
from src.data_loader import detect_feature_types


class TestDetectLeakage:

    def test_catches_identical_feature(self):
        X = pd.DataFrame({"a": [1, 0, 1, 0, 1]})
        y = pd.Series([1, 0, 1, 0, 1])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0
        assert any("a" in w for w in warnings)

    def test_no_false_positives_on_clean_data(self):
        np.random.seed(42)
        X = pd.DataFrame({"income": np.random.uniform(20, 200, 100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_high_correlation(self):
        vals = np.arange(50, dtype=float)
        X = pd.DataFrame({"leaky_feat": vals})
        y = pd.Series((vals > 25).astype(int))
        warnings = detect_leakage(X, y, threshold_corr=0.85)
        assert len(warnings) > 0

    def test_empty_dataframe_no_crash(self):
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)


class TestDetectFeatureTypes:

    def test_binary_detected(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 0], "target": [0, 1, 0, 0, 1]})
        _, _, bin_cols = detect_feature_types(df, target_col="target", threshold=10)
        assert "flag" in bin_cols

    def test_target_excluded(self):
        df = pd.DataFrame({"income": [10.0, 20.0, 30.0, 40.0, 50.0], "target": [0, 1, 0, 0, 1]})
        ord_cols, cont_cols, bin_cols = detect_feature_types(df, target_col="target", threshold=10)
        assert "target" not in ord_cols + cont_cols + bin_cols

    def test_continuous_detected(self):
        df = pd.DataFrame({
            "income": [10.5, 20.1, 55.3, 80.0, 110.5, 200.3, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
            "target": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })
        _, cont_cols, _ = detect_feature_types(df, target_col="target", threshold=10)
        assert "income" in cont_cols


class TestTuneThreshold:

    def test_returns_float_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)
        thr = tune_threshold(y_true, y_prob)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_perfect_prob_gives_low_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        thr = tune_threshold(y_true, y_prob)
        assert thr <= 0.9


class TestPSI:

    def test_identical_distributions_near_zero(self):
        x = np.random.RandomState(0).normal(0, 1, 500)
        assert psi(x, x) < 0.05

    def test_shifted_distribution_higher_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)
        assert psi(ref, new) > psi(ref, ref)

    def test_uses_reference_edges_not_actual(self):
        """Old-bug regression test: PSI must bin BOTH arrays on reference-only edges."""
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 500)
        new = rng.normal(5, 1, 500)
        assert psi(ref, new) > 0.20


class TestRecallAndLiftAtK:

    def test_recall_top_scores_captured(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        assert recall_at_k(y, p, k=0.3) >= 0.66

    def test_lift_above_one_for_good_model(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        assert lift_at_k(y, p, k=0.3) > 1.0

    def test_recall_at_k_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(recall_at_k(y, p), float)


class TestKSStatistic:

    def test_returns_float_in_range(self):
        np.random.seed(1)
        y = np.random.randint(0, 2, 100)
        p = np.random.uniform(0, 1, 100)
        ks = ks_statistic(y, p)
        assert isinstance(ks, float)
        assert 0.0 <= ks <= 1.0

    def test_perfect_model_high_ks(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        assert ks_statistic(y, p) > 0.7
