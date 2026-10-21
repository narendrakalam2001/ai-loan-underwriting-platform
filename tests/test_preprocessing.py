# ============================================================
# PYTEST — Preprocessing (Clipper, build_preprocessors, safe_k)
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.preprocessing import Clipper, build_preprocessors, safe_k


class TestClipper:

    def test_fit_transform_shape(self):
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        """Leakage prevention: Clipper must use TRAIN bounds on test data."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test = np.array([[100.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0


class TestBuildPreprocessors:

    def _sample_df(self):
        return pd.DataFrame({
            "age":       [25, 35, 45, 55, 65, 30, 40, 50],
            "income":    [30.0, 60.0, 90.0, 120.0, 150.0, 45.0, 75.0, 105.0],
            "family":    [1, 2, 3, 4, 5, 2, 3, 1],
            "education": [1, 2, 3, 1, 2, 3, 1, 2],
            "online":    [0, 1, 0, 1, 0, 1, 0, 1],
        })

    def test_returns_four_outputs(self):
        df = self._sample_df()
        result = build_preprocessors(["education", "family"], ["age", "income"], ["online"], df)
        assert len(result) == 4

    def test_categorical_indices_are_list(self):
        df = self._sample_df()
        _, _, cat_idx, _ = build_preprocessors(["education"], ["age", "income"], ["online"], df)
        assert isinstance(cat_idx, list)

    def test_feature_order_coverage(self):
        df = self._sample_df()
        ord_cols, cont_cols, bin_cols = ["education", "family"], ["age", "income"], ["online"]
        _, _, _, feat_order = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        for col in ord_cols + cont_cols + bin_cols:
            assert col in feat_order

    def test_scaled_preprocessor_transforms_without_nan(self):
        df = self._sample_df()
        pre_scaled, _, _, _ = build_preprocessors(["education"], ["age", "income"], ["online"], df)
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_unscaled_preprocessor_transforms_without_nan(self):
        df = self._sample_df()
        _, pre_unscaled, _, _ = build_preprocessors(["education"], ["age", "income"], ["online"], df)
        out = pre_unscaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_safe_k_never_exceeds_available_features(self):
        df = self._sample_df()
        pre_scaled, _, _, _ = build_preprocessors(["education"], ["age", "income"], ["online"], df)
        k = safe_k(requested_k=1000, preprocessor=pre_scaled, X_sample=df)
        assert k <= pre_scaled.fit_transform(df).shape[1]
