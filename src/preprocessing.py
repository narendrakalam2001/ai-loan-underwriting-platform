# ============================================================
# PREPROCESSING — AI Loan Underwriting Platform
# ============================================================
# One shared preprocessing module used by BOTH the Credit Risk
# Engine and the Fraud Detection Engine, so the two models are
# built on an identical, audited transformation contract.
# ============================================================

import numpy as np
import pandas as pd
import logging

from typing import List, Tuple

from sklearn.base              import BaseEstimator, TransformerMixin, clone
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.config import CLIP_FOLD, SELECT_K

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER — IQR-based outlier clipping transformer
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips values to [Q1 - fold*IQR, Q3 + fold*IQR].
    Fitted on train set only, applied to train + test/inference.
    Prevents outlier leakage through scaling.

    get_feature_names_out() is implemented so sklearn's
    ColumnTransformer can extract clean feature names (avoids the
    generic f0, f1... fallback warning).
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q1  = np.quantile(X, 0.25, axis=0)
        q3  = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr

        eps         = 1e-9
        self.upper_ = np.where(self.upper_ == self.lower_, self.upper_ + eps, self.upper_)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# SAFE FEATURE COUNT HELPER
# ============================================================

def safe_k(requested_k: int, preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> int:
    """Returns min(requested_k, actual_output_features_of_preprocessor)."""
    fitted = preprocessor.fit(X_sample)

    try:
        feat_names = fitted.get_feature_names_out()
    except Exception:
        arr = fitted.transform(X_sample)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        feat_names = [f"f{i}" for i in range(arr.shape[1])]

    k = min(requested_k, max(1, len(feat_names)))
    logger.info("safe_k: requested=%d available=%d using=%d", requested_k, len(feat_names), k)
    return k


# ============================================================
# PREPROCESSOR BUILDER (shared by credit_engine + fraud_engine)
# ============================================================

def build_preprocessors(
    ord_cols:  List[str],
    cont_cols: List[str],
    bin_cols:  List[str],
    X_train:   pd.DataFrame,
    clip_fold: float = CLIP_FOLD
) -> Tuple[ColumnTransformer, ColumnTransformer, List[int], List[str]]:
    """
    Returns:
        preprocessor_scaled   — for distance/linear models (LR, SVM, KNN, SGD, NB)
        preprocessor_unscaled — for tree models (RF, XGB, LGBM, CatBoost, ExtraTrees)
        categorical_indices   — indices for SMOTE/SMOTENC
        feature_order         — column order after transform
    """
    skewed     = [c for c in cont_cols if abs(X_train[c].skew()) > 0.8]
    non_skewed = [c for c in cont_cols if c not in skewed]

    logger.info("Skewed cols (%d): %s", len(skewed), skewed)
    logger.info("Non-skewed cols (%d): %s", len(non_skewed), non_skewed)

    scaled_transformers = []
    if skewed:
        scaled_transformers.append((
            "skewed",
            Pipeline([
                ("clip",  Clipper(fold=clip_fold)),
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", StandardScaler())
            ]),
            skewed
        ))
    if non_skewed:
        scaled_transformers.append((
            "non_skew",
            Pipeline([
                ("clip",  Clipper(fold=clip_fold)),
                ("scale", StandardScaler())
            ]),
            non_skewed
        ))

    unscaled_transformers = []
    if skewed:
        unscaled_transformers.append((
            "skewed", Pipeline([("clip", Clipper(fold=clip_fold))]), skewed
        ))
    if non_skewed:
        unscaled_transformers.append((
            "non_skew", Pipeline([("clip", Clipper(fold=clip_fold))]), non_skewed
        ))

    # clone(ord_pipeline) pattern: two independent OrdinalEncoder copies so
    # preprocessor_scaled and preprocessor_unscaled never share fitted state.
    def _ord_pipeline():
        return Pipeline([
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("ord", clone(_ord_pipeline()), ord_cols),
            *scaled_transformers,
            ("bin", "passthrough", bin_cols)
        ],
        remainder="drop"
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("ord", clone(_ord_pipeline()), ord_cols),
            *unscaled_transformers,
            ("bin", "passthrough", bin_cols)
        ],
        remainder="drop"
    )

    feature_order        = ord_cols + skewed + non_skewed + bin_cols
    categorical_set      = set(ord_cols + bin_cols)
    categorical_indices  = [i for i, c in enumerate(feature_order) if c in categorical_set]

    logger.info("Feature order: %s", feature_order)
    logger.info("Categorical indices for SMOTENC: %s", categorical_indices)

    return preprocessor_scaled, preprocessor_unscaled, categorical_indices, feature_order
