# ============================================================
# PYTEST — model_tuning safety nets (native-library crash, EXCLUDE_MODELS)
# Regression for a real incident: LightGBM "access violation" killed a 3.5 h
# training run and nothing was saved.
# ============================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import model_tuning
from src.model_tuning import tune_models, native_library_works, _excluded_models


class CrashingClassifier(BaseEstimator, ClassifierMixin):
    """Stands in for a native library that blows up inside fit (like LightGBM did)."""
    def __init__(self, c=1.0):
        self.c = c

    def fit(self, X, y):
        raise OSError("exception: access violation reading 0x0000000000000000")


def _data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(200, 4), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.rand(200) * 0.3 > 0.6).astype(int))
    pre = ColumnTransformer([("num", StandardScaler(), list("abcd"))])
    return X, y, pre


def _models():
    return {
        "LogisticRegression": (LogisticRegression(max_iter=200), {"classifier__C": [0.1, 1.0]}),
        "Crasher": (CrashingClassifier(), {"classifier__c": [1.0, 2.0]}),
    }


def test_crashing_model_is_skipped_and_others_survive():
    X, y, pre = _data()
    out = tune_models(_models(), pre, [], X, y, use_smote=False, selector_k=3)
    assert set(out) == {"LogisticRegression"}


def test_all_models_failing_raises_clear_error():
    X, y, pre = _data()
    with pytest.raises(RuntimeError, match="No model could be tuned"):
        tune_models({"Crasher": _models()["Crasher"]}, pre, [], X, y, use_smote=False, selector_k=3)


def test_exclude_models_env_var(monkeypatch):
    monkeypatch.setenv("EXCLUDE_MODELS", " logisticregression , KNN ")
    assert _excluded_models() == {"logisticregression", "knn"}
    X, y, pre = _data()
    with pytest.raises(RuntimeError):                       # the only healthy model was excluded
        tune_models({"LogisticRegression": _models()["LogisticRegression"]}, pre, [], X, y,
                    use_smote=False, selector_k=3)


def test_probe_reports_failure_when_subprocess_crashes(monkeypatch):
    monkeypatch.setitem(model_tuning._PROBE_CODE, "lightgbm", "import os; os._exit(139)")
    assert native_library_works("lightgbm") is False


def test_probe_reports_success_for_healthy_code(monkeypatch):
    monkeypatch.setitem(model_tuning._PROBE_CODE, "lightgbm", "x = 1")
    assert native_library_works("lightgbm") is True
