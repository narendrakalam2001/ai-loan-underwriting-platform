"""explain_single_prediction must work for XGBoost even when `shap` is not installed
(the lean production image does not ship it)."""
import numpy as np
import pandas as pd
import pytest

xgboost = pytest.importorskip("xgboost")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.evaluation as ev


def _fit():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(400, 4), columns=["a", "b", "c", "d"])
    y = (X["a"] * 2 - X["b"] + rng.randn(400) * 0.3 > 0).astype(int)
    pipe = Pipeline([("preprocessor", StandardScaler()),
                     ("classifier", xgboost.XGBClassifier(n_estimators=30, max_depth=3, verbosity=0))])
    pipe.fit(X, y)
    return pipe, X


def test_xgboost_drivers_returned_without_shap(monkeypatch):
    monkeypatch.setattr(ev, "SHAP_AVAILABLE", False)
    pipe, X = _fit()
    out = ev.explain_single_prediction(pipe, X, X.iloc[[0]], top_k=3)
    assert len(out) == 3
    assert {"feature", "shap_value"} <= set(out[0])
    assert out[0]["feature"] in ("a", "b")          # the two informative features lead
    assert abs(out[0]["shap_value"]) >= abs(out[1]["shap_value"])


def test_native_contribs_sum_to_model_margin(monkeypatch):
    monkeypatch.setattr(ev, "SHAP_AVAILABLE", False)
    pipe, X = _fit()
    clf = pipe.named_steps["classifier"]
    Xs = pipe.named_steps["preprocessor"].transform(X.iloc[[5]])
    c = clf.get_booster().predict(xgboost.DMatrix(Xs), pred_contribs=True)
    margin = clf.get_booster().predict(xgboost.DMatrix(Xs), output_margin=True)
    assert np.isclose(c.sum(), margin[0], atol=1e-4)
