"""The operating threshold must come from the validation split, not the test set."""
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.evaluation import evaluate_models
from src.metrics import tune_threshold


def _data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)))
    y = pd.Series((X[0] + rng.normal(size=n) > 1.6).astype(int))
    return X, y


def test_threshold_taken_from_validation_split_not_test():
    X, y = _data()
    Xtr, ytr = X[:2000], y[:2000]
    Xv, yv = X[2000:3000], y[2000:3000]
    Xte, yte = X[3000:], y[3000:]
    m = LogisticRegression().fit(Xtr, ytr)

    s = evaluate_models({"LR": m}, Xtr, ytr, Xte, yte, model_dir=tempfile.mkdtemp(),
                        X_thr=Xv, y_thr=yv)
    thr_val = tune_threshold(yv.values, m.predict_proba(Xv)[:, 1])
    thr_test = tune_threshold(yte.values, m.predict_proba(Xte)[:, 1])

    assert abs(s.loc[0, "threshold"] - round(thr_val, 4)) < 1e-3
    assert thr_val != thr_test  # otherwise this test proves nothing
    exp = accuracy_score(yte, (m.predict_proba(Xte)[:, 1] >= thr_val).astype(int))
    assert abs(s.loc[0, "test_acc"] - exp) < 1e-4


def test_falls_back_to_test_threshold_when_no_validation_given():
    X, y = _data(1)
    m = LogisticRegression().fit(X[:2000], y[:2000])
    s = evaluate_models({"LR": m}, X[:2000], y[:2000], X[2000:], y[2000:],
                        model_dir=tempfile.mkdtemp())
    thr_test = tune_threshold(y[2000:].values, m.predict_proba(X[2000:])[:, 1])
    assert abs(s.loc[0, "threshold"] - round(thr_test, 4)) < 1e-3


def test_cv_f1_is_scored_at_operating_threshold_not_default_cutoff():
    """Balanced class weights make 0.5-cutoff F1 tiny; CV F1 must track the tuned-threshold F1."""
    rng = np.random.default_rng(7)
    n = 6000
    X = pd.DataFrame(rng.normal(size=(n, 4)))
    y = pd.Series((X[0] * 1.5 + rng.normal(size=n) > 3.6).astype(int))   # <1% positives
    Xtr, ytr, Xv, yv, Xte, yte = X[:3000], y[:3000], X[3000:4500], y[3000:4500], X[4500:], y[4500:]
    m = LogisticRegression(class_weight="balanced").fit(Xtr, ytr)

    s = evaluate_models({"LR": m}, Xtr, ytr, Xte, yte, model_dir=tempfile.mkdtemp(),
                        X_thr=Xv, y_thr=yv, cv_folds=3)
    row = s.iloc[0]
    from sklearn.model_selection import cross_val_score
    naive = cross_val_score(m, Xtr, ytr, scoring="f1", cv=3).mean()
    assert row["cv_mean_f1"] > naive + 0.05          # old metric was badly understated
    assert row["cv_test_gap"] < 0.30                  # and is now comparable with test F1
