import numpy as np, pandas as pd, tempfile
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation import select_best_model, compute_feature_importance


def _row(name, f1, pr, roc):
    return dict(model=name, train_test_gap=0.0, cv_std_f1=0.01, f1=f1, precision=0.9,
                recall=0.8, pr_auc=pr, roc_auc=roc)


def test_selection_prefers_pr_auc_over_noisy_f1():
    summary = pd.DataFrame([_row("A", 0.80, 0.70, 0.90), _row("B", 0.79, 0.75, 0.96)])
    pipes = {"A": object(), "B": object()}
    name, _ = select_best_model(summary, pipes, {}, {})
    assert name == "B"


def test_permutation_importance_uses_real_column_names():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=["a", "signal", "c"])
    y = pd.Series((X["signal"] + 0.2 * rng.normal(size=600) > 0.8).astype(int))
    pipe = Pipeline([("preprocessor", StandardScaler()),
                     ("classifier", MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=0))])
    pipe.fit(X, y)
    fi = compute_feature_importance(pipe, X, y)
    assert fi is not None and fi.index[0] == "signal"
