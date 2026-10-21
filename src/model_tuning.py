# ============================================================
# MODEL TUNING — AI Loan Underwriting Platform
# ============================================================
# One shared tuning engine (`tune_models`, `train_mlp_pipeline`)
# used by BOTH modules, with separate model grids because the
# two problems have very different imbalance profiles:
#   - Credit Risk : ~10% positive rate  → moderate imbalance
#   - Fraud       : <1% positive rate   → extreme imbalance
#     (scale_pos_weight / SMOTENC settings tuned accordingly)
# ============================================================

import os
import sys
import logging
import subprocess

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier, GradientBoostingClassifier,
                                    AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes   import GaussianNB, BernoulliNB
from xgboost               import XGBClassifier

from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# ── Native-library safety ──────────────────────────────────────────
# LightGBM / CatBoost are compiled libraries. On some Windows + numpy 2 setups
# LightGBM dies with "access violation" inside a joblib worker — a native crash that
# Python cannot catch in-process. That killed a ~3.5 hour training run at the very
# last model. So before using either library we fit a tiny model in a throw-away
# SUBPROCESS: if the subprocess crashes, the library is simply left out (and the
# log says so). You can also leave any model out by hand:
#     set EXCLUDE_MODELS=LightGBM,KNN        (comma-separated, case-insensitive)

_PROBE_CODE = {
    "lightgbm": (
        "import numpy as np\n"
        "from lightgbm import LGBMClassifier\n"
        "X = np.random.RandomState(0).rand(300, 6); y = (X[:, 0] > 0.5).astype(int)\n"
        "LGBMClassifier(n_estimators=5, verbose=-1, n_jobs=2).fit(X, y)\n"
    ),
    "catboost": (
        "import numpy as np\n"
        "from catboost import CatBoostClassifier\n"
        "X = np.random.RandomState(0).rand(300, 6); y = (X[:, 0] > 0.5).astype(int)\n"
        "CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False, thread_count=2).fit(X, y)\n"
    ),
}


def native_library_works(name: str, timeout: float = 120.0) -> bool:
    """True if a tiny fit of `name` ('lightgbm' | 'catboost') succeeds in a subprocess."""
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE_CODE[name]],
                           capture_output=True, timeout=timeout)
        if r.returncode != 0:
            logger.error("%s self-test FAILED (exit code %s) — it will be excluded from tuning. %s",
                         name, r.returncode, r.stderr.decode(errors="replace")[-200:].strip())
        return r.returncode == 0
    except Exception as e:
        logger.error("%s self-test could not run (%s) — it will be excluded from tuning", name, e)
        return False


def _adaboost_kwargs() -> dict:
    # scikit-learn 1.4/1.5 print a FutureWarning (once per CV fit — hundreds of lines) unless
    # algorithm="SAMME" is given; from 1.6 the parameter itself is deprecated, so only pass it before.
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    return {"algorithm": "SAMME"} if (major, minor) < (1, 6) else {}


_ADABOOST_KW = _adaboost_kwargs()


def _excluded_models() -> set:
    return {m.strip().lower() for m in os.getenv("EXCLUDE_MODELS", "").split(",") if m.strip()}


try:
    from lightgbm import LGBMClassifier
    if not native_library_works("lightgbm"):
        LGBMClassifier = None
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
    if not native_library_works("catboost"):
        CatBoostClassifier = None
except Exception:
    CatBoostClassifier = None

from src.config        import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS, SELECT_K, SELECT_K_MAX
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# MODULE 1 — CREDIT RISK MODEL GRIDS  (~10% positive rate)
# ============================================================

credit_scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        {
            "classifier__penalty":      ["l1", "l2"],
            "classifier__C":            [0.01, 0.1, 1, 10],
            "classifier__solver":       ["liblinear", "saga"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "classifier__n_neighbors": [3, 5, 9],
            "classifier__weights":     ["uniform", "distance"],
        }
    ),
    "SGD": (
        SGDClassifier(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3),
        {
            "classifier__loss":         ["log_loss"],
            "classifier__alpha":        [1e-4, 1e-3, 1e-2],
            "classifier__penalty":      ["l2", "elasticnet"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),
    "GaussianNB": (GaussianNB(), {}),
}

credit_unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__max_depth":        [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    ),
    "RandomForest": (
        RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators":     [100, 200],
            "classifier__max_depth":        [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2],
        }
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [None, 10, 20],
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [3, 5],
            "classifier__subsample":     [0.8, 1.0],
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE, **_ADABOOST_KW),
        {
            "classifier__n_estimators":  [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 1.0],
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":     [100, 200],
            "classifier__learning_rate":    [0.05, 0.1],
            "classifier__max_depth":        [3, 5],
            "classifier__scale_pos_weight": [5, 9, 15],   # ~neg/pos ratio at 10% positive rate
        }
    ),
    "BernoulliNB": (BernoulliNB(), {}),
}

if LGBMClassifier is not None:
    credit_unscaled_models["LightGBM"] = (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, is_unbalance=True),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [-1, 10],
            "classifier__num_leaves":    [31, 63],
        }
    )

if CatBoostClassifier is not None:
    credit_unscaled_models["CatBoost"] = (
        # allow_writing_files=False: CatBoost otherwise creates a "catboost_info"
        # working directory on disk. With n_jobs=-1, multiple parallel worker
        # processes try to create that same folder simultaneously, which fails
        # with "Can't create train working dir: catboost_info" on Windows. We
        # don't need those training-log files, so disable them entirely.
        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, auto_class_weights="Balanced",
                            allow_writing_files=False),
        {
            "classifier__iterations":    [100, 200],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__depth":         [4, 6],
        }
    )


# ============================================================
# MODULE 2 — FRAUD MODEL GRIDS  (<1% positive rate → extreme imbalance)
# ============================================================

fraud_scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        {
            "classifier__C":            [0.01, 0.1, 1, 10],
            "classifier__penalty":      ["l2"],
            "classifier__class_weight": ["balanced"],
        }
    ),
    "SGD": (
        SGDClassifier(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3),
        {
            "classifier__loss":         ["log_loss"],
            "classifier__alpha":        [1e-4, 1e-3],
            "classifier__class_weight": ["balanced"],
        }
    ),
    "GaussianNB": (GaussianNB(), {}),
}

fraud_unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "RandomForest": (
        RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced_subsample"),
        {
            "classifier__n_estimators":     [200, 300],
            "classifier__max_depth":        [None, 12, 20],
            "classifier__min_samples_leaf": [1, 2],
        }
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth":    [None, 12, 20],
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":     [200, 300],
            "classifier__learning_rate":    [0.05, 0.1],
            "classifier__max_depth":        [3, 5],
            # fraud positive rate is typically < 1% → much higher scale_pos_weight
            "classifier__scale_pos_weight": [50, 100, 200],
        }
    ),
}

if LGBMClassifier is not None:
    fraud_unscaled_models["LightGBM"] = (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, is_unbalance=True),
        {
            "classifier__n_estimators":  [200, 300],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__num_leaves":    [31, 63],
        }
    )


# ============================================================
# TUNE MODELS  (shared by both engines)
# ============================================================

def tune_models(
    models:       Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor: ColumnTransformer,
    cat_indices:  List[int],
    X_train:      "pd.DataFrame",
    y_train:      "pd.Series",
    use_smote:    bool = True,
    selector_k:   int = SELECT_K,
) -> Dict[str, ImbPipeline]:
    """
    For each model: preprocessor → [SMOTENC] → SelectKBest → classifier.
    Tuned with RandomizedSearchCV (scoring = F1).
    Returns dict of {model_name: best_pipeline}.
    """
    final_pipelines: Dict[str, ImbPipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    # SMOTENC needs at least a handful of minority samples per CV fold
    min_class_count = y_train.value_counts().min()
    n_splits = min(CV_FOLDS, max(2, int(min_class_count)))
    if n_splits < CV_FOLDS:
        logger.warning("Reducing CV folds to %d due to small minority class (n=%d)",
                        n_splits, min_class_count)

    excluded = _excluded_models()
    skipped: Dict[str, str] = {}

    for name, (clf, param_dist) in models.items():
        if name.lower() in excluded:
            logger.warning("Skipping %s (listed in EXCLUDE_MODELS)", name)
            skipped[name] = "excluded by EXCLUDE_MODELS"
            continue
        logger.info("Tuning: %s", name)

        steps = [("preprocessor", preprocessor)]

        if use_smote and len(cat_indices) > 0 and min_class_count >= 2:
            k_neighbors = max(1, min(5, min_class_count - 1))
            steps.append((
                "smote",
                SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE,
                         k_neighbors=k_neighbors)
            ))

        steps.append(("selector", SelectKBest(mutual_info_classif, k=k_safe)))
        steps.append(("classifier", clf))

        pipe = ImbPipeline(steps)

        n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1",
            cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=0,
            error_score="raise",
        )

        try:
            search.fit(X_train, y_train)
        except Exception as e:          # one broken model must not waste hours of finished work
            logger.error("Tuning FAILED for %s — skipping it and continuing: %s: %s",
                         name, type(e).__name__, str(e)[:200])
            skipped[name] = f"{type(e).__name__}"
            continue

        logger.info("%s best params: %s", name, search.best_params_)
        final_pipelines[name] = search.best_estimator_

    if skipped:
        logger.warning("Models NOT included in this run: %s", skipped)
    if not final_pipelines:
        raise RuntimeError(f"No model could be tuned successfully: {skipped}")

    return final_pipelines


# ============================================================
# NEURAL NETWORK — trained separately (no CV search)
# ============================================================

def train_mlp_pipeline(X_train, y_train, preprocessor, cat_indices: List[int]):
    """MLP trained outside RandomizedSearchCV — training time makes CV search impractical."""
    from sklearn.neural_network import MLPClassifier

    logger.info("Training Neural Network (MLP) ...")

    min_class_count = y_train.value_counts().min()

    steps = [("preprocessor", preprocessor)]
    if len(cat_indices) > 0 and min_class_count >= 2:
        k_neighbors = max(1, min(5, min_class_count - 1))
        steps.append((
            "smote",
            SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE,
                     k_neighbors=k_neighbors)
        ))

    steps.append(("classifier", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=min(512, max(8, len(X_train) // 4)),
        learning_rate="adaptive",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
    )))

    pipe = ImbPipeline(steps)
    pipe.fit(X_train, y_train)

    logger.info("MLP training done")
    return pipe
