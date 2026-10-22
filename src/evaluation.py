# ============================================================
# EVALUATION — AI Loan Underwriting Platform
# ============================================================
# Shared evaluation toolkit used by BOTH the Credit Risk Engine
# and the Fraud Detection Engine. Every function that needs to
# know "which engine" takes an explicit `model_dir` / `project`
# argument rather than hardcoding a path, so the same code trains,
# scores and saves either model.
# ============================================================

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection      import permutation_importance
from sklearn.pipeline        import Pipeline

from src.config  import RANDOM_STATE, N_JOBS, CV_FOLDS
from src.metrics import tune_threshold, ks_statistic

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT PROBA
# ============================================================

def safe_predict_proba(pipe, X):
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        logger.warning("predict_proba failed — falling back to manual extraction")
        try:
            pre = pipe.named_steps["preprocessor"].transform(X)
            sel = pipe.named_steps.get("selector", None)
            X_in = sel.transform(pre) if sel is not None else pre
            clf = pipe.named_steps["classifier"]
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(X_in)[:, 1]
        except Exception as e:
            logger.error("safe_predict_proba fully failed: %s", e)
    return None


# ============================================================
# EVALUATE ALL MODELS — summary table
# ============================================================

def f1_at_threshold_scorer(thr):
    """CV scorer: F1 at the model's operating threshold `thr`.

    cross_val_score(scoring="f1") uses predict(), i.e. the default 0.5 cutoff.
    For models trained with SMOTE / class weights that cutoff yields many false
    positives, so CV F1 (~0.1) looked unrelated to the tuned-threshold test F1
    (~0.8). Scoring at the same threshold keeps the two comparable.
    Falls back to plain f1 when the model has no probabilities (thr is None).
    """
    if thr is None:
        return "f1"

    def _scorer(estimator, X, y):
        prob = safe_predict_proba(estimator, X)
        if prob is None:
            return f1_score(y, estimator.predict(X), zero_division=0)
        return f1_score(y, (prob >= thr).astype(int), zero_division=0)

    return _scorer


def evaluate_models(
    pipelines: Dict,
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    model_dir: str,
    cv_folds:  int = CV_FOLDS,
    X_thr:     pd.DataFrame = None,
    y_thr:     pd.Series = None,
) -> pd.DataFrame:
    """X_thr/y_thr: held-out data used ONLY to pick each model's operating
    threshold. The test set is then used only to report metrics at that
    threshold. If omitted, the threshold falls back to the test set (old,
    optimistic behaviour — kept for backwards compatibility)."""

    rows = []
    n_splits = min(cv_folds, max(2, int(y_train.value_counts().min())))

    for name, pipe in pipelines.items():
        logger.info("Evaluating: %s", name)

        # Every metric in the row (accuracy included) uses the SAME operating
        # threshold, so the table agrees with the classification report.
        y_prob = safe_predict_proba(pipe, X_test)
        if y_prob is not None:
            if X_thr is not None and y_thr is not None:
                thr_prob = safe_predict_proba(pipe, X_thr)
                thr = tune_threshold(np.asarray(y_thr), thr_prob)
            else:
                thr = tune_threshold(y_test.values, y_prob)
            y_pred = (y_prob >= thr).astype(int)
        else:
            thr = None
            y_pred = pipe.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        try:
            if thr is not None:
                train_prob = safe_predict_proba(pipe, X_train)
                train_pred = (train_prob >= thr).astype(int)
            else:
                train_pred = pipe.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
        except Exception:
            train_acc = None

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else None
        ks = ks_statistic(y_test, y_prob) if y_prob is not None else None
        brier = brier_score_loss(y_test, y_prob) if y_prob is not None else None

        try:
            cv_scores = cross_val_score(
                pipe, X_train, y_train, scoring=f1_at_threshold_scorer(thr),
                cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=N_JOBS,
            )
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = None, None

        rows.append({
            "model":          name,
            "train_acc":      round(train_acc, 4) if train_acc else None,
            "test_acc":       round(test_acc, 4),
            "train_test_gap": round(abs((train_acc or 0) - test_acc), 4),
            "cv_mean_f1":     round(cv_mean, 4) if cv_mean else None,
            "cv_std_f1":      round(cv_std, 4) if cv_std else None,
            "cv_test_gap":    round(abs((cv_mean or 0) - f1), 4),
            "precision":      round(prec, 4),
            "recall":         round(rec, 4),
            "f1":             round(f1, 4),
            "roc_auc":        round(roc_auc, 4) if roc_auc else None,
            "pr_auc":         round(pr_auc, 4) if pr_auc else None,
            "ks":             round(ks, 4) if ks else None,
            "brier":          round(brier, 4) if brier else None,
            "threshold":      round(thr, 4) if thr else None,
        })

    summary = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    os.makedirs(model_dir, exist_ok=True)
    summary.to_csv(os.path.join(model_dir, "model_experiment_results.csv"), index=False)
    return summary


# ============================================================
# SELECT BEST MODEL — with generalization filter
# ============================================================

def select_best_model(
    summary:        pd.DataFrame,
    pipelines:      Dict,
    scaled_pipes:   Dict,
    unscaled_pipes: Dict,
) -> Tuple[str, object]:

    def _filter(df, gen_gap, cv_std, min_f1, min_prec, min_rec):
        mask = (
            df["train_test_gap"].notna() & df["cv_std_f1"].notna() &
            (df["train_test_gap"] <= gen_gap) & (df["cv_std_f1"] <= cv_std) &
            (df["f1"] >= min_f1) & (df["precision"] >= min_prec) & (df["recall"] >= min_rec)
        )
        return df[mask].copy()

    thresholds = [
        (0.05, 0.05, 0.70, 0.60, 0.60),
        (0.08, 0.07, 0.65, 0.55, 0.55),
        (0.10, 0.10, 0.60, 0.50, 0.50),
        (0.20, 0.20, 0.00, 0.00, 0.00),
        (1.00, 1.00, 0.00, 0.00, 0.00),   # final fallback — always non-empty
    ]

    candidates = pd.DataFrame()
    for gg, cvs, mf1, mp, mr in thresholds:
        candidates = _filter(summary, gg, cvs, mf1, mp, mr)
        if not candidates.empty:
            logger.info("Candidates found with gen_gap<=%.2f cv_std<=%.2f min_f1>=%.2f", gg, cvs, mf1)
            break

    if candidates.empty:
        selected_name = summary.iloc[0]["model"]
    else:
        # Rank by PR-AUC first: it is threshold-free and the standard metric for
        # imbalanced data. F1 at a chosen threshold, measured on a handful of
        # positives (e.g. 95 frauds in the test split), differs by 1-2 cases
        # between models and is mostly noise.
        candidates = candidates.assign(_pr=candidates["pr_auc"].fillna(0.0))
        candidates = candidates.sort_values(
            ["_pr", "f1", "roc_auc", "cv_std_f1"], ascending=[False, False, False, True]
        ).drop(columns="_pr").reset_index(drop=True)
        selected_name = candidates.iloc[0]["model"]

    logger.info("Selected model: %s", selected_name)

    selected_pipe = pipelines.get(selected_name) or scaled_pipes.get(selected_name) or unscaled_pipes.get(selected_name)
    if selected_pipe is None:
        raise RuntimeError(f"Selected model '{selected_name}' not found in any pipeline dict")

    return selected_name, selected_pipe


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_with_holdout(selected_pipe, X_cal, y_cal, method: str = "isotonic"):
    clf = selected_pipe.named_steps["classifier"]
    pre = selected_pipe.named_steps["preprocessor"]
    sel = selected_pipe.named_steps.get("selector", None)

    X_pre = pre.transform(X_cal)
    X_sel = sel.transform(X_pre) if sel is not None else X_pre

    method_use = method
    if method == "isotonic" and len(y_cal) < 200:
        method_use = "sigmoid"
        logger.info("Calibration: switching to sigmoid (cal set < 200 rows)")

    try:
        try:
            # sklearn >= 1.6: cv="prefit" was removed in favor of wrapping the
            # already-fitted estimator in FrozenEstimator.
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method_use)
        except ImportError:
            try:
                cal = CalibratedClassifierCV(estimator=clf, cv="prefit", method=method_use)
            except TypeError:
                cal = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method=method_use)

        cal.fit(X_sel, y_cal)

        steps = [("preprocessor", pre)]
        if sel is not None:
            steps.append(("selector", sel))
        steps.append(("classifier", cal))

        cal_pipe = Pipeline(steps)
        logger.info("Calibration done | method=%s", method_use)
        return cal_pipe

    except Exception as e:
        logger.exception("Calibration failed: %s", e)
        return None


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(selected_pipe, X_train, y_train, top_k: int = 20):
    clf = _unwrap_classifier(selected_pipe.named_steps["classifier"])
    names = _get_feature_names(selected_pipe, X_train)

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
        names = names if names and len(names) == len(imp) else [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False).head(top_k)

    if hasattr(clf, "coef_"):
        coef = np.abs(clf.coef_).ravel()
        names = names if names and len(names) == len(coef) else [f"f{i}" for i in range(len(coef))]
        return pd.Series(coef, index=names).sort_values(ascending=False).head(top_k)

    logger.info("Using permutation importance (slow fallback)...")
    # Plain int numpy labels + a capped sample: avoids pandas/dtype quirks in the
    # joblib workers and keeps this fallback fast on 100k+ row datasets.
    y_arr = np.asarray(y_train).astype(int)
    if len(y_arr) > 20000:
        rng = np.random.RandomState(RANDOM_STATE)
        keep = rng.choice(len(y_arr), 20000, replace=False)
        X_use, y_use = X_train.iloc[keep], y_arr[keep]
    else:
        X_use, y_use = X_train, y_arr
    cols = list(X_use.columns)
    for n_jobs in (N_JOBS, 1):          # retry single-process if workers fail
        try:
            r = permutation_importance(
                selected_pipe, X_use, y_use, n_repeats=5, scoring="average_precision",
                n_jobs=n_jobs, random_state=RANDOM_STATE,
            )
            idx = np.argsort(r.importances_mean)[::-1][:top_k]
            return pd.Series(r.importances_mean[idx], index=[cols[i] for i in idx])
        except Exception as e:
            logger.warning("Permutation importance failed (n_jobs=%s): %s", n_jobs, e)
    logger.error("Permutation importance failed in all attempts; feature importance omitted")
    return None


def _unwrap_classifier(clf):
    """
    Both compute_shap() and explain_single_prediction() need the ORIGINAL
    fitted estimator (e.g. XGBClassifier), but after calibrate_with_holdout()
    the pipeline's 'classifier' step is a CalibratedClassifierCV wrapping it.
    This unwraps one level so tree-based SHAP explainers still fire on the
    calibrated production model.
    """
    if type(clf).__name__ == "CalibratedClassifierCV" and getattr(clf, "calibrated_classifiers_", None):
        inner = clf.calibrated_classifiers_[0]
        estimator = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None) or clf
        # sklearn >= 1.6 wraps the prefit estimator in FrozenEstimator — unwrap once more
        if type(estimator).__name__ == "FrozenEstimator":
            estimator = getattr(estimator, "estimator", estimator)
        return estimator
    return clf


def _get_feature_names(pipe, X_sample):
    try:
        pre = pipe.named_steps["preprocessor"]
        raw_names = list(pre.get_feature_names_out())

        clean_names = []
        for n in raw_names:
            clean_names.append(n.split("__", 1)[1] if "__" in n else n)

        sel = pipe.named_steps.get("selector", None)
        if sel is not None:
            mask = sel.get_support()
            clean_names = [n for n, m in zip(clean_names, mask) if m]

        return clean_names
    except Exception as e:
        logger.warning("Could not extract feature names: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(selected_pipe, X_train, X_explain, max_explain: int = 200):
    if not SHAP_AVAILABLE:
        logger.info("SHAP not installed — skipping")
        return None

    clf = _unwrap_classifier(selected_pipe.named_steps["classifier"])
    pre = selected_pipe.named_steps["preprocessor"]
    sel = selected_pipe.named_steps.get("selector", None)

    try:
        X_pre = pre.transform(X_train)
        X_tr = sel.transform(X_pre) if sel is not None else X_pre
        X_ex = sel.transform(pre.transform(X_explain)) if sel is not None else pre.transform(X_explain)
        X_ex = np.asarray(X_ex)[:max_explain]

        names = _get_feature_names(selected_pipe, X_train)

        tree_types = ("RandomForestClassifier", "ExtraTreesClassifier",
                      "GradientBoostingClassifier", "XGBClassifier",
                      "LGBMClassifier", "CatBoostClassifier", "DecisionTreeClassifier",
                      "AdaBoostClassifier")

        if type(clf).__name__ in tree_types:
            try:
                explainer = shap.TreeExplainer(clf)
                try:
                    vals = np.array(explainer(X_ex).values)
                except Exception:
                    vals = np.array(explainer.shap_values(X_ex))
                    if isinstance(vals, list):
                        vals = vals[1]
            except ValueError as e:
                # Known shap<->xgboost version-mismatch bug: some XGBoost builds
                # serialize `base_score` as a string like "[1.5545572E-1]" instead
                # of a plain float, which older/newer shap TreeExplainer builds
                # fail to parse ("could not convert string to float"). Rather than
                # require an exact shap/xgboost version pin, fall back to
                # XGBoost's OWN native SHAP computation (pred_contribs=True),
                # which bypasses the shap package's XGBoost parsing entirely and
                # gives numerically-identical SHAP values.
                if type(clf).__name__ == "XGBClassifier" and "could not convert string to float" in str(e):
                    logger.warning("shap.TreeExplainer failed on XGBoost base_score parsing (%s) — "
                                    "falling back to XGBoost's native pred_contribs", e)
                    booster = clf.get_booster()
                    dmat = __import__("xgboost").DMatrix(X_ex)
                    contribs = booster.predict(dmat, pred_contribs=True)
                    vals = contribs[:, :-1]   # drop the trailing bias/expected-value column

                    class _NativeXGBExplainer:  # placeholder so the logging/return below stays uniform
                        pass
                    explainer = _NativeXGBExplainer()
                else:
                    raise
        else:
            bg_sample = shap.sample(X_tr, min(200, len(X_tr)))

            def model_fn(x):
                return clf.predict_proba(x)[:, 1]

            explainer = shap.KernelExplainer(model_fn, bg_sample)
            vals = np.array(explainer.shap_values(X_ex, nsamples=100))

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        if vals.ndim == 3:              # some explainers return (n, features, classes)
            vals = vals[:, :, -1]

        mean_abs = np.abs(vals).mean(axis=0)
        feat_names = names[:mean_abs.shape[0]] if names else [f"f{i}" for i in range(mean_abs.shape[0])]

        fi_series = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        logger.info("SHAP done | explainer=%s", type(explainer).__name__)
        return {"shap_top": fi_series.head(20).to_dict(), "explainer": type(explainer).__name__}

    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return None


def explain_single_prediction(selected_pipe, X_train, x_row: pd.DataFrame, top_k: int = 5):
    """
    Returns a small, plain-English-ready list of (feature, shap_value) for ONE
    applicant/transaction row — used by decision_engine.py to build the
    "Approved/Declined because ..." explanation shown in the User Portal.
    """
    clf = _unwrap_classifier(selected_pipe.named_steps["classifier"])
    pre = selected_pipe.named_steps["preprocessor"]
    sel = selected_pipe.named_steps.get("selector", None)

    # XGBoost is explained with its own native contributions (pred_contribs):
    # exact tree-SHAP values in log-odds. This needs no `shap` package, so the lean
    # production install (requirements_api.txt) still gets explanations, and it
    # sidesteps the shap<->xgboost base_score parsing bug. Other tree models still
    # need `shap`; without it they return no drivers.
    if not SHAP_AVAILABLE and type(clf).__name__ != "XGBClassifier":
        return []

    try:
        X_pre = pre.transform(x_row)
        X_in = sel.transform(X_pre) if sel is not None else X_pre
        names = _get_feature_names(selected_pipe, x_row)

        tree_types = ("RandomForestClassifier", "ExtraTreesClassifier",
                      "GradientBoostingClassifier", "XGBClassifier",
                      "LGBMClassifier", "CatBoostClassifier")

        if type(clf).__name__ == "XGBClassifier":
            import xgboost
            contribs = clf.get_booster().predict(xgboost.DMatrix(X_in), pred_contribs=True)
            vals = contribs[:, :-1]          # last column is the bias term
        elif type(clf).__name__ in tree_types:
            try:
                explainer = shap.TreeExplainer(clf)
                vals = np.array(explainer.shap_values(X_in))
                if isinstance(vals, list):
                    vals = vals[-1]
            except ValueError as e:
                # See the matching comment in compute_shap() above — same
                # shap<->xgboost base_score parsing bug, same native fallback.
                if type(clf).__name__ == "XGBClassifier" and "could not convert string to float" in str(e):
                    booster = clf.get_booster()
                    dmat = __import__("xgboost").DMatrix(X_in)
                    contribs = booster.predict(dmat, pred_contribs=True)
                    vals = contribs[:, :-1]
                else:
                    raise
        else:
            return []   # kept fast/cheap for non-tree models at inference time

        row_vals = np.asarray(vals)[0]
        names = names[:len(row_vals)] if names else [f"f{i}" for i in range(len(row_vals))]

        pairs = sorted(zip(names, row_vals), key=lambda t: abs(t[1]), reverse=True)[:top_k]
        return [{"feature": n, "shap_value": round(float(v), 4)} for n, v in pairs]

    except Exception as e:
        logger.warning("explain_single_prediction failed: %s", e)
        return []


# ============================================================
# SAVE MODEL + REGISTRY
# ============================================================

def save_model_and_card(
    selected_name:   str,
    pipe,
    model_card:      dict,
    model_dir:       str,
    file_prefix:     str,
    version:         str = "v1",
    model_card_path: str = "",
    update_registry: bool = True,
):
    """
    Saves the trained pipeline as .joblib. `file_prefix` is e.g.
    'credit_model' or 'fraud_model' so both engines can share this one
    function while writing to their own model_dir.

    IMPORTANT — Champion-Challenger ordering: by default (`update_registry=
    True`) this immediately points `latest_model.json` at the file just
    saved. `training_pipeline.py` calls this with `update_registry=False`
    and a unique per-run `version` (a timestamp) instead, so:
      (a) the new model file never overwrites the file an *existing*
          champion of the same model type is using, and
      (b) `model_loader.run_challenger_comparison()` — called *after* this
          function — still finds the registry pointing at the true old
          champion, instead of comparing the new model against itself.
    Only `run_challenger_comparison()`'s own registry update (on PROMOTED)
    should flip the pointer in that flow. `update_registry=True` remains
    the default for any other caller (e.g. ad-hoc scripts) that wants the
    old "always publish immediately" behaviour.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{file_prefix}_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)
    logger.info("Model saved   → %s", model_path)

    if update_registry:
        active_thr = (
            model_card.get("thresholds", {}).get("active")
            or model_card.get("threshold_calibrated")
            or model_card.get("threshold_uncalibrated")
            or 0.5
        )
        registry = {
            "model_name":      os.path.basename(model_path),
            "threshold":       active_thr,
            "model_card_path": model_card_path or "",
        }
        with open(os.path.join(model_dir, "latest_model.json"), "w") as f:
            json.dump(registry, f, indent=2)
        logger.info("Registry      → %s/latest_model.json", model_dir)

    return model_path


# ============================================================
# MODEL CARD  (Google Model Cards standard — kept in this file
# since the platform folder structure has no separate model_card.py)
# ============================================================

def build_model_card(
    project:          str,
    selected_name:    str,
    train_fit_size:   int,
    cal_size:         int,
    test_size:        int,
    positive_rate_train: float,
    metrics:          dict,
    thr_uncalibrated: float,
    thr_calibrated:   Optional[float],
    cost_result:      dict,
    decision_counts:  dict,
    feature_order:    list,
    cat_indices:      list,
    selector_k:       int,
    version:          str = "v1",
    fi_dict:          Optional[dict] = None,
    shap_dict:        Optional[dict] = None,
) -> dict:
    import time as _time

    card = {
        "model_version": version,
        "model_name":    selected_name,
        "trained_at":    _time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":       project,
        "dataset": {
            "train_fit_size":     train_fit_size,
            "calibration_size":   cal_size,
            "test_size":          test_size,
            "positive_rate_train": round(float(positive_rate_train), 6),
        },
        "metrics": {
            "test_f1":        round(float(metrics.get("test_f1", 0)), 4),
            "test_precision": round(float(metrics.get("precision", 0)), 4),
            "test_recall":    round(float(metrics.get("recall", 0)), 4),
            "roc_auc":        round(float(metrics.get("roc_auc", 0)), 4),
            "ks_statistic":   round(float(metrics.get("ks", 0)), 4),
            "recall_at_5pct": round(float(metrics.get("recall_at_5", 0)), 4),
            "lift_at_5pct":   round(float(metrics.get("lift_at_5", 0)), 4),
            "brier_score":    round(float(metrics.get("brier", 0)), 4),
        },
        "thresholds": {
            "uncalibrated": round(float(thr_uncalibrated), 4),
            "calibrated":   round(float(thr_calibrated), 4) if thr_calibrated is not None else None,
            "active":       round(float(thr_calibrated if thr_calibrated is not None else thr_uncalibrated), 4),
            "tuned_on":     "held-out calibration split (not the test set)",
        },
        "decisions":       decision_counts,
        "cost_evaluation": cost_result,
        "pipeline_config": {
            "feature_order":       feature_order,
            "categorical_indices": cat_indices,
            "selector_k":          selector_k,
        },
    }

    if fi_dict is not None:
        card["feature_importances"] = fi_dict
    if shap_dict is not None:
        card["shap_top_features"] = shap_dict

    return card


def save_model_card(card: dict, model_dir: str, selected_name: str, version: str = "v1") -> str:
    os.makedirs(model_dir, exist_ok=True)
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved → %s", card_path)
    return card_path


def load_model_card(model_dir: str, selected_name: str, version: str = "v1") -> dict:
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")
    with open(card_path) as f:
        card = json.load(f)
    logger.info("Model card loaded ← %s", card_path)
    return card


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    experiment_name: str,
    run_name:        str,
    selected_name:   str,
    pipe,
    model_card:      dict,
    X_train_sample:  "pd.DataFrame" = None,
):
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed — skipping")
        return

    try:
        # Explicitly point MLflow at a project-local SQLite tracking store
        # instead of letting it resolve a default:
        #   1. Newer MLflow versions deprecate the plain filesystem ('./mlruns')
        #      backend in favor of a database backend — sqlite avoids that.
        #   2. Windows usernames containing a space (e.g. "Narendra Kalam")
        #      break MLflow's default home-directory-based file:// URI
        #      resolution ("Narendra%20Kalam" gets mis-encoded and then used
        #      as a literal path segment, causing "Access is denied" errors).
        #      A local sqlite file path sidesteps that resolution entirely.
        tracking_dir = Path(os.path.abspath("mlruns"))
        tracking_dir.mkdir(parents=True, exist_ok=True)
        db_path = (tracking_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        mlflow.set_experiment(experiment_name)
        metrics_dict = model_card.get("metrics", model_card)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", selected_name)
            mlflow.log_param("selector_k", model_card.get("pipeline_config", {}).get("selector_k", "?"))
            mlflow.log_param("threshold", model_card.get("thresholds", {}).get("active", "?"))

            mlflow.log_metric("test_f1", float(metrics_dict.get("test_f1", 0)))
            mlflow.log_metric("test_precision", float(metrics_dict.get("test_precision", 0)))
            mlflow.log_metric("test_recall", float(metrics_dict.get("test_recall", 0)))
            mlflow.log_metric("roc_auc", float(metrics_dict.get("roc_auc", 0)))
            mlflow.log_metric("ks_statistic", float(metrics_dict.get("ks_statistic", 0)))
            mlflow.log_metric("brier_score", float(metrics_dict.get("brier_score", 0)))

            try:
                log_kwargs = {
                    "sk_model": pipe,
                    "name": "model",
                    "pip_requirements": [
                        "scikit-learn", "lightgbm", "xgboost", "imbalanced-learn",
                        "pandas", "numpy",
                    ],
                    # MLflow 3.x defaults to the "skops" format, which refuses our pipeline's
                    # sklearn types as "untrusted" and silently drops the model artifact.
                    # cloudpickle stores it. (The production model is the joblib file in models/;
                    # this only affects the copy inside MLflow.)
                    "serialization_format": "cloudpickle",
                }
                if X_train_sample is not None:
                    log_kwargs["input_example"] = X_train_sample.iloc[:5].astype(float)
                mlflow.sklearn.log_model(**log_kwargs)
            except Exception as e:
                logger.warning("mlflow log_model failed: %s", e)

        logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.exception("MLflow logging failed: %s", e)