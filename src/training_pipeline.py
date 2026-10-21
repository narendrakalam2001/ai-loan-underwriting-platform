# ============================================================
# TRAINING PIPELINE — AI Loan Underwriting Platform
# ============================================================
# Orchestrates BOTH Module 1 (Credit Risk) and Module 2 (Fraud)
# training runs through one shared, generic pipeline function —
# the two engines differ only in their data loader, target column,
# model grids, decision engine and cost model, all passed in as
# parameters.
# ============================================================

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless-safe for server/CI environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                              brier_score_loss, accuracy_score, classification_report,
                              confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)

from src.config import RANDOM_STATE, MODEL_DIR, SELECT_K, CREDIT_MODEL_SUBDIR, FRAUD_MODEL_SUBDIR, DOCS_DIR
from src.preprocessing import build_preprocessors, safe_k
from src.leakage_check import detect_leakage
from src.model_tuning import (
    credit_scaled_models, credit_unscaled_models,
    fraud_scaled_models, fraud_unscaled_models,
    tune_models, train_mlp_pipeline,
)
from src.evaluation import (
    evaluate_models, select_best_model, calibrate_with_holdout,
    compute_feature_importance, compute_shap, save_model_and_card,
    mlflow_log_run, safe_predict_proba, build_model_card, save_model_card,
    f1_at_threshold_scorer,
)
from src.model_loader import run_challenger_comparison
from src.config import MIN_ROCAUC_BY_ENGINE, MIN_ROCAUC_THRESHOLD
from src.metrics import tune_threshold, psi, ks_statistic, recall_at_k, lift_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Some shap versions log very verbose internal diagnostics (num_full_subsets,
# raw phi arrays, etc.) at INFO level from KernelExplainer, which floods the
# console for every non-tree model (MLP/NeuralNet, LogisticRegression, SGD,
# GaussianNB) without indicating any actual problem. Cap it at WARNING.
logging.getLogger("shap").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================
# GENERIC TRAINING ORCHESTRATOR
# ============================================================

def _run_generic_training(
    project_label:   str,
    df:              pd.DataFrame,
    target_col:      str,
    scaled_models:   dict,
    unscaled_models: dict,
    model_dir:       str,
    file_prefix:     str,
    cost_fn,
    engine_decisions_fn,
    mlflow_experiment: str,
    cost_kwargs:     dict = None,
):
    """
    One generic run: split → leakage check → preprocess → tune →
    evaluate → select → calibrate → business rules → cost eval →
    SHAP/importances → PSI drift → model card → save → challenger.

    Returns: (selected_name, final_pipeline, model_card_dict)
    """
    cost_kwargs = cost_kwargs or {}
    os.makedirs(model_dir, exist_ok=True)
    plots_dir = os.path.join(DOCS_DIR, "plots", project_label)
    os.makedirs(plots_dir, exist_ok=True)

    from src.data_loader import detect_feature_types

    ord_cols, cont_cols, bin_cols = detect_feature_types(df, target_col=target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info("[%s] Dataset | shape=%s | positive_rate=%.4f", project_label, df.shape, y.mean())

    # ── Split: test / train_fit / calibration ────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    logger.info("[%s] Split | train_fit=%d cal=%d test=%d",
                project_label, len(X_train_fit), len(X_cal), len(X_test))

    # ── Leakage check ─────────────────────────────────────────
    leak_warnings = detect_leakage(X_train_fit, y_train_fit)
    if not leak_warnings:
        logger.info("[%s] No obvious leakage detected", project_label)

    # ── Preprocessors ─────────────────────────────────────────
    pre_scaled, pre_unscaled, cat_indices, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )
    k_safe = safe_k(SELECT_K, pre_scaled, X_train_fit)

    # ── Tune scaled + unscaled models ────────────────────────
    logger.info("[%s] Tuning scaled models ...", project_label)
    scaled_pipelines = tune_models(scaled_models, pre_scaled, cat_indices,
                                    X_train_fit, y_train_fit, use_smote=True, selector_k=k_safe)

    logger.info("[%s] Tuning unscaled models ...", project_label)
    unscaled_pipelines = tune_models(unscaled_models, pre_unscaled, cat_indices,
                                      X_train_fit, y_train_fit, use_smote=True, selector_k=k_safe)

    logger.info("[%s] Training Neural Network (MLP) ...", project_label)
    mlp_pipe = train_mlp_pipeline(X_train_fit, y_train_fit, pre_scaled, cat_indices)

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines, "NeuralNet": mlp_pipe}

    # ── Evaluate + select ─────────────────────────────────────
    summary = evaluate_models(all_pipelines, X_train_fit, y_train_fit, X_test, y_test,
                              model_dir=model_dir, X_thr=X_cal, y_thr=y_cal)
    logger.info("[%s] Model comparison:\n%s", project_label, summary.to_string())

    selected_name, selected_pipe = select_best_model(summary, all_pipelines, scaled_pipelines, unscaled_pipelines)

    y_prob_sel = safe_predict_proba(selected_pipe, X_test)
    # Threshold is chosen on the held-out calibration split, NOT the test set,
    # so test precision/recall/F1 are an honest estimate.
    thr_sel = tune_threshold(y_cal.values, safe_predict_proba(selected_pipe, X_cal))
    y_pred_sel = (y_prob_sel >= thr_sel).astype(int)

    logger.info("[%s] BEST MODEL: %s\n%s", project_label, selected_name,
                classification_report(y_test, y_pred_sel, zero_division=0))

    # ── Plots ──────────────────────────────────────────────────
    try:
        cm = confusion_matrix(y_test, y_pred_sel)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{project_label} — {selected_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        RocCurveDisplay.from_predictions(y_test, y_prob_sel, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y_test, y_prob_sel, ax=ax[1])
        plt.suptitle(f"{project_label} — {selected_name} Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_pr_curves.png"))
        plt.close()
    except Exception as e:
        logger.warning("[%s] Plot generation failed: %s", project_label, e)

    # ── Business-rule decisions ───────────────────────────────
    decisions = engine_decisions_fn(X_test, y_prob_sel, thr_sel)
    decision_counts = pd.Series(decisions).value_counts()
    logger.info("[%s] Decision distribution:\n%s", project_label, decision_counts)

    # ── Cost-sensitive evaluation ──────────────────────────────
    cost_result = cost_fn(X_test, y_test, y_pred_sel, **cost_kwargs)

    # ── Calibration ────────────────────────────────────────────
    cal_pipe = calibrate_with_holdout(selected_pipe, X_cal, y_cal)
    thr_cal = thr_sel
    y_prob_cal = None
    if cal_pipe is not None:
        y_prob_cal = cal_pipe.predict_proba(X_test)[:, 1]
        thr_cal = tune_threshold(y_cal.values, cal_pipe.predict_proba(X_cal)[:, 1])
        brier_before = brier_score_loss(y_test, y_prob_sel)
        brier_after = brier_score_loss(y_test, y_prob_cal)
        logger.info("[%s] Brier | before=%.6f after=%.6f | threshold_cal=%.4f",
                    project_label, brier_before, brier_after, thr_cal)

    # ── Repeated CV stability ───────────────────────────────────
    try:
        min_class = y_train_fit.value_counts().min()
        n_splits = min(5, max(2, int(min_class)))
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=RANDOM_STATE)
        rep_scores = cross_val_score(selected_pipe, X_train_fit, y_train_fit,
                                    scoring=f1_at_threshold_scorer(thr_sel), cv=rskf, n_jobs=1)
        logger.info("[%s] Repeated CV | mean_f1=%.4f std=%.4f", project_label, rep_scores.mean(), rep_scores.std())
    except Exception as e:
        logger.warning("[%s] Repeated CV failed: %s", project_label, e)

    # ── Feature importance + SHAP ───────────────────────────────
    fi = compute_feature_importance(selected_pipe, X_train_fit, y_train_fit)
    shap_result = compute_shap(selected_pipe, X_train_fit, X_test.head(200))

    # ── PSI drift (train vs test) ───────────────────────────────
    psi_scores = {}
    for col in cont_cols:
        if col in X_train_fit.columns and col in X_test.columns:
            psi_scores[col] = psi(X_train_fit[col].values, X_test[col].values)
    psi_df = pd.Series(psi_scores).sort_values(ascending=False)
    psi_df.reset_index().rename(columns={"index": "feature", 0: "drift_score"}).to_csv(
        os.path.join(model_dir, "feature_drift_report.csv"), index=False
    )

    # ── Monitor scores (used by the ops dashboard) ────────────
    final_probs = y_prob_cal if y_prob_cal is not None else y_prob_sel
    final_preds = (final_probs >= thr_cal).astype(int)
    final_decisions = engine_decisions_fn(X_test, final_probs, thr_cal)
    monitor_df = pd.DataFrame({
        "score": final_probs, "decision": final_decisions, "label": y_test.values,
    })
    monitor_df.to_csv(os.path.join(model_dir, "monitor_scores.csv"), index=False)

    # ── Model card ────────────────────────────────────────────
    model_card = build_model_card(
        project=project_label,
        selected_name=selected_name,
        train_fit_size=int(len(X_train_fit)),
        cal_size=int(len(X_cal)),
        test_size=int(len(X_test)),
        positive_rate_train=float(y_train_fit.mean()),
        metrics={
            "test_f1": float(f1_score(y_test, y_pred_sel, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_sel, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_sel, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob_sel)),
            "ks": float(ks_statistic(y_test, y_prob_sel)),
            "recall_at_5": float(recall_at_k(y_test, y_prob_sel, 0.05)),
            "lift_at_5": float(lift_at_k(y_test, y_prob_sel, 0.05)),
            "brier": float(brier_score_loss(y_test, y_prob_sel)),
        },
        thr_uncalibrated=float(thr_sel),
        thr_calibrated=float(thr_cal) if thr_cal != thr_sel else None,
        cost_result=cost_result,
        decision_counts=decision_counts.to_dict(),
        feature_order=feature_order,
        cat_indices=cat_indices,
        selector_k=k_safe,
        fi_dict=fi.head(20).to_dict() if fi is not None else None,
        shap_dict=shap_result.get("shap_top", {}) if shap_result is not None else None,
    )
    # Unique per-run version (NOT the old hardcoded "v1") so this run's
    # model/card files never overwrite an existing champion of the same
    # model type before run_challenger_comparison() has had a chance to
    # decide whether this challenger actually deserves to replace it.
    run_version = time.strftime("%Y%m%d_%H%M%S")
    card_path = save_model_card(model_card, model_dir, selected_name, version=run_version)

    # ── Save model (registry NOT updated here — see docstring on
    #    save_model_and_card; only a PROMOTED challenger should flip
    #    latest_model.json, and that happens inside run_challenger_comparison
    #    below) ─────────────────────────────────────────────────
    final_pipe = cal_pipe if cal_pipe is not None else selected_pipe
    model_path = save_model_and_card(selected_name, final_pipe, model_card, model_dir=model_dir,
                                      file_prefix=file_prefix, version=run_version,
                                      model_card_path=card_path, update_registry=False)

    # ── Champion-Challenger ────────────────────────────────────
    challenger_result = run_challenger_comparison(
        model_dir=model_dir,
        min_roc_auc=MIN_ROCAUC_BY_ENGINE.get(project_label, MIN_ROCAUC_THRESHOLD),
        challenger_name=selected_name,
        challenger_f1=float(f1_score(y_test, y_pred_sel, zero_division=0)),
        challenger_roc_auc=float(roc_auc_score(y_test, y_prob_sel)),
        challenger_gap=float(abs(
            accuracy_score(y_train_fit, (safe_predict_proba(selected_pipe, X_train_fit) >= thr_sel).astype(int))
            - accuracy_score(y_test, y_pred_sel)
        )),
        challenger_model_path=model_path,
        challenger_threshold=float(thr_cal),
        challenger_card_path=card_path,
    )
    logger.info("[%s] CHALLENGER RESULT: %s | %s", project_label,
                challenger_result["decision"], challenger_result["reason"])

    # ── MLflow ──────────────────────────────────────────────────
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{selected_name}"
    mlflow_log_run(mlflow_experiment, run_name, selected_name, final_pipe, model_card, X_train_sample=X_train_fit)

    return selected_name, final_pipe, model_card


# ============================================================
# MODULE 1 — CREDIT RISK TRAINING ENTRY POINT
# ============================================================

def train_credit_model(data_path: str = "data/raw/credit_risk_data.csv"):
    from src.data_loader import load_credit_dataset
    from src.credit_engine import credit_engine
    from src.metrics import cost_sensitive_evaluation_credit

    logger.info("=" * 60)
    logger.info("MODULE 1 — CREDIT RISK TRAINING")
    logger.info("=" * 60)

    df = load_credit_dataset(data_path)

    return _run_generic_training(
        project_label="credit_risk",
        df=df,
        target_col="serious_dlqin2yrs",
        scaled_models=credit_scaled_models,
        unscaled_models=credit_unscaled_models,
        model_dir=CREDIT_MODEL_SUBDIR,
        file_prefix="credit_model",
        cost_fn=cost_sensitive_evaluation_credit,
        engine_decisions_fn=lambda X, probs, thr: credit_engine(X, probs, thr),
        mlflow_experiment="credit_risk_experiments",
        cost_kwargs={"loan_amount_col": "monthly_income"},
    )


# ============================================================
# MODULE 2 — FRAUD TRAINING ENTRY POINT
# ============================================================

def train_fraud_model(data_path: str = "data/raw/fraud_transactions_data.csv"):
    from src.data_loader import load_fraud_dataset, augment_minority_for_demo
    from src.fraud_engine import fraud_engine
    from src.metrics import cost_sensitive_evaluation_fraud

    logger.info("=" * 60)
    logger.info("MODULE 2 — FRAUD DETECTION TRAINING")
    logger.info("=" * 60)

    df = load_fraud_dataset(data_path)

    # See docstring on augment_minority_for_demo() — this is a no-op on the
    # full production dataset (492 frauds); it only activates on the tiny
    # repo sample (1 fraud row) so the pipeline can be exercised end-to-end.
    df = augment_minority_for_demo(df, target_col="Class", min_minority=20)

    return _run_generic_training(
        project_label="fraud",
        df=df,
        target_col="Class",
        scaled_models=fraud_scaled_models,
        unscaled_models=fraud_unscaled_models,
        model_dir=FRAUD_MODEL_SUBDIR,
        file_prefix="fraud_model",
        cost_fn=cost_sensitive_evaluation_fraud,
        engine_decisions_fn=lambda X, probs, thr: fraud_engine(X, probs, thr),
        mlflow_experiment="fraud_detection_experiments",
        cost_kwargs={"amount_col": "Amount_original"},
    )


# ============================================================
# RUN BOTH
# ============================================================

def run_training():
    credit_result = train_credit_model()
    fraud_result = train_fraud_model()

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Credit Risk best model : {credit_result[0]}")
    print(f"  Fraud best model       : {fraud_result[0]}")
    print("=" * 60)

    return credit_result, fraud_result


if __name__ == "__main__":
    start = time.time()
    run_training()
    logger.info("Finished in %.1fs", time.time() - start)
