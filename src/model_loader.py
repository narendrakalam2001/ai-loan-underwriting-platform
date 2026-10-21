# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — AI Loan Underwriting Platform
# ============================================================
#
# CHALLENGER MODEL SYSTEM (identical logic to both reference
# projects, generalized to run for EITHER engine):
#
#   Champion  = current production model (<model_dir>/latest_model.json)
#   Challenger = newly trained model (passed in from training_pipeline)
#
#   Promotion logic — challenger promoted ONLY if it beats
#   champion on ALL 3 gates:
#     1. F1 improvement  >= MIN_F1_IMPROVEMENT
#     2. ROC-AUC         >= per-engine floor (config.MIN_ROCAUC_BY_ENGINE)
#     3. Train-test gap  <= MAX_GENERALIZATION_GAP
#
#   If challenger loses → champion stays, challenger archived.
#   Full comparison saved to <model_dir>/challenger_log.json
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP

logger = logging.getLogger(__name__)


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model(model_dir: str):
    """
    Reads <model_dir>/latest_model.json → loads .joblib + threshold.
    Returns: (model_pipeline, threshold)
    """
    registry_path = os.path.join(model_dir, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. Run training_pipeline.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(model_dir, registry["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    threshold = float(registry.get("threshold", 0.5))

    logger.info("Champion model loaded: %s | threshold=%.4f", model_path, threshold)
    return model, threshold


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics(model_dir: str) -> dict:
    registry_path = os.path.join(model_dir, "latest_model.json")
    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    card_path = registry.get("model_card_path", "")

    if not card_path:
        model_name = registry.get("model_name", "")
        # Fallback for older registries without model_card_path: try to infer
        # from the filename pattern <prefix>_<Name>_<version>.joblib
        parts = os.path.splitext(model_name)[0]
        card_path = os.path.join(model_dir, f"model_card_{parts.split('_', 1)[-1]}.json")
        logger.warning("model_card_path missing in registry — using fallback: %s", card_path)

    if not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", card)

    return {
        "model_name": card.get("model_name", "unknown"),
        "f1":         float(metrics.get("test_f1", 0)),
        "roc_auc":    float(metrics.get("roc_auc", 0)),
        "threshold":  float(card.get("thresholds", {}).get("active", 0.5)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    model_dir:             str,
    challenger_name:       str,
    challenger_f1:         float,
    challenger_roc_auc:    float,
    challenger_gap:        float,
    challenger_model_path: str,
    challenger_threshold:  float,
    challenger_card_path:  str = "",
    min_roc_auc:           float = None,
) -> dict:
    """
    Compares challenger vs current champion for the given engine
    (pass model_dir = CREDIT_MODEL_SUBDIR or FRAUD_MODEL_SUBDIR).
    Returns result dict with decision: 'PROMOTED' or 'REJECTED'.
    """
    os.makedirs(model_dir, exist_ok=True)
    challenger_log = os.path.join(model_dir, "challenger_log.json")
    roc_floor = MIN_ROCAUC_THRESHOLD if min_roc_auc is None else float(min_roc_auc)

    champion = _load_champion_metrics(model_dir)

    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(model_dir, challenger_model_path, challenger_threshold, challenger_card_path)
        result = {
            "decision":           "PROMOTED",
            "reason":             "No existing champion — first model auto-promoted",
            "challenger_name":    challenger_name,
            "challenger_f1":      round(challenger_f1, 4),
            "challenger_roc_auc": round(challenger_roc_auc, 4),
            "champion_name":      None,
            "champion_f1":        None,
            "evaluated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(challenger_log, result)
        return result

    champion_f1   = champion.get("f1", 0.0)
    champion_roc  = champion.get("roc_auc", 0.0)
    champion_name = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER  (%s)", model_dir)
    logger.info("  Champion  : %-20s F1=%.4f ROC=%.4f", champion_name, champion_f1, champion_roc)
    logger.info("  Challenger: %-20s F1=%.4f ROC=%.4f", challenger_name, challenger_f1, challenger_roc_auc)
    logger.info("=" * 55)

    gate1 = (challenger_f1 - champion_f1) >= MIN_F1_IMPROVEMENT
    gate2 = challenger_roc_auc >= roc_floor
    gate3 = challenger_gap <= MAX_GENERALIZATION_GAP
    gates_passed = gate1 and gate2 and gate3

    if gates_passed:
        decision = "PROMOTED"
        reason = (f"Challenger beats champion: "
                  f"F1 {champion_f1:.4f} → {challenger_f1:.4f} (+{challenger_f1 - champion_f1:.4f})")
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(model_dir, challenger_model_path, challenger_threshold, challenger_card_path)
    else:
        decision = "REJECTED"
        failed = []
        if not gate1:
            failed.append(f"F1 improvement {challenger_f1 - champion_f1:+.4f} < {MIN_F1_IMPROVEMENT}")
        if not gate2:
            failed.append(f"ROC-AUC {challenger_roc_auc:.4f} < {roc_floor}")
        if not gate3:
            failed.append(f"train-test gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":           decision,
        "reason":             reason,
        "evaluated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":    challenger_name,
        "challenger_f1":      round(challenger_f1, 4),
        "challenger_roc_auc": round(challenger_roc_auc, 4),
        "challenger_gap":     round(challenger_gap, 4),
        "champion_name":      champion_name,
        "champion_f1":        round(champion_f1, 4),
        "champion_roc_auc":   round(champion_roc, 4),
        "gates": {
            "f1_improvement_passed": gate1,
            "roc_auc_passed":        gate2,
            "gap_passed":            gate3,
        },
    }
    _save_challenger_log(challenger_log, result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_dir: str, model_path: str, threshold: float, model_card_path: str = None):
    registry = {
        "model_name":      os.path.basename(model_path),
        "threshold":       round(threshold, 4),
        "model_card_path": model_card_path or "",
    }
    with open(os.path.join(model_dir, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _save_challenger_log(challenger_log_path: str, result: dict):
    history = []
    if os.path.exists(challenger_log_path):
        try:
            with open(challenger_log_path) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(challenger_log_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved → %s", challenger_log_path)
