"""Gate 2 (ROC-AUC floor) must be per engine, or credit models can never be promoted."""
import json, os, tempfile
from src.config import MIN_ROCAUC_BY_ENGINE
from src.model_loader import run_challenger_comparison, _update_registry


def _seed_champion(d, f1, roc):
    mp = os.path.join(d, "champ.joblib"); open(mp, "wb").write(b"x")
    cp = os.path.join(d, "card.json")
    json.dump({"model_name": "Champ", "metrics": {"f1": f1, "roc_auc": roc}}, open(cp, "w"))
    _update_registry(d, mp, 0.5, cp)


def _challenge(d, f1, roc, floor):
    cm = os.path.join(d, "ch.joblib"); open(cm, "wb").write(b"x")
    return run_challenger_comparison(d, "Ch", f1, roc, 0.01, cm, 0.5, "", min_roc_auc=floor)


def test_credit_challenger_with_realistic_auc_can_be_promoted():
    d = tempfile.mkdtemp(); _seed_champion(d, 0.40, 0.85)
    r = _challenge(d, 0.43, 0.858, MIN_ROCAUC_BY_ENGINE["credit_risk"])
    assert r["decision"] == "PROMOTED", r


def test_old_global_floor_would_have_blocked_it():
    d = tempfile.mkdtemp(); _seed_champion(d, 0.40, 0.85)
    r = _challenge(d, 0.43, 0.858, 0.95)
    assert r["decision"] == "REJECTED"


def test_floor_still_rejects_a_bad_model():
    d = tempfile.mkdtemp(); _seed_champion(d, 0.40, 0.85)
    r = _challenge(d, 0.50, 0.70, MIN_ROCAUC_BY_ENGINE["credit_risk"])
    assert r["decision"] == "REJECTED"
