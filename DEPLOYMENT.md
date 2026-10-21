# Deployment guide — GitHub → Render (API) → Streamlit Community Cloud (portal + dashboard)

## What gets pushed (see `.gitignore`)
Pushed: code, tests, docs, stand-in CSVs, synthetic KB + statements, the **two champion `.joblib` files**,
`latest_model.json`, model cards, `challenger_log.json`, `monitor_scores.csv`, `feature_drift_report.csv`.
Not pushed: `.env`/keys, Kaggle CSVs, `mlruns/`, old `.joblib` files, SHAP PNGs, audit trail, built KB index, caches.
After every retrain+PROMOTED: edit the two `!models/...joblib` lines in `.gitignore` to the new champion names.

## 1. Pre-push checks
    python scripts/check_repo_size.py       # nothing >= 50 MB
    python scripts/print_pins.py            # copy xgboost / imbalanced-learn versions into requirements_api.txt
    pytest tests/ -q

## 2. GitHub
    git init && git branch -M main
    git add . && git status                 # confirm .env is NOT listed
    git commit -m "AI Loan Underwriting Platform"
    git remote add origin https://github.com/<user>/ai-loan-underwriting-platform.git
    git push -u origin main

## 3. Render — API (required)
New + → Web Service (or Blueprint, which reads `render.yaml`) → connect repo.
Build: `pip install -r requirements_api.txt && python scripts/ingest_knowledge_base.py`
Start: `uvicorn serving.api:app --host 0.0.0.0 --port $PORT` · Health check: `/health`
Env: `PYTHON_VERSION=3.11.9`, optional `GEMINI_API_KEY`. Open `<url>/docs` to test.
Free tier sleeps when idle (first call 30-60 s) and its disk is ephemeral (audit trail resets on restart).

## 4. Streamlit Community Cloud — portal and dashboard (two apps)
share.streamlit.io → Create app → repo, branch `main`.
- Portal: main file `frontend/app.py` · Dashboard: main file `monitoring/dashboard.py`
- Each folder has its own `requirements.txt` (Streamlit installs the one next to the entry file).
- Advanced settings → Secrets (portal): `UNDERWRITING_API_URL = "https://<your-api>.onrender.com"`
- The dashboard reads files from the repo only; it does not see the Render API's live audit trail,
  so "Recent applications" is empty there (model/PSI/challenger sections work).
