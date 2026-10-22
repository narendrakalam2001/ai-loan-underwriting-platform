# 🏦 AI Loan Underwriting Platform — Explainable Credit, Fraud & Document Decisions

[![CI](https://github.com/narendrakalam2001/ai-loan-underwriting-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/narendrakalam2001/ai-loan-underwriting-platform/actions)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Portal%20%2B%20Dashboard-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Champion-XGBoost-brightgreen.svg)](https://xgboost.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-242%20passing-brightgreen.svg)](tests/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange.svg)](https://mlflow.org)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20(optional)-8E75B2.svg)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready%20(unbuilt)-2496ED.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Domain:** BFSI / Fintech — Consumer Lending  
> **Problem:** One unified, explainable underwriting decision (`APPROVE` / `MANUAL_REVIEW` / `DECLINE`) from three independent signals — credit default risk, fraud risk and bank-statement evidence — with a policy-aware assistant on top  
> **Datasets:** [Kaggle *Give Me Some Credit*](https://www.kaggle.com/c/GiveMeSomeCredit) (150,000 borrowers, real 90-day-default label) · [Kaggle *Credit Card Fraud Detection*](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17 % fraud)  
> **Industry Context:** Banks, NBFCs and digital lenders combine bureau-style scoring, fraud screening and document verification — and must give the applicant a *reason* for every adverse decision

---

## 💡 Why This Project Matters

A lending decision is never one model. A real underwriting desk checks whether the application is *fraudulent*, whether the borrower can *repay*, and whether the *documents back up* what the applicant claimed — then has to explain the outcome in plain language. This platform wires those pieces into one pipeline:

- A **fraud gate runs first** — a fraud `BLOCK` is a final `DECLINE` and the credit model is never consulted; a fraud `REVIEW` caps the outcome at `MANUAL_REVIEW`
- A **calibrated XGBoost credit model** (isotonic calibration, threshold tuned on a held-out split) gives a real default probability, and **hard business rules run alongside it** (debt ratio, serious delinquency, income floor, dependents)
- A **document floor** — a bank-statement PDF is parsed (PyMuPDF + regex + spaCy); unreliable extraction can only make the verdict *more* cautious, never less
- A **weighted confidence score** (credit 0.55 · fraud 0.30 · document 0.15) and a **full JSON audit trail** for every decision
- **Module 5** turns the audit trail into something a human can read: RAG over policy documents, LLM-written explanations in English / हिन्दी, a decision-letter PDF and a loan chatbot — all with offline fallbacks, and with the LLM never allowed to decide or to invent numbers
- Every model promotion goes through a **3-gate champion–challenger** check, with a per-engine ROC-AUC floor

---

## 🏆 Model Results (held-out test split)

Both champions were selected by **PR-AUC** (threshold-free, the right metric for imbalanced data). Operating thresholds were tuned on a separate **calibration split — never on the test set** — so the precision / recall / F1 below are honest estimates.

### Credit risk — `XGBoost` (Give Me Some Credit)

| Metric | Score |
|---|---|
| **ROC-AUC** | `0.8579` |
| **PR-AUC** | `0.3636` |
| **KS statistic** | `0.5743` |
| **Precision / Recall / F1** (default class, threshold 0.6447) | `0.375 / 0.503 / 0.430` |
| **Brier score** (before → after isotonic calibration) | `0.0945 → 0.0510` |
| **Repeated-CV F1** (at operating threshold) | `0.4222 ± 0.0094` |
| **Calibrated operating threshold** | `0.2321` |
| **Rows after cleaning / default rate** | `148,091 / 6.74 %` |
| **Split (train_fit / calibration / test)** | `94,777 / 23,695 / 29,619` |
| **Models compared** | `13` (LightGBM crashes on the author's Windows setup and is skipped by the fail-safe) |

### Fraud screening — `XGBoost` (Credit Card Fraud Detection)

| Metric | Score |
|---|---|
| **ROC-AUC** | `0.9551` |
| **PR-AUC** | `0.7590` |
| **KS statistic** | `0.8192` |
| **Precision / Recall / F1** (fraud class, 95 frauds in test) | `0.866 / 0.747 / 0.802` |
| **Brier score** (before → after isotonic calibration) | `0.000605 → 0.000562` |
| **Repeated-CV F1** (at operating threshold) | `0.8006 ± 0.0314` |
| **Rows after de-duplication / fraud rate** | `283,726 / 0.167 %` (1,081 duplicates dropped) |
| **Split (train_fit / calibration / test)** | `181,584 / 45,396 / 56,746` |

> With only 95 frauds in the test split, F1 moves by ±0.04 if a handful of cases flip — treat the third decimal as noise.

### Decision distribution on the test split

| Credit engine | Rows | | Fraud engine | Rows |
|---|---|---|---|---|
| `APPROVE` | 20,501 | | `APPROVE` | 56,645 |
| `REVIEW_MODEL` | 4,410 | | `BLOCK_MODEL` | 82 |
| `DECLINE_MODEL` | 2,688 | | `BLOCK_RULE` | 13 |
| `DECLINE_RULE` | 1,665 | | `REVIEW` | 6 |
| `REVIEW_INCOME` | 314 | | | |
| `REVIEW_DEPENDENTS` | 41 | | | |

![Credit Risk Model Summary](docs/reports/credit_risk_model_summary.png)
![Fraud Model Summary](docs/reports/fraud_model_summary.png)

---

## 🔗 Live Links

| Service | URL |
|---|---|
| 🚀 **FastAPI (Swagger UI)** | [https://ai-loan-underwriting-platform.onrender.com/docs](https://ai-loan-underwriting-platform.onrender.com/docs) |
| 🧾 **User Portal (Streamlit)** | [https://ai-loan-underwriting-platform.streamlit.app](https://ai-loan-underwriting-platform.streamlit.app) |
| 📊 **Ops Dashboard (Streamlit)** | [https://STREAMLIT-DASHBOARD-URL.streamlit.app](https://STREAMLIT-DASHBOARD-URL.streamlit.app) |
| 📓 **EDA Notebooks** | [credit](notebooks/credit_risk_eda.ipynb) · [fraud](notebooks/fraud_eda.ipynb) · [unified analysis](notebooks/unified_risk_analysis.ipynb) |

> ⚠️ Render free tier: the API sleeps when idle — the first request may take 30–60 seconds (cold start). Open `/health` once before using the portal.  
> ⚠️ Free-tier disk is ephemeral: the audit trail resets when the API restarts, and the Streamlit dashboard shows the model/monitoring files committed to the repo (it cannot see the live API's audit trail).

---

## 🏗️ System Architecture

![System Architecture](docs/architecture/ai_loan_architecture.svg)

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║             AI LOAN UNDERWRITING PLATFORM — 6-LAYER PRODUCTION SYSTEM            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌──────────────────────────────── DATA LAYER ────────────────────────────────┐  ║
║  │  Give Me Some Credit · Credit Card Fraud · PDF bank statements             │  ║
║  │  Validate → Clean (96/98 codes · DebtRatio fix) → Leakage check            │  ║
║  │  3-way stratified split: train_fit 64% · calibration 16% · test 20%        │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                        ▼                                         ║
║  ┌────────────── TRAINING PIPELINE  (shared by credit + fraud) ───────────────┐  ║
║  │  SMOTE + SelectKBest(k=10) → RandomizedSearchCV · up to 14 families        │  ║
║  │  LR·KNN·SGD·NB·Tree·RF·ET·GB·Ada·XGB·LGBM·Cat·MLP (per-model fail-safe)    │  ║
║  │  Select by PR-AUC → threshold on calibration split → isotonic calibration  │  ║
║  │  SHAP (native XGB contribs fallback) · MLflow · JSON model card            │  ║
║  │                                                                            │  ║
║  │  CREDIT  → XGBoost  ROC-AUC 0.8579  PR-AUC 0.3636  KS 0.5743               │  ║
║  │  FRAUD   → XGBoost  ROC-AUC 0.9551  PR-AUC 0.7590  KS 0.8192               │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                        ▼                                         ║
║  ┌────────────────────────── CHAMPION – CHALLENGER ───────────────────────────┐  ║
║  │  Gate 1: F1 improvement >= 0.005                          PASS / FAIL      │  ║
║  │  Gate 2: ROC-AUC floor  credit >= 0.80 · fraud >= 0.90    PASS / FAIL      │  ║
║  │  Gate 3: train–test gap  <= 0.10                          PASS / FAIL      │  ║
║  │                                                                            │  ║
║  │  ALL gates pass → PROMOTED (latest_model.json updated)                     │  ║
║  │  ANY gate fails → REJECTED (champion retained, run logged)                 │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                        ▼                                         ║
║  ┌───────────────── SERVING LAYER — FastAPI (serving/api.py) ─────────────────┐  ║
║  │  POST /underwrite       applicant (+ document_id) → decision + audit       │  ║
║  │  POST /upload-document  bank-statement PDF → income · EMIs · stability     │  ║
║  │  GET  /application/{id} audit trail · GET /health · GET /model-info        │  ║
║  │                                                                            │  ║
║  │  Module 1 Credit ─┐                                                        │  ║
║  │  Module 2 Fraud  ─┼─▶Module 4 DECISION ENGINE ─▶audit_trail/<APP id>.json │  ║
║  │  Module 3 Docs   ─┘                                                        │  ║
║  │                                                                            │  ║
║  │  1. Fraud BLOCK → DECLINE (credit never consulted) · REVIEW → capped       │  ║
║  │  2. Document warnings / low confidence → floor at MANUAL_REVIEW            │  ║
║  │  3. Credit: hard rules first, then calibrated-probability bands            │  ║
║  │  4. Confidence = 0.55 credit + 0.30 fraud + 0.15 document quality          │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                        ▼                                         ║
║  ┌──────────────── MODULE 5 — POLICY INTELLIGENCE & ASSISTANT ────────────────┐  ║
║  │  POST /explain · GET /application/id/letter · POST /chat                   │  ║
║  │                                                                            │  ║
║  │  rag_engine      TF-IDF over 4 policy documents (86 chunks)                │  ║
║  │  llm_explainer   Gemini REST (optional)  |  EN / HI template fallback      │  ║
║  │  doc_generator   decision-letter PDF · every number from loan_math.py      │  ║
║  │  chatbot         tools: eligibility · EMI · status · FAQ | rules fallback  │  ║
║  │                                                                            │  ║
║  │  Guardrails: the LLM never decides and never invents numbers               │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                        ▼                                         ║
║  ┌──────────────────── FRONTENDS & MONITORING — Streamlit ────────────────────┐  ║
║  │  User Portal    apply · processing · decision (EN/HI + letter) · chat      │  ║
║  │  Ops Dashboard  1 Alerts · 2 Champion–Challenger · 3 KPIs                  │  ║
║  │                 4 PSI drift · 5 Recent applications (credit + fraud)       │  ║
║  │  Simulator      3 scenarios (safe · risky · random) → /underwrite          │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### Decision hierarchy (Module 4)

1. Fraud `BLOCK` → final `DECLINE` (fraud always wins) · fraud `REVIEW` → at best `MANUAL_REVIEW`
2. Document extraction with warnings or low confidence → floor at `MANUAL_REVIEW`
3. Otherwise the credit verdict drives the outcome: hard rules first (debt ratio > 100 % → decline; ≥ 3 instances of 90+ days late → decline; income < ₹5,000/month or ≥ 6 dependents → manual review), then the calibrated probability — `< 0.30` approve · `0.30–0.60` manual review · `≥ 0.60` decline (all in `src/config.py`)
4. Confidence = `0.55 × credit + 0.30 × fraud + 0.15 × document quality`; the explanation is SHAP-aware and carries improvement suggestions for any non-approval

---

## 📸 Screenshots

### 🖥️ Operations Dashboard

Five monitoring sections for each engine (credit risk · fraud): real-time alerts · champion vs challenger · performance KPIs · PSI feature drift · recent applications.

![Operations Dashboard](docs/screenshots/operations_dashboard.png)

---

### 🧾 User Portal — Apply for a Loan

Application form with personal + financial details and bank-statement upload with an extraction preview.

![Apply for Loan](docs/screenshots/user_portal_apply_for_loan.png)

---

### 🔎 User Portal — Track an Application

Look up any past application by ID — same explanation and the decision-letter download.

![Track Application](docs/screenshots/user_portal_track_application.png)

---

### 💳 Credit Risk Monitoring

| Model Performance KPIs | PSI Score |
|---|---|
| ![Credit KPIs](docs/screenshots/credit_risk_model_performance_kpis.png) | ![Credit PSI](docs/screenshots/credit_risk_psi_score.png) |

| Feature Drift Report | Recent Applications |
|---|---|
| ![Credit Drift](docs/screenshots/credit_risk_feature_drift_report.png) | ![Credit Recent](docs/screenshots/credit_risk_recent_applications.png) |

**SHAP — full explanation of one application (credit)**

![Credit SHAP](docs/screenshots/credit_risk_view_full%20_shap.png)

---

### 🚨 Fraud Monitoring

| Model Performance KPIs | PSI Score |
|---|---|
| ![Fraud KPIs](docs/screenshots/fraud_model_performance_kpis.png) | ![Fraud PSI](docs/screenshots/fraud_psi_score.png) |

| Feature Drift Report | Recent Applications |
|---|---|
| ![Fraud Drift](docs/screenshots/fraud_feature_drift_report.png) | ![Fraud Recent](docs/screenshots/fraud_recent_applications.png) |

**SHAP — full explanation of one application (fraud)**

![Fraud SHAP](docs/screenshots/fraud_view_full_shap.png)

---

## 📊 Training Reports

| Credit — Confusion Matrix | Credit — ROC & PR Curves |
|---|---|
| ![Credit Confusion](docs/plots/credit_risk/confusion_matrix.png) | ![Credit ROC PR](docs/plots/credit_risk/roc_pr_curves.png) |

| Fraud — Confusion Matrix | Fraud — ROC & PR Curves |
|---|---|
| ![Fraud Confusion](docs/plots/fraud/confusion_matrix.png) | ![Fraud ROC PR](docs/plots/fraud/roc_pr_curves.png) |

| Knowledge-Base Ingestion | Simulation Run |
|---|---|
| ![Knowledge Base](docs/reports/knowledge_base.png) | ![Simulation](docs/reports/simulation.png) |

| Test Coverage |
|---|
| ![Test Coverage1](docs/reports/test_coverage1.png) |
| ![Test Coverage2](docs/reports/test_coverage2.png) |
| ![Test Coverage3](docs/reports/test_coverage3.png) |

---

## 🎬 Demo

**End-to-end system demo**

![System Demo](docs/gifs/system_demo.gif)

**User portal walkthrough**

![User Portal](docs/gifs/User_Portal.gif)

---

## 🧠 Module 5 — Policy Intelligence & Assistant

### 5A — Policy RAG (`src/rag_engine.py`)
Structure-aware chunker (headings stay attached to their rules; FAQ Q/A pairs are atomic; long sections split on sentence boundaries) + a **scikit-learn TF-IDF** index with a light stemmer, saved as `data/knowledge_base/_index.joblib` (86 chunks from 4 documents). Build it with `python scripts/ingest_knowledge_base.py`.

**Deliberate decision — no LangChain, no ChromaDB.** Both pin `numpy<2`, which breaks this project's numpy 2-era pickles. Please don't reintroduce them.

The knowledge base in `data/knowledge_base/` is **synthetic** (files are named `*_SYNTHETIC.txt`) and aligned with the engine's real thresholds. It is **not** real RBI text. Drop the real RBI Master Circular PDF into that folder and re-run the ingest script to use it.

### 5B — Explanations (`src/llm_explainer.py`)
Gemini via plain REST (`requests`) when `GEMINI_API_KEY` is set (`GEMINI_MODEL`, default `gemini-2.5-flash`), a deterministic English/Hindi template otherwise. Safety is enforced **in code**:
- the LLM is only a *writer*: its reply must open with the correct verdict tag (`[APPROVED]` / `[UNDER REVIEW]` / `[DECLINED]`) or it is discarded and the template is used;
- only a minimal fact sheet is sent (first name, verdict, loan amount/tenure/purpose, driver *labels* with direction, the engine's explanation sentence, suggestions) — never the full name, raw income or bureau numbers.

### 5C — Decision letter (`src/doc_generator.py`)
PDF per decision: approval letter + **Loan Offer Summary** + full EMI schedule (approve), reasons + improvement tips (decline), "under review" notice (manual review). **Every number on the page comes from `src/loan_math.py`, never from the LLM.** The offer is indicative and uses an illustrative risk-based pricing curve (`10.5 % + 15 × default probability`), and the PDF says so.

### 5D — Loan assistant (`src/chatbot.py`)
Tools: `eligibility_checker` (indicative, mirrors the hard rules), `emi_calculator`, `application_status` (audit-trail lookup by ID), `faq_retriever` (RAG). Two paths behind one `chat()`: Gemini native function-calling, or an offline rules path calling the same tools. Guardrails: loan-topic gate (incl. Hindi/Hinglish keywords), 500-char input cap, 5-turn window memory, validated application IDs; the bot can never approve or decline anything, and tool failures never crash the chat.

---

## 🖥️ Portal Screens (`frontend/app.py`)

1. **Application form** — personal + financial details, bank-statement upload with extraction preview
2. **Processing** — Document → Fraud → Credit → Decision steps
3. **Decision** — banner, EMI/offer, **EN / हिन्दी explanation** with an "✨ AI-written" or "📄 template" caption, SHAP key-factors chart, suggestions, **Download decision letter**
4. **Tracker** — look up any application by ID; same explanation + letter download
5. **Floating "💬 Loan Assistant"** (bottom-right on every screen, `st.popover`) → `POST /chat`

The whole portal UI (labels, buttons, steps, decision screen, chat) switches between English and हिन्दी with the language radio; explanations, letters and chat replies come from the API in the same language, while the improvement suggestions stay in English. Needs `streamlit>=1.31`. The pinned-button CSS targets Streamlit's `data-testid` attributes (`stPopover`, `stPopoverButton`), verified on Streamlit 1.65; on other versions the chat still works but may not be pinned to the corner.

---

## 📁 Project Structure

```
ai-loan-underwriting-platform/
│
├── src/                                    # Core ML system
│   ├── config.py                           # Every threshold, band, gate, weight and scaling constant
│   ├── data_loader.py                      # Credit + fraud loaders · validation · sentinel / debt-ratio cleaning
│   ├── preprocessing.py                    # Feature engineering · scalers · encoders
│   ├── leakage_check.py                    # Leakage guard
│   ├── metrics.py                          # KS · PSI · threshold tuning · cost-sensitive evaluation
│   ├── model_tuning.py                     # RandomizedSearchCV over 14 model families · per-model fail-safe
│   ├── evaluation.py                       # Model comparison · selection · calibration · SHAP · MLflow · model card
│   ├── model_loader.py                     # Champion–challenger 3-gate system
│   ├── training_pipeline.py                # Shared training orchestrator (credit + fraud)
│   ├── credit_engine.py                    # Hard rules + calibrated-probability bands
│   ├── fraud_engine.py                     # Rule → model → review bands
│   ├── document_parser.py                  # Bank-statement PDF → income · EMIs · stability
│   ├── decision_engine.py                  # Fraud gate → credit → document floor → confidence → audit trail
│   ├── rag_engine.py                       # Module 5A — TF-IDF policy retrieval
│   ├── llm_explainer.py                    # Module 5B — Gemini / template explanations
│   ├── doc_generator.py                    # Module 5C — decision-letter PDF
│   ├── chatbot.py                          # Module 5D — loan assistant
│   ├── loan_math.py                        # EMI / schedule / offer maths (single source of numbers)
│   └── synthetic_data_gen.py               # Synthetic bank statements + stand-in data
│
├── serving/
│   └── api.py                              # FastAPI: /underwrite · /upload-document · /explain · /chat · letter …
│
├── services/
│   └── prediction_service.py               # Feature prep · ₹→model-unit income scaling · scoring bridge
│
├── frontend/
│   ├── app.py                              # Streamlit user portal
│   └── requirements.txt                    # Lean deps for Streamlit Community Cloud
│
├── monitoring/
│   ├── dashboard.py                        # Streamlit ops dashboard (5 sections × 2 engines)
│   └── requirements.txt                    # Lean deps for Streamlit Community Cloud
│
├── simulation/
│   └── data_simulator.py                   # safe · risky · random applicant traffic
│
├── scripts/
│   ├── train_model.py                      # python scripts/train_model.py --engine credit|fraud
│   ├── ingest_knowledge_base.py            # Build the RAG index
│   ├── inspect_credit_data.py              # Read-only diagnostic for the real GMSC file
│   ├── generate_synthetic_docs.py          # Synthetic bank statements
│   ├── run_api.py                          # python scripts/run_api.py
│   ├── run_dashboard.py                    # python scripts/run_dashboard.py [--portal]
│   ├── run_simulation.py                   # python scripts/run_simulation.py --scenario safe|risky|random|all --n 30
│   ├── check_repo_size.py                  # Pre-push check: files too big for GitHub
│   └── print_pins.py                       # Print library versions to pin in requirements_api.txt
│
├── notebooks/
│   ├── credit_risk_eda.ipynb               # Credit-risk exploratory analysis
│   ├── fraud_eda.ipynb                     # Fraud exploratory analysis
│   ├── unified_risk_analysis.ipynb         # Cross-module analysis
│   └── reports/
│       ├── credit_eda_decisions.json       # Decisions taken from the credit EDA
│       └── decision_engine_config.json     # Decision-engine configuration export
│
├── data/
│   ├── raw/                                # Stand-in CSVs (NOT the real datasets — see Limitations)
│   ├── synthetic_statements/               # Synthetic bank-statement PDFs + ground truth
│   ├── knowledge_base/                     # 4 synthetic policy documents (+ built TF-IDF index, git-ignored)
│   └── document_cache/                     # Parsed statements (runtime)
│
├── models/                                 # Champion artifacts — registry/cards/CSVs pushed to GitHub
│   ├── credit_risk/
│   │   ├── latest_model.json               # Champion model registry
│   │   ├── challenger_log.json             # Full champion–challenger comparison history
│   │   ├── model_card_XGBoost_*.json       # Model card (metrics · thresholds · SHAP top features)
│   │   ├── monitor_scores.csv              # Score/decision monitor data for the dashboard
│   │   ├── feature_drift_report.csv        # PSI drift per feature
│   │   └── credit_model_XGBoost_*.joblib   # Champion model (only the champion is pushed)
│   └── fraud/                              # Same layout for the fraud engine
│
├── mlruns/                                 # MLflow experiment tracking (git-ignored)
├── logs/                                   # audit_trail/<APP id>.json (runtime, git-ignored)
│
├── docs/
│   ├── architecture/
│   │   └── ai_loan_architecture.svg        # 6-layer system architecture diagram
│   ├── plots/
│   │   ├── credit_risk/                    # confusion_matrix.png · roc_pr_curves.png
│   │   └── fraud/                          # confusion_matrix.png · roc_pr_curves.png
│   ├── reports/
│   │   ├── credit_risk_model_summary.png   # Credit training & evaluation summary
│   │   ├── fraud_model_summary.png         # Fraud training & evaluation summary
│   │   ├── knowledge_base.png              # RAG ingestion terminal output
│   │   ├── simulation.png                  # Simulation run terminal output
│   │   └── test_coverage1.png … 3.png      # pytest 242/242 + coverage report
│   ├── screenshots/                        # Ops dashboard + portal screens (credit + fraud)
│   └── gifs/
│       ├── system_demo.gif                 # End-to-end system demo
│       └── User_Portal.gif                 # User portal walkthrough
│
├── tests/                                  # 242 pytest tests across 14 files (see Test Coverage)
├── .github/workflows/ci.yml                # ingest → pytest → API smoke test (underwrite → explain → letter → chat)
├── .streamlit/config.toml                  # Forces the light theme (portal CSS is light-only)
├── Dockerfile                              # FastAPI production image
├── Dockerfile.dashboard                    # Streamlit dashboard container
├── docker-compose.yml                      # API + dashboard + portal (ports 8000 · 8501 · 8502)
├── render.yaml                             # Render.com deployment config
├── DEPLOYMENT.md                           # GitHub → Render → Streamlit Cloud steps
├── .env.example                            # Environment variables template (real .env is git-ignored)
├── .gitignore
├── .dockerignore
├── requirements.txt                        # Full dependencies (training + CI)
├── requirements_api.txt                    # Lean API deps for Render / Docker (versions pinned)
├── requirements_dashboard.txt              # Lean dashboard / portal deps
├── LICENSE                                 # MIT License
└── README.md                               # This file
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/narendrakalam2001/ai-loan-underwriting-platform.git
cd ai-loan-underwriting-platform
pip install -r requirements.txt
```

> Keep `numpy` / `scikit-learn` at the versions in your environment — upgrading them can break saved models.

### 2. Get the datasets

Download **Give Me Some Credit** (`cs-training.csv`) and **Credit Card Fraud Detection** (`creditcard.csv`) from Kaggle and keep them anywhere on disk — you pass their paths to the trainer. The CSVs in `data/raw/` are tiny stand-ins so the tests can run; **do not report metrics from them.**

### 3. Train both engines

```bash
python scripts/train_model.py --engine credit --credit-path "path/to/cs-training.csv"
python scripts/train_model.py --engine fraud  --fraud-path  "path/to/creditcard.csv"
```

Expected output (credit run, abridged from a real log):
```
INFO  Credit data validation passed | shape=(148091, 14) | default_rate=0.0674
INFO  [credit_risk] Split | train_fit=94777 cal=23695 test=29619
INFO  Tuning: XGBoost  ...
INFO  Selected model: XGBoost
INFO  [credit_risk] Brier | before=0.094456 after=0.050985 | threshold_cal=0.2321
INFO  [credit_risk] Repeated CV | mean_f1=0.4222 std=0.0094
INFO  Model saved   → models\credit_risk\credit_model_XGBoost_20261015_164445.joblib
INFO  [credit_risk] CHALLENGER RESULT: PROMOTED | No existing champion — first model auto-promoted
```

Each run takes hours on a laptop (RandomizedSearchCV over 14 model families). If LightGBM crashes natively on your machine, skip it — and any other model — without code changes:

```powershell
$env:EXCLUDE_MODELS="LightGBM,KNN"      # Windows PowerShell
export EXCLUDE_MODELS=LightGBM,KNN       # macOS / Linux
```

If you change the credit dataset or schema, delete `models/credit_risk/*` before retraining — a model trained on a different schema is incompatible.

### 4. Build the knowledge base

```bash
python scripts/ingest_knowledge_base.py
```

### 5. (Optional) Enable Gemini

Create a key at [Google AI Studio](https://aistudio.google.com/apikey) and set it **in the terminal that runs the API**:

```powershell
$env:GEMINI_API_KEY="your-key"           # Windows PowerShell
export GEMINI_API_KEY=your-key           # macOS / Linux
# optional: GEMINI_MODEL=gemini-2.5-flash · HINDI_FONT_PATH=/path/to/devanagari.ttf
```

Without a key everything still works through the offline fallbacks.

### 6. Run

```bash
python scripts/run_api.py                        # API        → http://localhost:8000/docs
python scripts/run_dashboard.py --portal         # Portal     → http://localhost:8502
python scripts/run_dashboard.py                  # Dashboard  → http://localhost:8501
python scripts/run_simulation.py --scenario all --n 30      # traffic for the dashboard
```

Use one terminal per process; the API keeps running while the others start.

### 7. Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# 242 collected · 242 passed
```

---

## 🐳 Docker

```bash
docker compose up --build                       # API :8000 · Ops dashboard :8501 · Portal :8502
GEMINI_API_KEY=... docker compose up --build    # optional LLM features
docker compose down
```

```bash
docker compose up api          # API only
docker compose up dashboard    # Ops dashboard only
docker compose up portal       # User portal only
```

| Service | URL |
|---|---|
| FastAPI + Swagger | `http://localhost:8000/docs` |
| Ops Dashboard | `http://localhost:8501` |
| User Portal | `http://localhost:8502` |

The API image copies `data/knowledge_base/` and builds the index during the build. Compose mounts only `data/document_cache` (mounting all of `./data` would hide the built index). `render.yaml` runs the ingest script in the API's build command; set `GEMINI_API_KEY` in the Render dashboard (declared `sync: false`). **The Docker and Render files were not built or run during development — build them yourself before relying on them.**

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health`, `/model-info` | Liveness / model registry |
| `POST` | `/upload-document` | Bank-statement PDF → extraction + `document_id` |
| `POST` | `/underwrite` | Applicant (+ optional `document_id`) → decision; stores an `application` summary in the audit trail |
| `GET` | `/application/{id}` | Full audit trail of a past decision |
| `POST` | `/explain` | `{application_id, language: en\|hi}` → `{text, source: gemini\|template, language}` |
| `GET` | `/application/{id}/letter?language=en\|hi` | `application/pdf` (headers `X-Language-Used`, `X-Text-Source`) |
| `POST` | `/chat` | `{session_id, message, language}` → `{reply, source, tool_calls, blocked, language}` |

IDs are validated (`[A-Za-z0-9_-]{1,64}`; malformed → 400, unknown → 404). Chat sessions live in a bounded in-memory dict (max 500, least-recently-used evicted) — per-process, lost on restart. The audit trail stores only `full_name`, `loan_amount_requested`, `tenure_months`, `loan_purpose` from the request, which is exactly what the letter's offer page needs.

### POST /underwrite

```bash
curl -X POST "http://localhost:8000/underwrite" -H "Content-Type: application/json" -d '{
  "full_name": "Asha Verma", "age": 35, "monthly_income": 60000, "dependents": 1,
  "existing_emis": 5000, "credit_utilization_pct": 25, "open_credit_lines": 3,
  "real_estate_loans": 0, "times_30_59_days_late": 0, "times_60_89_days_late": 0,
  "times_90_days_late": 0, "loan_amount_requested": 500000, "tenure_months": 36,
  "loan_purpose": "Personal" }'
```

`monthly_income` is an actual ₹ amount per month. The response contains `final_verdict`, `confidence_score`, `explanation`, `improvement_suggestions`, `module_outputs`, `shap_top_drivers`, `application` and `decided_at`.

### GET /health · GET /model-info

```json
{"status": "running", "credit_model_loaded": true, "fraud_model_loaded": true}
```

`/model-info` returns the `latest_model.json` registry entry of each engine (champion file name, threshold, model-card path).

### Example response of POST /underwrite

Captured from the real API code path with stub models (so the probabilities are illustrative; with the real champions `shap_top_drivers` carries the top SHAP features):

```json
{
  "application_id": "APP03676DD79D",
  "final_verdict": "APPROVE",
  "confidence_score": 0.8645,
  "explanation": "Approved because: strong income stability + low existing obligations + clean fraud check",
  "improvement_suggestions": [],
  "module_outputs": {
    "credit_risk":     {"default_probability": 0.05, "risk_band": "LOW", "verdict": "APPROVE", "rule_triggered": null},
    "fraud_detection": {"fraud_probability": 0.01, "fraud_band": "LOW", "verdict": "APPROVE", "rule_triggered": null},
    "document_parser": null
  },
  "shap_top_drivers": [],
  "decided_at": "2026-10-22T02:04:29.627020Z",
  "application": {"full_name": "Asha Verma", "loan_amount_requested": 500000.0, "tenure_months": 36, "loan_purpose": "Personal"},
  "latency_seconds": 0.045
}
```

---

## 📊 Credit Models Compared (test split, tuned operating thresholds)

| Rank | Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | KS |
|---|---|---|---|---|---|---|---|
| 1 ⭐ | **XGBoost** | 0.375 | 0.503 | **0.430** | 0.858 | **0.364** | 0.574 |
| 2 | CatBoost | 0.365 | 0.504 | 0.424 | 0.850 | 0.348 | 0.554 |
| 3 | LogisticRegression | 0.365 | 0.443 | 0.400 | 0.848 | 0.331 | 0.551 |
| 4 | GaussianNB | 0.327 | 0.512 | 0.399 | 0.843 | 0.310 | 0.538 |
| 5 | SGD | 0.365 | 0.438 | 0.398 | 0.848 | 0.331 | 0.555 |
| 6 | BernoulliNB | 0.358 | 0.444 | 0.396 | 0.779 | 0.297 | 0.519 |
| 7 | GradientBoosting | 0.304 | 0.515 | 0.382 | 0.841 | 0.284 | 0.543 |
| 8 | RandomForest | 0.294 | 0.545 | 0.382 | 0.841 | 0.280 | 0.542 |
| 9 | AdaBoost | 0.295 | 0.540 | 0.381 | 0.839 | 0.288 | 0.543 |
| 10 | DecisionTree | 0.305 | 0.491 | 0.376 | 0.819 | 0.251 | 0.522 |
| 11 | ExtraTrees | 0.299 | 0.506 | 0.376 | 0.838 | 0.279 | 0.545 |
| 12 | NeuralNet (MLP) | 0.295 | 0.426 | 0.349 | 0.773 | 0.266 | 0.425 |
| 13 | KNN | 0.230 | 0.401 | 0.293 | 0.769 | 0.200 | 0.442 |

> LightGBM was excluded in this run (native crash on the author's Windows machine — handled by a per-model fail-safe, not a code bug). ROC-AUC ≈ 0.86 is in line with published results on this dataset.

## 📊 Fraud Models Compared

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | KS |
|---|---|---|---|---|---|---|
| **XGBoost** ⭐ | 0.866 | 0.747 | 0.802 | 0.955 | **0.759** | 0.819 |
| GaussianNB | 0.896 | 0.726 | 0.802 | 0.955 | 0.652 | 0.861 |
| NeuralNet (MLP) | 0.875 | 0.737 | 0.800 | 0.906 | 0.723 | 0.807 |
| ExtraTrees | 0.885 | 0.726 | 0.798 | 0.931 | 0.692 | 0.832 |
| SGD | 0.883 | 0.716 | 0.791 | 0.959 | 0.670 | 0.846 |
| LogisticRegression | 0.870 | 0.705 | 0.779 | 0.969 | 0.690 | 0.848 |
| RandomForest | 0.878 | 0.684 | 0.769 | 0.957 | 0.703 | 0.848 |

> Five models are within ±0.03 F1 of each other — on 95 test frauds that is indistinguishable. XGBoost is chosen on PR-AUC, which is threshold-free and uses every score, not just the thresholded cases.

---

## 🏆 Champion vs Challenger — 3-Gate Promotion

Every new training run is compared against the production champion for **its own engine** (`models/credit_risk/` or `models/fraud/`):

| Gate | Condition | Rationale |
|---|---|---|
| F1 improvement | Challenger beats champion by ≥ `0.005` | Promote meaningful gains only |
| ROC-AUC floor | **Per engine:** credit `≥ 0.80` · fraud `≥ 0.90` | Minimum viable discrimination for *that* problem |
| Generalisation gap | Train–test gap ≤ `0.10` | No overfitting (both at the same operating threshold) |

All gates pass → `PROMOTED` (`latest_model.json` updated). Any gate fails → `REJECTED` (champion retained). Every run is appended to `models/<engine>/challenger_log.json` and shown in the dashboard. A fresh registry auto-promotes its first model.

> The ROC-AUC floor is deliberately **not** one global number: a single 0.95 floor suits fraud (~0.955) but would reject every credit challenger forever, since Give-Me-Some-Credit models top out near 0.86.

---

**Real run — first model auto-promotion (credit):**

```
[credit_risk] CHALLENGER RESULT: PROMOTED | No existing champion — first model auto-promoted
  Champion  : credit_model_XGBoost_20261015_164445.joblib
  Threshold : 0.2321 (tuned on the calibration split)
  ROC-AUC   : 0.8579   PR-AUC : 0.3636   F1 : 0.4298
[fraud] CHALLENGER RESULT: PROMOTED | No existing champion — first model auto-promoted
  Champion  : fraud_model_XGBoost_20261015_121938.joblib
  ROC-AUC   : 0.9551   PR-AUC : 0.7590   F1 : 0.8023
```

> Every fresh registry auto-promotes its first model — later retrains must clear all 3 gates to replace the champion. Results are logged to `models/<engine>/challenger_log.json` and shown in dashboard Section 2 with per-gate ✅/❌ status.

---

## 🧱 Training Pipeline Highlights

| Step | Implementation |
|---|---|
| **Cleaning (credit)** | `96`/`98` placeholder codes in the three days-past-due columns → `0` + flag `late_count_was_sentinel`; `DebtRatio` that is really an absolute amount (missing income) → median ratio of rows with income |
| **Split** | 3-way stratified: train_fit · calibration · test |
| **Imbalance** | SMOTE-style resampling inside an `imblearn` pipeline (never leaks into validation folds) + `SelectKBest` (k = 10) |
| **Search** | `RandomizedSearchCV` over up to 14 model families; a model that crashes natively is skipped and logged, never kills the run (`EXCLUDE_MODELS` for manual control) |
| **Selection** | Filter on generalisation gap / CV std / minimum F1, then rank by **PR-AUC** |
| **Threshold** | F1-optimal threshold tuned on the **calibration split**, not the test set |
| **Calibration** | Isotonic regression on the calibration split (sigmoid if < 200 rows) |
| **CV metric** | CV F1 is scored at each model's own operating threshold so it is comparable with test F1 |
| **Explainability** | SHAP — `TreeExplainer`, native XGBoost `pred_contribs` fallback, permutation fallback for other models |
| **Tracking** | MLflow (params · metrics · model) + JSON model card per run with thresholds, metrics and SHAP top features |
| **Governance** | 3-gate champion–challenger with per-engine ROC-AUC floor |

---

## 💱 Income Scaling — An Explicit Assumption

GMSC `MonthlyIncome` is dollar-scale (median ≈ 5,400); the portal collects ₹. Live income is converted before it reaches the model:

```
model_income = ₹ income / INCOME_INR_PER_MODEL_UNIT      # src/config.py, default 5.0
```

The default maps a typical Indian monthly income (~₹27,000) onto GMSC's median ("income-rank matching"). **It is an assumption, not a measurement.** Alternatives: ~20–22 (PPP rate) or ~83–88 (market FX). Override without retraining: `set INCOME_INR_PER_MODEL_UNIT=21`. `debt_ratio` is computed from the real ₹ values first (it is unit-free), and the ₹5,000 income floor is applied in ₹ at serving and as `MIN_INCOME_RULE_MODEL_UNITS` inside the engine.

---

## 🏦 Underwriting Rules & Decision Bands

Business rules are evaluated **alongside** the ML probability — the model never replaces them. All values live in `src/config.py`.

### Credit engine

| Decision | Trigger | Meaning |
|---|---|---|
| `DECLINE_RULE` | Debt ratio > 100 % **or** ≥ 3 instances of 90+ days late | Hard rule — decline regardless of model score |
| `REVIEW_INCOME` | Monthly income < ₹5,000 | Cannot be assessed automatically — human review |
| `REVIEW_DEPENDENTS` | ≥ 6 dependents | Affordability needs a human check |
| `APPROVE` | Calibrated default probability `< 0.30` | Low risk |
| `REVIEW_MODEL` | `0.30 ≤ p < 0.60` | Medium risk — manual review |
| `DECLINE_MODEL` | `p ≥ 0.60` | High risk |

> The bands are fixed policy numbers, so the approve / review / decline mix does not shift when the model's F1 threshold (0.2321) is retuned.

### Fraud engine

| Decision | Trigger | Effect on the final decision |
|---|---|---|
| `BLOCK` (`HIGH_AMOUNT_RULE`) | Requested amount > ₹50 lakh (card-scale rule of 5,000 is used only in batch evaluation) | Final `DECLINE` — credit model never consulted |
| `REVIEW` (`CARD_TESTING_PATTERN`) | Requested amount < ₹5,000 | Capped at `MANUAL_REVIEW` |
| `BLOCK` (model) | Fraud probability ≥ threshold (0.50) | Final `DECLINE` |
| `REVIEW` | Probability ≥ 0.6 × threshold | Capped at `MANUAL_REVIEW` |
| `APPROVE` | Otherwise | No effect |

### Final decision priority

`DECLINE` > `MANUAL_REVIEW` > `APPROVE` — the worst verdict from fraud, credit and the document floor wins. Confidence = `0.55 × credit + 0.30 × fraud + 0.15 × document quality`.

---

## 📈 Monitoring Dashboard — 5 Sections

Each section is shown for **both engines** (credit risk · fraud):

| Section | What it shows |
|---|---|
| **1. Real-Time Alerts** | High average score · decline/block rate > 25 % · review queue > 20 % · PSI moderate / critical drift |
| **2. Champion vs Challenger** | Latest decision badge · 3-gate pass/fail status · full history table |
| **3. Model Performance KPIs** | Score distribution · decision distribution · score statistics |
| **4. Feature Drift (PSI)** | Per-feature PSI with stable / moderate / critical colour coding |
| **5. Recent Applications** | Last 50 decisions from the audit trail · pick an application to view its full explanation (SHAP drivers) |

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/credit_risk_eda.ipynb` | Credit-risk exploratory data analysis |
| `notebooks/fraud_eda.ipynb` | Fraud-detection exploratory data analysis |
| `notebooks/unified_risk_analysis.ipynb` | Cross-module analysis feeding the decision engine |
| `notebooks/reports/credit_eda_decisions.json` | Cleaning / modelling decisions recorded from the credit EDA |
| `notebooks/reports/decision_engine_config.json` | Decision-engine configuration export |

---

## 🧪 Test Coverage

```
242 tests across 14 files — all offline (Gemini HTTP mocked; API tests use a stub model)

  test_module5.py                124  chunker · retrieval · loan math · explainer · letter PDF ·
                                      audit trail · chatbot (rules + Gemini loop) · API endpoints
  test_engines.py                 19  credit + fraud engine rules and bands
  test_leakage_and_metrics.py     17  leakage guard · KS · PSI · threshold tuning · lift
  test_income_scaling.py          15  ₹ → model-unit conversion · income floor consistency
  test_decision_engine.py         14  fraud gate · document floor · confidence · audit trail
  test_preprocessing.py           12  feature engineering
  test_credit_data_cleaning.py    10  sentinel codes · debt-ratio fix · de-duplication
  test_config.py                   9  constants sanity
  test_document_parser.py          7  statement parsing
  test_model_tuning_safety.py      5  crashing model skipped · EXCLUDE_MODELS · probe
  test_champion_gates.py           3  per-engine ROC-AUC floor
  test_threshold_selection.py      3  threshold tuned on validation, CV F1 at operating threshold
  test_selection_and_importance.py 2  PR-AUC ranking · permutation importance names
  test_explain_without_shap.py     2  XGBoost explanations work without the `shap` package (lean prod image)
```

`scripts/ingest_knowledge_base.py` is not required to run the tests locally, but CI runs it first, then `pytest`, then an API smoke test that walks underwrite → explain → letter → chat.

Measured line coverage of `src/` (`pytest --cov=src`, 240-test run): **72 %** overall — `llm_explainer` 100 %, `chatbot` 94 %, `decision_engine` 92 %, `rag_engine` / `doc_generator` / `preprocessing` ≈ 88 %. The training orchestrator (`training_pipeline.py` 0 %) and most of `evaluation.py` (35 %) are **not** exercised by pytest because a real training run takes hours; they were run end-to-end on the real data and on tiny synthetic datasets instead.

![Test Coverage 1](docs/reports/test_coverage1.png)
![Test Coverage 2](docs/reports/test_coverage2.png)
![Test Coverage 3](docs/reports/test_coverage3.png)

---

## 🧠 Technical Standards

| Component | Implementation |
|---|---|
| **Credit model** | XGBoost champion · isotonic calibration · SHAP explanations · KS / PR-AUC reporting |
| **Fraud model** | XGBoost champion · isotonic calibration · rule layer (`BLOCK_RULE`) in front of the model |
| **Business rules** | Evaluated alongside the ML probability, never replacing it; every threshold lives in `src/config.py` |
| **Leakage control** | Leakage check · calibration/test splits never used for fitting · threshold never tuned on test |
| **Explainability** | SHAP drivers in the audit trail · plain-language explanation · improvement suggestions |
| **LLM safety** | LLM writes text only · verdict-tag validation · minimal fact sheet · numbers only from `loan_math` |
| **RAG** | scikit-learn TF-IDF (no LangChain / ChromaDB) · structure-aware chunking |
| **Audit** | One JSON per decision under `logs/audit_trail/` |
| **Monitoring** | PSI drift per feature · score alerts · champion–challenger history · recent applications |
| **CI/CD** | GitHub Actions — ingest → pytest → API smoke test |
| **Deployment** | Render.com (FastAPI) + Streamlit Community Cloud (portal + dashboard) · Dockerfiles + `docker-compose.yml` (not built in development) |

---

## 📊 Business Impact

Computed from the **held-out test split** with the production decision bands (credit engine, `n = 29,619`; fraud engine, `n = 56,746`):

| Metric | Value |
|---|---|
| Applications auto-approved (credit engine) | **69.2 %** (20,501) |
| Applications auto-declined (model + hard rules) | **14.7 %** (4,353) |
| Applications routed to manual review | **16.1 %** (4,765) |
| **Straight-through decisions (no human needed)** | **83.9 %** |
| Fraud caught at the operating threshold | **71 of 95** (recall 0.747 · precision 0.866 · 11 false alarms) |
| Fraud engine flags on the test split | 95 blocks (82 model + 13 rule) + 6 reviews out of 56,746 (0.17 % blocked) |
| Review queue vs dashboard alert limit | 16.1 % vs 20 % limit — inside tolerance |
| Decline/block rate vs dashboard alert limit | 14.7 % vs 25 % limit — inside tolerance |

**Illustrative staffing impact** — *assumption, change it for your desk*: if one manual underwriting review takes **15 minutes**, then per 1,000 applications 839 straight-through decisions free about **210 reviewer-hours** (`839 × 15 min`), while the remaining 161 stay with humans.

**Model cost evaluation (test split, model units):**

| Engine | Missed positives (FN) | False alarms (FP) | Missed-loss | Review cost | Total |
|---|---|---|---|---|---|
| Credit | 992 | 1,669 | 3,548,350.20 | 83,450.00 | 3,631,800.20 |
| Fraud | 24 | 11 | 4,850.29 | 165.00 | 5,015.29 |

> These totals are in the model's own units (GMSC dollar-scale income, card-scale amounts), not real rupees — use them to **compare models**, not to quote a business saving. Credit loss = `monthly_income × 0.6` per missed default, review cost = 50 per false alarm; fraud review cost = 15 per false alarm.

---

## ⚠️ Known Limitations & Honest Caveats

1. **`data/raw/*.csv` are stand-ins** (the credit one is a 900-row synthetic GMSC-shaped file) so the code and tests can run. Real metrics exist only after training on the full Kaggle files. Never quote numbers from a stand-in run.
2. **Real-GMSC quirks are handled, but only by retraining.** 269 rows (0.18 %) carry placeholder codes 96/98 (replaced by 0 and flagged; those rows default at ~55 %). Where income is missing (19.8 %) `DebtRatio` is an absolute amount, not a ratio, and had made the `debt_ratio > 1` rule auto-decline ~23 % of the file. A credit model trained before this fix must be retrained.
3. **Income units are an assumption** (see *Income Scaling* above). The default factor 5.0 is a modelling choice, not a measurement.
4. **The credit cost figure is illustrative.** The credit "estimated loss" uses `monthly_income × 0.6` as an exposure proxy in GMSC dollar units, not a real rupee loan balance. Use it to compare models, not to quote a business impact.
5. **The fraud model is a structural demo.** It is trained on the anonymised card-fraud dataset (V1–V28); for a loan application those components are filled deterministically, so only the loan-amount rule is a genuinely meaningful signal. A real application-fraud model needs device, velocity, IP/geo signals.
6. **Small test counts.** 95 test frauds mean fraud precision/recall/F1 carry wide uncertainty; the calibrated fraud threshold (0.5000) comes from ~76 frauds in the calibration split and sits on a coarse isotonic step.
7. **Thresholds are tuned on one calibration split** that also fits the isotonic calibrator — slightly optimistic compared with a third, separate split, but far less so than tuning on the test set.
8. **Model selection still reads test-split metrics.** A separate validation split for selection would be stricter.
9. **LightGBM is skipped** on the author's Windows setup (native access violation inside the pipeline); XGBoost and CatBoost cover gradient boosting.
10. **Gemini output was validated with mocked HTTP responses.** Request/response shapes follow Google's public REST API; verify the live call with your own key. Model names change — override with `GEMINI_MODEL`.
11. **Hindi PDF rendering needs a Devanagari TTF** (`HINDI_FONT_PATH`, Windows Nirmala/Mangal, or Linux Noto/Lohit). reportlab does no complex-script shaping, so conjuncts and matras may look wrong; without a font the letter falls back to English (`X-Language-Used: en`). On-screen Hindi explanations and chat replies are the reliable path, and the Hindi text is template-written and not reviewed by a native speaker.
12. **TF-IDF retrieval is lexical.** "3 serious delinquencies" can rank the right policy second; the chatbot shows the runner-up when scores are close. Upgrade to embeddings if the knowledge base grows.
13. **Knowledge base and loan offer are synthetic / illustrative** — not real RBI text, not a real rate card.
14. **The eligibility checker is indicative only** — it mirrors the hard rules and cannot run the ML model.
15. **Privacy detail:** the fact sheet sent to Gemini excludes name, income and bureau numbers, but the engine's explanation sentence may contain an existing-EMI amount taken from an uploaded statement.
16. **Guardrails reduce, not eliminate, risk.** Topic gating and prompt rules cannot fully prevent prompt injection or off-topic drift.
17. **Chat sessions are in-memory and per-process** (single worker only; lost on restart).
18. **Docker files were never built or run.** The Render API and the Streamlit portal were deployed by hand and checked manually, not load-tested; the portal's pinned chat button was verified on one Streamlit version only.
19. **Free-tier hosting is not durable.** The Render API sleeps when idle (30–60 s cold start) and its disk is ephemeral, so audit trails and chat sessions reset on restart; the Streamlit dashboard reads only the files committed to the repo and cannot show live applications.
20. **Model pickles are version-bound.** `requirements_api.txt` pins `numpy`, `pandas`, `scikit-learn`, `scipy`, `xgboost`, `imbalanced-learn` and `joblib` to the training environment; upgrading any of them can break the champion files.

---

## 🛡️ Ethical Considerations

- Automated credit decisions must have human oversight — `MANUAL_REVIEW` exists for exactly this reason, and the chatbot can never approve or decline anything
- GMSC is a US-origin dataset; applying it to Indian borrowers (even with explicit income scaling) is a **modelling assumption**, not a validated local credit model
- Fairness is not audited: age is a model input, and performance may vary across income, age and region. A production system needs a fairness review and adverse-action compliance review
- Applicant data (name, income, bank statements) is personal data — a real deployment needs consent, minimisation and retention rules (e.g. under India's DPDP Act) and must not send raw identifiers to a third-party LLM
- Synthetic policy documents are placeholders; use official regulator text before giving applicants any policy guidance

---

## 👨‍💻 About

**Narendra Kalam** — MSc Computer Science (Gold Medalist — NASSCOM, Full Stack Data Science + AI)

> Building 20+ industry-level, end-to-end ML systems across all domains.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/narendra-kalam/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?logo=kaggle)](https://www.kaggle.com/narendrakalam)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green?logo=github)](https://narendrakalam2001.github.io/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:kalamnarendra2001@gmail.com)

### Portfolio Projects

| # | Project | Domain | Champion Model | Key Metric |
|---|---|---|---|---|
| 1 | Credit Card Fraud Detection | BFSI / Fintech | ExtraTrees | F1 = 0.8962 · 284K transactions |
| 2 | Credit Risk Prediction | BFSI / Lending | LightGBM | F1 = 0.9741 · ROC-AUC = 0.9991 |
| 3 | Customer Churn Prediction | Telecom / BFSI | CatBoost | F1 = 0.634 · Recall = 0.7312 |
| 4 | House Price Prediction | Real Estate | CatBoost | RMSE = $20,128 · R² = 0.9053 |
| 5 | Store Sales Forecasting | Retail / Supply Chain | LightGBM (Ensemble) | RMSLE = 0.3739 · R² = 0.9761 |
| 6 | Energy Demand Forecasting | Energy / Utilities | ElasticNet | RMSE = 712.04 MW · R² = 0.9759 |
| 7 | Stock Price & Risk Forecasting | Fintech / Capital Markets | Ridge | DirAcc = 53.44% · Sharpe = 0.80 |
| 8 | Resume Screener AI | HR Tech | LightGBM | F1 = 0.7608 · Top-3 = 0.9416 |
| 9 | ABSA Sentiment Analysis | E-Commerce / Banking | RidgeClassifier | Macro-F1 = 0.6212 · ROC-AUC = 0.823 |
| 10 | Fake News Detector | Media Tech / Gov Tech | XGBoost | F1 = 0.9993 · ROC = 1.0000 |
| 11 | BC5CDR Clinical NER | Biomedical NLP | BioBERT | F1 = 0.8847 · Chemical F1 = 0.9239 |
| 12 | News Topic Modeling | Media Analytics | LDA (Gensim) | Cv = 0.6225 · Diversity = 0.92 |
| 13 | Chest X-Ray Diagnosis | Healthcare AI | DenseNet121 | Mean AUC = 0.7864 · 14 classes |
| 14 | Real-Time Object Detection | Computer Vision / Retail-Security | YOLOv8s | mAP50-95 = 0.5341 · 32 FPS |
| 15 | Face Emotion Recognition | EdTech / Retail CX | CNN-from-scratch | Macro-F1 = 0.5950 · 7 classes |
| 16 | Customer Segmentation Engine | E-Commerce / BFSI | DBSCAN (Unsupervised) | Silhouette = 0.4056 |
| 17 | Market Basket Analysis (Instacart) | Retail / Quick-Commerce | Apriori | 68,820 rules · mean lift = 15.66 |
| 18 | E-Commerce / OTT Recommender | E-Commerce / Streaming | Hybrid (SVD + Content) | NDCG@10 = 0.0407 · 4 candidates |
| 19 | Hospital Readmission Prediction | Healthcare / Hospital Ops | ExtraTrees | F1 = 0.2702 · ROC-AUC = 0.6513 |
| 20 | HR Policy Intelligence Chatbot | HR Tech / Enterprise GenAI | Gemini 3.6 Flash + RAG | 30/30 tests · guardrail threshold=0.35 |
| 21 | Employee Attrition Prediction | HR Tech / People Analytics | NeuralNet (MLP) | F1 = 0.3902 · ROC-AUC = 0.6698 |
| 22 | ANN From Scratch — MNIST Digit Recognizer | Deep Learning Fundamentals | From-scratch ANN | Test Acc = 0.9740 · Macro F1 = 0.9739 |
| 23 | Insurance Premium Prediction | Insurance / Actuarial ML | RandomForest | RMSLE = 1.1586 · 1.2M real policies |
| 24 | Health Insurance Cross-Sell | Insurance / BFSI | ExtraTrees | ROC-AUC = 0.8404 · 67.8% calls saved |
| 25 | Ride Fare Price Prediction | Mobility / Ride-Hailing | LightGBM | RMSE = $3.81 · R² = 0.8419 · 55M rows |
| 26 | E-Commerce BI & SQL Analytics | Retail Analytics / BI | SQL + FastAPI + Streamlit | 40 tests passing · 100K orders |
| 27 | AgriSense AI | AgriTech | EfficientNetB3 | Accuracy = 98.44% · Macro-F1 = 0.9843 · 38 classes |
| 28 | **AI Loan Underwriting Platform** | **BFSI / Fintech** | **XGBoost (credit + fraud)** | **Credit ROC-AUC = 0.858 · Fraud ROC-AUC = 0.955 · 242 tests** |

---

## 📚 References

- Give Me Some Credit — [Kaggle competition](https://www.kaggle.com/c/GiveMeSomeCredit)
- Credit Card Fraud Detection (ULB) — [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Chen & Guestrin (2016) — [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- Lundberg & Lee (2017) — [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)
- Niculescu-Mizil & Caruana (2005) — [Predicting Good Probabilities With Supervised Learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- Chawla et al. (2002) — [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)