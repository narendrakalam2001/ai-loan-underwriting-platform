# ============================================================
# MONITORING DASHBOARD — AI Loan Underwriting Platform
# ============================================================
# Internal, bank-staff-facing Streamlit dashboard. Structure
# follows the reference projects' monitoring_dashboard.py, but
# covers BOTH engines side by side (a "Credit Risk" / "Fraud"
# selector at the top of each section) plus the Decision Engine's
# own audit-trail-derived KPIs.
#
# Sections:
#   1. Alerts (drift / score / decision-rate)
#   2. Champion vs Challenger
#   3. KPIs + charts
#   4. PSI Drift Monitor
#   5. Recent Predictions (from each engine's monitor_scores.csv
#      + the Decision Engine's audit trail)
# ============================================================

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import (
    CREDIT_MODEL_SUBDIR, FRAUD_MODEL_SUBDIR, AUDIT_TRAIL_DIR,
    PSI_MODERATE, PSI_HIGH, SCORE_MEAN_ALERT,
)

st.set_page_config(page_title="Underwriting Platform — Ops Dashboard", layout="wide")
st.title("🏦 AI Loan Underwriting Platform — Operations Dashboard")
st.caption("Internal view for underwriting / risk operations staff")

ENGINES = {
    "Credit Risk": {"model_dir": CREDIT_MODEL_SUBDIR, "decision_col": "decision"},
    "Fraud":       {"model_dir": FRAUD_MODEL_SUBDIR,  "decision_col": "decision"},
}


def _paths(model_dir: str):
    return {
        "monitor":    os.path.join(model_dir, "monitor_scores.csv"),
        "psi":        os.path.join(model_dir, "feature_drift_report.csv"),
        "challenger": os.path.join(model_dir, "challenger_log.json"),
        "registry":   os.path.join(model_dir, "latest_model.json"),
    }


# ============================================================
# SECTION 1 — REAL-TIME ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Section 1 — Real-Time Monitoring Alerts")

alert_cols = st.columns(2)

for col, (engine_name, engine_cfg) in zip(alert_cols, ENGINES.items()):
    with col:
        st.markdown(f"**{engine_name} Engine**")
        p = _paths(engine_cfg["model_dir"])
        alerts_found = False

        if os.path.exists(p["monitor"]):
            df_monitor = pd.read_csv(p["monitor"])
            if "score" in df_monitor.columns:
                avg_score = df_monitor["score"].mean()
                if avg_score > SCORE_MEAN_ALERT:
                    st.error(f"🔴 HIGH AVG SCORE: {avg_score:.4f} (threshold {SCORE_MEAN_ALERT})")
                    alerts_found = True
            if "decision" in df_monitor.columns:
                decline_rate = df_monitor["decision"].astype(str).str.contains("DECLINE|BLOCK").mean()
                if decline_rate > 0.25:
                    st.warning(f"🟡 HIGH DECLINE/BLOCK RATE: {decline_rate:.1%} (expected < 25%)")
                    alerts_found = True
                review_rate = df_monitor["decision"].astype(str).str.contains("REVIEW").mean()
                if review_rate > 0.20:
                    st.warning(f"🟡 HIGH REVIEW QUEUE: {review_rate:.1%} (expected < 20%)")
                    alerts_found = True
        else:
            st.info("No monitor scores yet — run training_pipeline.py")

        if os.path.exists(p["psi"]):
            df_psi = pd.read_csv(p["psi"])
            if "drift_score" in df_psi.columns and len(df_psi) > 0:
                max_psi = df_psi["drift_score"].max()
                top_feature = df_psi.iloc[0]["feature"] if "feature" in df_psi.columns else "unknown"
                if max_psi >= PSI_HIGH:
                    st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}' — retrain now")
                    alerts_found = True
                elif max_psi >= PSI_MODERATE:
                    st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}' — monitor")
                    alerts_found = True

        if not alerts_found:
            st.success("✅ All systems normal")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER
# ============================================================

st.markdown("---")
st.subheader("🏆 Section 2 — Champion vs Challenger")

engine_tab_names = list(ENGINES.keys())
tabs = st.tabs(engine_tab_names)

for tab, engine_name in zip(tabs, engine_tab_names):
    with tab:
        p = _paths(ENGINES[engine_name]["model_dir"])

        if os.path.exists(p["challenger"]):
            with open(p["challenger"]) as f:
                challenger_log = json.load(f)

            if challenger_log:
                latest = challenger_log[-1]
                decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
                icon = "✅" if latest["decision"] == "PROMOTED" else "❌"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest Challenger", latest.get("challenger_name", "—"))
                c2.metric("Challenger F1", latest.get("challenger_f1", "—"))
                c3.metric("Champion F1", latest.get("champion_f1") or "First Run")
                c4.metric("Challenger ROC-AUC", latest.get("challenger_roc_auc", "—"))

                st.markdown(
                    f"<h4 style='color:{decision_color}'>{icon} Decision: {latest['decision']} — "
                    f"{latest.get('reason', '')}</h4>",
                    unsafe_allow_html=True,
                )

                if latest.get("gates"):
                    g = latest["gates"]
                    g1, g2, g3 = st.columns(3)
                    g1.metric("F1 Gate", "✅ Pass" if g.get("f1_improvement_passed") else "❌ Fail")
                    g2.metric("ROC-AUC Gate", "✅ Pass" if g.get("roc_auc_passed") else "❌ Fail")
                    g3.metric("Gap Gate", "✅ Pass" if g.get("gap_passed") else "❌ Fail")

                if len(challenger_log) > 1:
                    with st.expander("View full challenger history"):
                        hist_df = pd.DataFrame(challenger_log)
                        cols = [c for c in ["evaluated_at", "challenger_name", "challenger_f1",
                                            "champion_name", "champion_f1", "decision", "reason"]
                                if c in hist_df.columns]
                        st.dataframe(hist_df[cols], use_container_width=True)
            else:
                st.info("Challenger log is empty.")
        else:
            st.info(f"No challenger log found for {engine_name} yet.")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Section 3 — Model Performance KPIs")

tabs2 = st.tabs(engine_tab_names)

for tab, engine_name in zip(tabs2, engine_tab_names):
    with tab:
        p = _paths(ENGINES[engine_name]["model_dir"])

        if os.path.exists(p["monitor"]):
            df = pd.read_csv(p["monitor"])

            k1, k2, k3, k4 = st.columns(4)
            if "label" in df.columns:
                k1.metric("Positive Rate (actual)", f"{df['label'].mean():.2%}")
            if "decision" in df.columns:
                dec = df["decision"].astype(str)
                k2.metric("Auto-Approve Rate", f"{(dec == 'APPROVE').mean():.1%}")
                k3.metric("Review Queue Rate", f"{dec.str.contains('REVIEW').mean():.1%}")
                k4.metric("Decline/Block Rate", f"{dec.str.contains('DECLINE|BLOCK').mean():.1%}")

            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Score Distribution**")
                if "score" in df.columns:
                    fig, ax = plt.subplots()
                    ax.hist(df["score"], bins=40, alpha=0.75, color="#1E5FCC")
                    ax.set_xlabel("Model score")
                    st.pyplot(fig)
                    plt.close(fig)
            with cb:
                st.markdown("**Decision Distribution**")
                if "decision" in df.columns:
                    st.bar_chart(df["decision"].value_counts())

            st.markdown("**Score Statistics**")
            if "score" in df.columns:
                st.write(df["score"].describe().to_frame().T.round(4))
        else:
            st.warning(f"No monitor_scores.csv found for {engine_name}. Run training_pipeline.py first.")


# ============================================================
# SECTION 4 — PSI DRIFT MONITOR
# ============================================================

st.markdown("---")
st.subheader("📉 Section 4 — Feature Drift Report (PSI)")

tabs3 = st.tabs(engine_tab_names)

for tab, engine_name in zip(tabs3, engine_tab_names):
    with tab:
        p = _paths(ENGINES[engine_name]["model_dir"])

        if os.path.exists(p["psi"]):
            df_psi = pd.read_csv(p["psi"])
            if "drift_score" in df_psi.columns and len(df_psi) > 0:
                def _flag(v):
                    if v >= PSI_HIGH:
                        return "🔴 CRITICAL"
                    elif v >= PSI_MODERATE:
                        return "🟡 MODERATE"
                    return "🟢 OK"

                df_psi["status"] = df_psi["drift_score"].apply(_flag)
                st.dataframe(df_psi.head(10), use_container_width=True)

                top10 = df_psi.head(10)
                colors = ["#E74C3C" if v >= PSI_HIGH else "#F39C12" if v >= PSI_MODERATE else "#2ECC71"
                          for v in top10["drift_score"]]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(top10["feature"][::-1], top10["drift_score"][::-1], color=colors[::-1])
                ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
                ax.axvline(PSI_HIGH, color="red", linestyle="--", label=f"Critical ({PSI_HIGH})")
                ax.set_xlabel("PSI Score")
                ax.legend(fontsize=8)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("Drift file found but empty / missing 'drift_score' column.")
        else:
            st.warning(f"No feature_drift_report.csv found for {engine_name}.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS (from Decision Engine audit trail)
# ============================================================

st.markdown("---")
st.subheader("📋 Section 5 — Recent Applications (Decision Engine)")

audit_files = sorted(glob.glob(os.path.join(AUDIT_TRAIL_DIR, "*.json")), key=os.path.getmtime, reverse=True)

if audit_files:
    rows = []
    for fp in audit_files[:50]:
        try:
            with open(fp) as f:
                d = json.load(f)
            rows.append({
                "application_id":  d.get("application_id"),
                "final_verdict":   d.get("final_verdict"),
                "confidence":      d.get("confidence_score"),
                "credit_risk_band": d.get("module_outputs", {}).get("credit_risk", {}).get("risk_band"),
                "fraud_verdict":    d.get("module_outputs", {}).get("fraud_detection", {}).get("verdict"),
                "decided_at":       d.get("decided_at"),
            })
        except Exception:
            continue

    recent_df = pd.DataFrame(rows)

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        verdict_filter = st.multiselect("Filter by decision",
                                         options=sorted(recent_df["final_verdict"].dropna().unique()),
                                         default=[])
    with fcol2:
        band_filter = st.multiselect("Filter by credit risk band",
                                      options=sorted(recent_df["credit_risk_band"].dropna().unique()),
                                      default=[])

    filtered = recent_df.copy()
    if verdict_filter:
        filtered = filtered[filtered["final_verdict"].isin(verdict_filter)]
    if band_filter:
        filtered = filtered[filtered["credit_risk_band"].isin(band_filter)]

    st.dataframe(filtered, use_container_width=True)

    selected_id = st.selectbox("View full SHAP explanation for application:",
                                options=["—"] + filtered["application_id"].tolist())
    if selected_id != "—":
        match = next((f for f in audit_files if selected_id in f), None)
        if match:
            with open(match) as f:
                st.json(json.load(f))
else:
    st.info(
        "No applications processed yet. Applications appear here once the User Portal "
        "(frontend/app.py) or the API's /underwrite endpoint has been used at least once."
    )
