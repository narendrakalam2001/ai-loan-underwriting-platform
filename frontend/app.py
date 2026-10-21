# ============================================================
# USER PORTAL — AI Loan Underwriting Platform (Frontend)
# ============================================================
# A professional, bank-grade Streamlit portal with 4 screens:
#   1. Loan Application Form (personal + financial + document upload)
#   2. Real-Time Processing View (animated pipeline steps)
#   3. Decision Dashboard (APPROVE / MANUAL_REVIEW / DECLINE)
#   4. Application Tracker (lookup by Application ID + decision letter)
# Plus (Module 5): EN/हिन्दी toggle, AI-written (or template) explanation,
# decision-letter PDF from the API, and a floating loan-assistant chat.
#
# Talks to serving/api.py over HTTP — run both with:
#   uvicorn serving.api:app --reload --port 8000
#   streamlit run frontend/app.py
# ============================================================

import os
import time
import uuid
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("UNDERWRITING_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Loan Underwriting Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS — bank blue/white theme, mobile-friendly
# ============================================================

st.markdown("""
<style>
    :root {
        --bank-navy:  #0B3D91;
        --bank-blue:  #1E5FCC;
        --bank-light: #EEF3FC;
        --bank-green: #1F8A4C;
        --bank-amber: #C77700;
        --bank-red:   #C0392B;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background-color: #FAFBFE;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .portal-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 24px;
        background: linear-gradient(135deg, var(--bank-navy), var(--bank-blue));
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(11,61,145,0.25);
    }
    .portal-header .logo {
        font-size: 34px;
        background: white;
        border-radius: 50%;
        width: 52px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .portal-header h1 {
        color: white;
        font-size: 22px;
        margin: 0;
        font-weight: 700;
    }
    .portal-header p {
        color: #D6E4FF;
        margin: 0;
        font-size: 13px;
    }

    .decision-banner {
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    .decision-approve { background: linear-gradient(135deg, #1F8A4C, #2FAE64); }
    .decision-review   { background: linear-gradient(135deg, #C77700, #E09A2E); }
    .decision-decline { background: linear-gradient(135deg, #C0392B, #E0574A); }
    .decision-banner h1 { color: white; margin: 0; font-size: 30px; }
    .decision-banner p  { color: rgba(255,255,255,0.9); margin-top: 6px; font-size: 15px; }

    .info-card {
        background: white;
        border: 1px solid #E4E9F5;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* ── Floating chat (the ONLY st.popover in the app is the chat) ── */
    div[data-testid="stPopover"] {
        position: fixed; bottom: 22px; right: 22px; z-index: 9999; width: auto !important;
    }
    [data-testid="stPopoverButton"] {
        background: #25D366; color: white; border: none; border-radius: 28px;
        padding: 10px 18px; font-weight: 600; box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    [data-testid="stPopoverButton"]:hover,
    [data-testid="stPopoverButton"][aria-expanded="true"] { background: #1EBE5A; color: white; }
    div[data-testid="stPopoverBody"] { width: min(380px, 92vw); }
    div[data-testid="stChatMessage"] { border-radius: 12px; padding: 8px 10px; background: #FFFFFF;
                                       border: 1px solid #E4E9F5; }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #DCF8C6; }

    @media (max-width: 640px) {
        .portal-header h1 { font-size: 18px; }
        .decision-banner h1 { font-size: 22px; }
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="portal-header">
        <div class="logo">🏦</div>
        <div>
            <h1>AI Loan Underwriting System</h1>
            <p>Instant, explainable loan decisions — powered by Credit Risk AI + Fraud Detection AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "apply"        # "apply" | "track"
if "step" not in st.session_state:
    st.session_state.step = 1              # 1=form, 2=processing, 3=decision
if "doc_result" not in st.session_state:
    st.session_state.doc_result = None
if "decision" not in st.session_state:
    st.session_state.decision = None
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = "web-" + uuid.uuid4().hex[:16]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of (role, text)
if "expl_cache" not in st.session_state:
    st.session_state.expl_cache = {}            # (application_id, lang) -> {"text","source"}
if "letter_cache" not in st.session_state:
    st.session_state.letter_cache = {}          # (application_id, lang) -> (bytes, language_used)


def current_lang() -> str:
    return "hi" if st.session_state.get("lang_choice") == "हिन्दी" else "en"


# ── API helpers (Module 5). Every call degrades gracefully: the portal must
#    still work when the AI explainer / letter service is unavailable. ───────

def fetch_explanation(application_id: str, lang: str):
    key = (application_id, lang)
    if key in st.session_state.expl_cache:          # cached: reruns (e.g. chat) must not re-call the LLM
        return st.session_state.expl_cache[key]
    try:
        r = requests.post(f"{API_URL}/explain", json={"application_id": application_id, "language": lang}, timeout=40)
        if r.status_code == 200:
            st.session_state.expl_cache[key] = r.json()
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return None


def fetch_letter(application_id: str, lang: str):
    """Returns (pdf_bytes, language_used) or None."""
    key = (application_id, lang)
    if key in st.session_state.letter_cache:
        return st.session_state.letter_cache[key]
    try:
        r = requests.get(f"{API_URL}/application/{application_id}/letter", params={"language": lang}, timeout=60)
        if r.status_code == 200:
            out = (r.content, r.headers.get("X-Language-Used", lang))
            st.session_state.letter_cache[key] = out
            return out
    except requests.exceptions.RequestException:
        pass
    return None


def render_letter_download(application_id: str, lang: str, key: str):
    letter = fetch_letter(application_id, lang)
    if letter is None:
        st.warning("The decision letter is unavailable right now. Please try again shortly.")
        return
    pdf_bytes, used = letter
    if lang == "hi" and used != "hi":
        st.caption("Hindi PDF is not available on this server (no Devanagari font) — the letter is in English.")
    st.download_button(
        "📄 Download Decision Letter (PDF)",
        data=pdf_bytes,
        file_name=f"{application_id}_decision_letter.pdf",
        mime="application/pdf",
        key=key,
    )


def render_explanation(application_id: str, fallback_text: str, lang: str):
    expl = fetch_explanation(application_id, lang)
    if expl:
        st.info(expl["text"])
        st.caption("✨ AI-written explanation" if expl.get("source") == "gemini"
                   else "📄 Standard explanation (template)")
    else:
        st.info(fallback_text)
        st.caption("Technical explanation (the explanation service is unavailable)")


def go_to(page):
    st.session_state.page = page


render_header()

nav1, nav2, _, nav3 = st.columns([1, 1, 2, 2])
with nav1:
    if st.button("📝 Apply for a Loan", use_container_width=True,
                 type="primary" if st.session_state.page == "apply" else "secondary"):
        go_to("apply")
with nav2:
    if st.button("🔎 Track Application", use_container_width=True,
                 type="primary" if st.session_state.page == "track" else "secondary"):
        go_to("track")
with nav3:
    st.radio("Language / भाषा", ["English", "हिन्दी"], horizontal=True, key="lang_choice")

st.markdown("")


# ============================================================
# SCREEN 4 — APPLICATION TRACKER
# ============================================================

def render_tracker():
    st.subheader("Application Tracker")
    app_id = st.text_input("Enter your Application ID", placeholder="e.g. APP1A2B3C4D5E")

    if st.button("Look up application") and app_id:
        try:
            r = requests.get(f"{API_URL}/application/{app_id}", timeout=15)
            if r.status_code == 200:
                data = r.json()
                _render_tracker_result(data)
            elif r.status_code == 404:
                st.error("No application found with that ID. Please check and try again.")
            else:
                st.error(f"API error: HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the underwriting API: {e}")


def _render_tracker_result(data: dict):
    verdict = data["final_verdict"]
    color = {"APPROVE": "🟢", "MANUAL_REVIEW": "🟡", "DECLINE": "🔴"}.get(verdict, "⚪")

    st.markdown(f"### {color} Status: **{verdict.replace('_', ' ')}**")

    timeline_cols = st.columns(3)
    timeline_cols[0].markdown(f"**Submitted**\n\n{data.get('decided_at', '—')[:19].replace('T', ' ')}")
    timeline_cols[1].markdown(f"**Decision**\n\n{verdict.replace('_', ' ').title()}")
    timeline_cols[2].markdown(f"**Confidence**\n\n{data.get('confidence_score', 0):.0%}")

    st.markdown("---")
    st.markdown("**Explanation**")
    render_explanation(data["application_id"], data.get("explanation", "—"), current_lang())

    if data.get("improvement_suggestions"):
        st.markdown("**Suggestions**")
        for s in data["improvement_suggestions"]:
            st.markdown(f"- {s}")

    with st.expander("Full submitted details"):
        st.json(data.get("module_outputs", {}))

    render_letter_download(data["application_id"], current_lang(), key="tracker_letter")


# ============================================================
# SCREEN 1 — LOAN APPLICATION FORM
# ============================================================

def render_application_form():
    st.subheader("Screen 1 — Loan Application")

    with st.form("loan_application_form"):
        st.markdown("##### Personal Details")
        c1, c2, c3 = st.columns(3)
        full_name = c1.text_input("Full Name", "")
        age = c2.number_input("Age", min_value=18, max_value=75, value=32)
        employment_type = c3.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner"])

        c4, c5, c6 = st.columns(3)
        monthly_income = c4.number_input("Monthly Income (₹)", min_value=0.0, value=45000.0, step=1000.0,
                                          help="Your actual take-home income per month, in ₹")
        loan_amount = c5.number_input("Loan Amount Requested (₹)", min_value=10000.0, value=500000.0, step=10000.0)
        loan_purpose = c6.selectbox("Loan Purpose", ["Personal", "Home Renovation", "Education", "Medical", "Business", "Wedding"])

        c7, c8 = st.columns(2)
        tenure_months = c7.slider("Tenure (months)", 6, 84, 36)
        years_employment = c8.number_input("Years of Employment", min_value=0.0, value=3.0, step=0.5)

        st.markdown("##### Financial Details")
        st.caption("These are normally pulled from a credit bureau report — shown here as fields "
                   "you can fill in yourself for this demo.")
        d1, d2, d3, d4 = st.columns(4)
        dependents = d1.number_input("Number of Dependents", min_value=0, max_value=15, value=1)
        existing_emis = d2.number_input("Existing EMIs (₹/month)", min_value=0.0, value=0.0, step=500.0)
        credit_utilization_pct = d3.number_input("Credit Utilization (%)", min_value=0.0, max_value=200.0,
                                                   value=20.0, step=5.0,
                                                   help="% of your total available credit limit currently in use")
        open_credit_lines = d4.number_input("Open Credit Lines / Loans", min_value=0, max_value=30, value=3)

        e1, e2, e3, e4 = st.columns(4)
        real_estate_loans = e1.number_input("Mortgages / Property Loans", min_value=0, max_value=10, value=0)
        times_30_59 = e2.number_input("Times 30-59 Days Late (2 yrs)", min_value=0, max_value=20, value=0)
        times_60_89 = e3.number_input("Times 60-89 Days Late (2 yrs)", min_value=0, max_value=20, value=0)
        times_90 = e4.number_input("Times 90+ Days Late (2 yrs)", min_value=0, max_value=20, value=0)

        st.markdown("##### Document Upload")
        uploaded_file = st.file_uploader("Upload Bank Statement (PDF)", type=["pdf"])

        submitted = st.form_submit_button("Continue → Submit Application", use_container_width=True)

    if uploaded_file is not None and st.session_state.doc_result is None:
        with st.spinner("Parsing bank statement..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                r = requests.post(f"{API_URL}/upload-document", files=files, timeout=30)
                if r.status_code == 200:
                    st.session_state.doc_result = r.json()
                else:
                    st.error(f"Document parsing failed: {r.json().get('detail', r.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the underwriting API: {e}")

    if st.session_state.doc_result:
        doc = st.session_state.doc_result
        st.markdown("###### Extracted from your statement")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected Income", f"₹{doc.get('avg_monthly_income', 0):,.0f}/mo")
        m2.metric("Employer", doc.get("employer") or "Not detected")
        m3.metric("Existing Obligations", f"₹{doc.get('existing_monthly_emi', 0):,.0f}/mo")
        m4.metric("Account Stability", doc.get("account_stability", "—"))
        if doc.get("warnings"):
            for w in doc["warnings"]:
                st.warning(w)

    if submitted:
        st.session_state.form_data = {
            "full_name": full_name or "Applicant",
            "age": float(age),
            "employment_type": employment_type,
            "years_employment": float(years_employment),
            "monthly_income": float(monthly_income),
            "dependents": int(dependents),
            "existing_emis": float(existing_emis),
            "credit_utilization_pct": float(credit_utilization_pct),
            "open_credit_lines": int(open_credit_lines),
            "real_estate_loans": int(real_estate_loans),
            "times_30_59_days_late": int(times_30_59),
            "times_60_89_days_late": int(times_60_89),
            "times_90_days_late": int(times_90),
            "loan_amount_requested": float(loan_amount),
            "loan_purpose": loan_purpose,
            "tenure_months": int(tenure_months),
            "document_id": (st.session_state.doc_result or {}).get("document_id"),
        }
        st.session_state.step = 2
        st.rerun()


# ============================================================
# SCREEN 2 — REAL-TIME PROCESSING VIEW
# ============================================================

def render_processing():
    st.subheader("Screen 2 — Processing Your Application")

    progress_box = st.container()
    steps = [
        "Document Verified",
        "Fraud Check",
        "Credit Risk Assessment",
        "Final Decision",
    ]
    placeholders = [progress_box.empty() for _ in steps]

    for ph, label in zip(placeholders, steps):
        ph.markdown(f"⟳ **Step:** {label} — Processing...")
        time.sleep(0.4)

    try:
        r = requests.post(f"{API_URL}/underwrite", json=st.session_state.form_data, timeout=60)
        r.raise_for_status()
        decision = r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the underwriting API: {e}")
        if st.button("← Back to form"):
            st.session_state.step = 1
            st.rerun()
        return

    doc_ok = st.session_state.form_data.get("document_id") is not None
    fraud = decision["module_outputs"]["fraud_detection"]
    credit = decision["module_outputs"]["credit_risk"]

    placeholders[0].markdown(
        "✅ **Step 1:** Document Verified" if doc_ok else "○ **Step 1:** No document uploaded — skipped"
    )
    time.sleep(0.3)
    fraud_icon = "✅" if fraud["verdict"] == "APPROVE" else ("⚠️" if fraud["verdict"] == "REVIEW" else "🛑")
    placeholders[1].markdown(
        f"{fraud_icon} **Step 2:** Fraud Check — {fraud['verdict']} (score: {fraud['fraud_probability']:.2f})"
    )
    time.sleep(0.3)
    placeholders[2].markdown(
        f"✅ **Step 3:** Credit Risk Assessment — {credit['risk_band']} risk "
        f"(default probability: {credit['default_probability']:.2f})"
    )
    time.sleep(0.3)
    placeholders[3].markdown(f"✅ **Step 4:** Final Decision — **{decision['final_verdict'].replace('_', ' ')}**")
    time.sleep(0.4)

    st.session_state.decision = decision
    st.session_state.step = 3
    time.sleep(0.5)
    st.rerun()


# ============================================================
# SCREEN 3 — DECISION DASHBOARD
# ============================================================

def render_decision():
    decision = st.session_state.decision
    verdict = decision["final_verdict"]

    banner_class = {"APPROVE": "decision-approve", "MANUAL_REVIEW": "decision-review",
                     "DECLINE": "decision-decline"}.get(verdict, "decision-review")
    banner_title = {"APPROVE": "✅ APPROVED", "MANUAL_REVIEW": "🕒 UNDER REVIEW",
                    "DECLINE": "❌ DECLINED"}.get(verdict, verdict)

    st.markdown(f"""
    <div class="decision-banner {banner_class}">
        <h1>{banner_title}</h1>
        <p>Application ID: {decision['application_id']}</p>
    </div>
    """, unsafe_allow_html=True)

    fd = st.session_state.form_data
    credit = decision["module_outputs"]["credit_risk"]

    if verdict == "APPROVE":
        annual_rate = 10.5 + credit["default_probability"] * 15   # risk-based pricing illustration
        principal = fd["loan_amount_requested"]
        n = fd["tenure_months"]
        r_m = annual_rate / 1200
        emi = principal * r_m * (1 + r_m) ** n / (((1 + r_m) ** n) - 1) if r_m > 0 else principal / n

        c1, c2, c3 = st.columns(3)
        c1.metric("Loan Amount", f"₹{principal:,.0f}")
        c2.metric("Interest Rate", f"{annual_rate:.1f}%")
        c3.metric("EMI", f"₹{emi:,.0f}/month")

    st.markdown("##### Explanation")
    render_explanation(decision["application_id"], decision["explanation"], current_lang())
    with st.expander("Technical summary (from the decision engine)"):
        st.write(decision["explanation"])

    if decision.get("shap_top_drivers"):
        st.markdown("##### Key Factors")
        shap_df = pd.DataFrame(decision["shap_top_drivers"])
        st.bar_chart(shap_df.set_index("feature")["shap_value"])

    if verdict == "MANUAL_REVIEW":
        st.warning("Expected review time: 2-3 business days. We may contact you for additional documents.")

    if decision.get("improvement_suggestions"):
        st.markdown("##### Suggestions")
        for s in decision["improvement_suggestions"]:
            st.markdown(f"- {s}")

    st.markdown("---")
    render_letter_download(decision["application_id"], current_lang(), key="decision_letter")
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("📋 Go to Application Tracker", use_container_width=True):
            st.session_state.page = "track"
            st.session_state.step = 1
            st.rerun()
    with cc2:
        if st.button("📝 Start a New Application", use_container_width=True):
            st.session_state.step = 1
            st.session_state.doc_result = None
            st.session_state.decision = None
            st.rerun()


# ============================================================
# FLOATING LOAN-ASSISTANT CHAT (Module 5D) — bottom-right, WhatsApp-style
# ============================================================

_CHAT_GREETING = {
    "en": "Hello! I can check indicative eligibility, calculate EMIs, look up an application by its ID and answer loan FAQs.",
    "hi": "नमस्ते! मैं पात्रता का संकेत, EMI गणना, आवेदन आईडी से स्टेटस और लोन FAQ में मदद कर सकता हूँ।",
}


def _send_chat(message: str, lang: str) -> str:
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"session_id": st.session_state.chat_session_id, "message": message, "language": lang},
            timeout=45,
        )
        if r.status_code == 200:
            return r.json().get("reply", "…")
        if r.status_code == 422:
            return "Your message is too long. Please keep it under 500 characters."
        return f"Sorry, the assistant is unavailable right now (HTTP {r.status_code})."
    except requests.exceptions.RequestException:
        return "Sorry, I couldn't reach the assistant. Please try again in a moment."


def render_chat():
    lang = current_lang()
    with st.popover("💬 Loan Assistant"):
        box = st.container(height=320)
        with st.form("chat_form", clear_on_submit=True):
            msg = st.text_input("Message", max_chars=500, label_visibility="collapsed",
                                placeholder="Ask about EMI, eligibility, status…")
            sent = st.form_submit_button("Send ➤", use_container_width=True)
        if sent and msg.strip():
            st.session_state.chat_history.append(("user", msg.strip()))
            with st.spinner("Typing…"):
                reply = _send_chat(msg.strip(), lang)
            st.session_state.chat_history.append(("assistant", reply))
        with box:
            st.chat_message("assistant").write(_CHAT_GREETING[lang])
            for role, text in st.session_state.chat_history:
                st.chat_message(role).write(text)
        st.caption("Indicative help only — the assistant cannot approve or decline a loan.")


# ============================================================
# ROUTER
# ============================================================

if st.session_state.page == "track":
    render_tracker()
else:
    if st.session_state.step == 1:
        render_application_form()
    elif st.session_state.step == 2:
        render_processing()
    elif st.session_state.step == 3:
        render_decision()

render_chat()
