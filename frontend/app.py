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

def _resolve_api_url() -> str:
    """Env var first (Render / Docker), then Streamlit secrets (Community Cloud),
    then localhost for local development."""
    url = os.getenv("UNDERWRITING_API_URL")
    if not url:
        try:
            url = st.secrets.get("UNDERWRITING_API_URL")
        except Exception:        # no secrets file configured
            url = None
    return (url or "http://localhost:8000").rstrip("/")


API_URL = _resolve_api_url()

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
    st.markdown(f"""
    <div class="portal-header">
        <div class="logo">🏦</div>
        <div>
            <h1>{tr("header_title")}</h1>
            <p>{tr("header_sub")}</p>
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


# ============================================================
# UI TRANSLATIONS (English / हिन्दी) — every static label in the portal.
# Text that comes from the API (the explanation, the letter, chat replies) is
# already written in the chosen language by the API itself.
# ============================================================

_TX = {
    "header_title": ("AI Loan Underwriting System", "एआई लोन अंडरराइटिंग सिस्टम"),
    "header_sub": ("Instant, explainable loan decisions — powered by Credit Risk AI + Fraud Detection AI",
                   "तुरंत और समझाने योग्य लोन निर्णय — क्रेडिट रिस्क एआई + फ्रॉड डिटेक्शन एआई द्वारा संचालित"),
    "nav_apply": ("📝 Apply for a Loan", "📝 लोन के लिए आवेदन करें"),
    "nav_track": ("🔎 Track Application", "🔎 आवेदन ट्रैक करें"),
    # tracker
    "tracker_title": ("Application Tracker", "आवेदन ट्रैकर"),
    "tracker_input": ("Enter your Application ID", "अपना आवेदन आईडी दर्ज करें"),
    "tracker_ph": ("e.g. APP1A2B3C4D5E", "जैसे APP1A2B3C4D5E"),
    "tracker_btn": ("Look up application", "आवेदन खोजें"),
    "not_found": ("No application found with that ID. Please check and try again.",
                  "इस आईडी से कोई आवेदन नहीं मिला। कृपया जाँचकर फिर कोशिश करें।"),
    "api_error": ("API error: HTTP {code}", "एपीआई त्रुटि: HTTP {code}"),
    "unreachable": ("Could not reach the underwriting API: {err}", "अंडरराइटिंग एपीआई से संपर्क नहीं हो सका: {err}"),
    "status": ("Status", "स्थिति"),
    "submitted": ("Submitted", "जमा किया गया"),
    "decision": ("Decision", "निर्णय"),
    "confidence": ("Confidence", "विश्वास स्तर"),
    "explanation": ("Explanation", "स्पष्टीकरण"),
    "suggestions": ("Suggestions", "सुझाव"),
    "full_details": ("Full submitted details", "जमा किए गए पूरे विवरण"),
    # verdict labels
    "v_APPROVE": ("APPROVE", "स्वीकृत"),
    "v_MANUAL_REVIEW": ("MANUAL REVIEW", "मैन्युअल समीक्षा"),
    "v_DECLINE": ("DECLINE", "अस्वीकृत"),
    "v_BLOCK": ("BLOCK", "ब्लॉक"),
    "v_REVIEW": ("REVIEW", "समीक्षा"),
    # form
    "form_title": ("Screen 1 — Loan Application", "स्क्रीन 1 — लोन आवेदन"),
    "personal": ("Personal Details", "व्यक्तिगत विवरण"),
    "full_name": ("Full Name", "पूरा नाम"),
    "age": ("Age", "आयु"),
    "emp_type": ("Employment Type", "रोज़गार का प्रकार"),
    "monthly_income": ("Monthly Income (₹)", "मासिक आय (₹)"),
    "monthly_income_help": ("Your actual take-home income per month, in ₹", "आपकी हर महीने की वास्तविक हाथ में आने वाली आय, ₹ में"),
    "loan_amount": ("Loan Amount Requested (₹)", "माँगी गई लोन राशि (₹)"),
    "loan_purpose": ("Loan Purpose", "लोन का उद्देश्य"),
    "tenure": ("Tenure (months)", "अवधि (महीने)"),
    "years_emp": ("Years of Employment", "नौकरी के वर्ष"),
    "financial": ("Financial Details", "वित्तीय विवरण"),
    "fin_caption": ("These are normally pulled from a credit bureau report — shown here as fields you can fill in yourself for this demo.",
                    "ये जानकारी सामान्यतः क्रेडिट ब्यूरो रिपोर्ट से ली जाती है — इस डेमो में आप इन्हें खुद भर सकते हैं।"),
    "dependents": ("Number of Dependents", "आश्रितों की संख्या"),
    "existing_emis": ("Existing EMIs (₹/month)", "मौजूदा EMI (₹/माह)"),
    "util": ("Credit Utilization (%)", "क्रेडिट उपयोग (%)"),
    "util_help": ("% of your total available credit limit currently in use", "आपकी कुल उपलब्ध क्रेडिट सीमा का कितना प्रतिशत अभी उपयोग में है"),
    "open_lines": ("Open Credit Lines / Loans", "चालू क्रेडिट लाइन / लोन"),
    "mortgages": ("Mortgages / Property Loans", "गृह / संपत्ति लोन"),
    "late_30": ("Times 30-59 Days Late (2 yrs)", "30-59 दिन देरी की बार (2 वर्ष)"),
    "late_60": ("Times 60-89 Days Late (2 yrs)", "60-89 दिन देरी की बार (2 वर्ष)"),
    "late_90": ("Times 90+ Days Late (2 yrs)", "90+ दिन देरी की बार (2 वर्ष)"),
    "doc_upload": ("Document Upload", "दस्तावेज़ अपलोड"),
    "upload_label": ("Upload Bank Statement (PDF)", "बैंक स्टेटमेंट अपलोड करें (PDF)"),
    "submit": ("Continue → Submit Application", "आगे बढ़ें → आवेदन जमा करें"),
    "parsing": ("Parsing bank statement...", "बैंक स्टेटमेंट पढ़ा जा रहा है..."),
    "parse_failed": ("Document parsing failed: {detail}", "दस्तावेज़ पढ़ने में विफल: {detail}"),
    "extracted": ("Extracted from your statement", "आपके स्टेटमेंट से निकाली गई जानकारी"),
    "det_income": ("Detected Income", "पहचानी गई आय"),
    "employer": ("Employer", "नियोक्ता"),
    "not_detected": ("Not detected", "पहचाना नहीं गया"),
    "obligations": ("Existing Obligations", "मौजूदा देनदारियाँ"),
    "stability": ("Account Stability", "खाते की स्थिरता"),
    "per_month": ("/mo", "/माह"),
    "opt_Salaried": ("Salaried", "वेतनभोगी"),
    "opt_Self-Employed": ("Self-Employed", "स्वरोजगार"),
    "opt_Business Owner": ("Business Owner", "व्यवसाय मालिक"),
    "opt_Personal": ("Personal", "व्यक्तिगत"),
    "opt_Home Renovation": ("Home Renovation", "घर का नवीनीकरण"),
    "opt_Education": ("Education", "शिक्षा"),
    "opt_Medical": ("Medical", "चिकित्सा"),
    "opt_Business": ("Business", "व्यवसाय"),
    "opt_Wedding": ("Wedding", "विवाह"),
    # processing
    "proc_title": ("Screen 2 — Processing Your Application", "स्क्रीन 2 — आपका आवेदन प्रोसेस हो रहा है"),
    "step_doc": ("Document Verified", "दस्तावेज़ सत्यापित"),
    "step_fraud": ("Fraud Check", "फ्रॉड जाँच"),
    "step_credit": ("Credit Risk Assessment", "क्रेडिट रिस्क आकलन"),
    "step_final": ("Final Decision", "अंतिम निर्णय"),
    "step_word": ("Step", "चरण"),
    "processing": ("Processing...", "प्रोसेस हो रहा है..."),
    "no_doc": ("No document uploaded — skipped", "कोई दस्तावेज़ अपलोड नहीं हुआ — छोड़ा गया"),
    "score": ("score", "स्कोर"),
    "risk": ("risk", "जोखिम"),
    "default_prob": ("default probability", "डिफ़ॉल्ट की संभावना"),
    "back_form": ("← Back to form", "← फ़ॉर्म पर वापस"),
    # decision
    "banner_APPROVE": ("✅ APPROVED", "✅ स्वीकृत"),
    "banner_MANUAL_REVIEW": ("🕒 UNDER REVIEW", "🕒 समीक्षाधीन"),
    "banner_DECLINE": ("❌ DECLINED", "❌ अस्वीकृत"),
    "app_id": ("Application ID", "आवेदन आईडी"),
    "m_amount": ("Loan Amount", "लोन राशि"),
    "m_rate": ("Interest Rate", "ब्याज दर"),
    "m_emi": ("EMI", "EMI"),
    "tech_summary": ("Technical summary (from the decision engine)", "तकनीकी सारांश (निर्णय इंजन से)"),
    "key_factors": ("Key Factors", "मुख्य कारक"),
    "review_time": ("Expected review time: 2-3 business days. We may contact you for additional documents.",
                    "अपेक्षित समीक्षा समय: 2-3 कार्य दिवस। हम आपसे अतिरिक्त दस्तावेज़ों के लिए संपर्क कर सकते हैं।"),
    "go_tracker": ("📋 Go to Application Tracker", "📋 आवेदन ट्रैकर पर जाएँ"),
    "new_app": ("📝 Start a New Application", "📝 नया आवेदन शुरू करें"),
    "letter_unavailable": ("The decision letter is unavailable right now. Please try again shortly.",
                           "निर्णय पत्र अभी उपलब्ध नहीं है। कृपया थोड़ी देर बाद फिर कोशिश करें।"),
    "letter_en_only": ("Hindi PDF is not available on this server (no Devanagari font) — the letter is in English.",
                       "इस सर्वर पर हिन्दी PDF उपलब्ध नहीं है (देवनागरी फ़ॉन्ट नहीं) — पत्र अंग्रेज़ी में है।"),
    "letter_btn": ("📄 Download Decision Letter (PDF)", "📄 निर्णय पत्र डाउनलोड करें (PDF)"),
    "expl_ai": ("✨ AI-written explanation", "✨ एआई द्वारा लिखा गया स्पष्टीकरण"),
    "expl_template": ("📄 Standard explanation (template)", "📄 मानक स्पष्टीकरण (टेम्पलेट)"),
    "expl_unavailable": ("Technical explanation (the explanation service is unavailable)",
                         "तकनीकी स्पष्टीकरण (स्पष्टीकरण सेवा अभी उपलब्ध नहीं है)"),
    # chat
    "chat_btn": ("💬 Loan Assistant", "💬 लोन सहायक"),
    "chat_ph": ("Ask about EMI, eligibility, status…", "EMI, पात्रता या स्थिति के बारे में पूछें…"),
    "chat_send": ("Send ➤", "भेजें ➤"),
    "chat_typing": ("Typing…", "लिख रहा है…"),
    "chat_note": ("Indicative help only — the assistant cannot approve or decline a loan.",
                  "केवल संकेतात्मक सहायता — सहायक लोन स्वीकृत या अस्वीकृत नहीं कर सकता।"),
    "chat_too_long": ("Your message is too long. Please keep it under 500 characters.",
                      "आपका संदेश बहुत लंबा है। कृपया 500 अक्षरों से कम रखें।"),
    "chat_unavailable": ("Sorry, the assistant is unavailable right now (HTTP {code}).",
                         "क्षमा करें, सहायक अभी उपलब्ध नहीं है (HTTP {code})।"),
    "chat_unreachable": ("Sorry, I couldn't reach the assistant. Please try again in a moment.",
                         "क्षमा करें, सहायक से संपर्क नहीं हो सका। कृपया थोड़ी देर बाद फिर कोशिश करें।"),
}


def tr(key: str, **kw) -> str:
    """Translate a UI label into the language chosen in the radio (English fallback)."""
    pair = _TX.get(key)
    if pair is None:
        return key
    s = pair[1] if current_lang() == "hi" else pair[0]
    return s.format(**kw) if kw else s


def verdict_label(v: str) -> str:
    return tr(f"v_{v}") if f"v_{v}" in _TX else str(v).replace("_", " ")



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
        st.warning(tr("letter_unavailable"))
        return
    pdf_bytes, used = letter
    if lang == "hi" and used != "hi":
        st.caption(tr("letter_en_only"))
    st.download_button(
        tr("letter_btn"),
        data=pdf_bytes,
        file_name=f"{application_id}_decision_letter.pdf",
        mime="application/pdf",
        key=key,
    )


def render_explanation(application_id: str, fallback_text: str, lang: str):
    expl = fetch_explanation(application_id, lang)
    if expl:
        st.info(expl["text"])
        st.caption(tr("expl_ai") if expl.get("source") == "gemini" else tr("expl_template"))
    else:
        st.info(fallback_text)
        st.caption(tr("expl_unavailable"))


def go_to(page):
    st.session_state.page = page


render_header()

nav1, nav2, _, nav3 = st.columns([1, 1, 2, 2])
with nav1:
    if st.button(tr("nav_apply"), use_container_width=True,
                 type="primary" if st.session_state.page == "apply" else "secondary"):
        go_to("apply")
with nav2:
    if st.button(tr("nav_track"), use_container_width=True,
                 type="primary" if st.session_state.page == "track" else "secondary"):
        go_to("track")
with nav3:
    st.radio("Language / भाषा", ["English", "हिन्दी"], horizontal=True, key="lang_choice")

st.markdown("")


# ============================================================
# SCREEN 4 — APPLICATION TRACKER
# ============================================================

def render_tracker():
    st.subheader(tr("tracker_title"))
    app_id = st.text_input(tr("tracker_input"), placeholder=tr("tracker_ph"), key="tracker_app_id")

    if st.button(tr("tracker_btn")) and app_id:
        try:
            r = requests.get(f"{API_URL}/application/{app_id}", timeout=15)
            if r.status_code == 200:
                data = r.json()
                _render_tracker_result(data)
            elif r.status_code == 404:
                st.error(tr("not_found"))
            else:
                st.error(tr("api_error", code=r.status_code))
        except requests.exceptions.RequestException as e:
            st.error(tr("unreachable", err=e))


def _render_tracker_result(data: dict):
    verdict = data["final_verdict"]
    color = {"APPROVE": "🟢", "MANUAL_REVIEW": "🟡", "DECLINE": "🔴"}.get(verdict, "⚪")

    st.markdown(f"### {color} {tr('status')}: **{verdict_label(verdict)}**")

    timeline_cols = st.columns(3)
    timeline_cols[0].markdown(f"**{tr('submitted')}**\n\n{data.get('decided_at', '—')[:19].replace('T', ' ')}")
    timeline_cols[1].markdown(f"**{tr('decision')}**\n\n{verdict_label(verdict)}")
    timeline_cols[2].markdown(f"**{tr('confidence')}**\n\n{data.get('confidence_score', 0):.0%}")

    st.markdown("---")
    st.markdown(f"**{tr('explanation')}**")
    render_explanation(data["application_id"], data.get("explanation", "—"), current_lang())

    if data.get("improvement_suggestions"):
        st.markdown(f"**{tr('suggestions')}**")
        for s in data["improvement_suggestions"]:
            st.markdown(f"- {s}")

    with st.expander(tr("full_details")):
        st.json(data.get("module_outputs", {}))

    render_letter_download(data["application_id"], current_lang(), key="tracker_letter")


# ============================================================
# SCREEN 1 — LOAN APPLICATION FORM
# ============================================================

def render_application_form():
    st.subheader(tr("form_title"))

    with st.form("loan_application_form"):
        st.markdown(f"##### {tr('personal')}")
        c1, c2, c3 = st.columns(3)
        full_name = c1.text_input(tr("full_name"), "", key="f_full_name")
        age = c2.number_input(tr("age"), min_value=18, max_value=75, value=32, key="f_age")
        employment_type = c3.selectbox(tr("emp_type"), ["Salaried", "Self-Employed", "Business Owner"],
                                      format_func=lambda o: tr(f"opt_{o}"), key=f"f_emp_type_{current_lang()}")

        c4, c5, c6 = st.columns(3)
        monthly_income = c4.number_input(tr("monthly_income"), min_value=0.0, value=45000.0, step=1000.0,
                                          help=tr("monthly_income_help"), key="f_income")
        loan_amount = c5.number_input(tr("loan_amount"), min_value=10000.0, value=500000.0, step=10000.0, key="f_loan_amount")
        loan_purpose = c6.selectbox(tr("loan_purpose"), ["Personal", "Home Renovation", "Education", "Medical", "Business", "Wedding"],
                                    format_func=lambda o: tr(f"opt_{o}"), key=f"f_purpose_{current_lang()}")

        c7, c8 = st.columns(2)
        tenure_months = c7.slider(tr("tenure"), 6, 84, 36, key="f_tenure")
        years_employment = c8.number_input(tr("years_emp"), min_value=0.0, value=3.0, step=0.5, key="f_years_emp")

        st.markdown(f"##### {tr('financial')}")
        st.caption(tr("fin_caption"))
        d1, d2, d3, d4 = st.columns(4)
        dependents = d1.number_input(tr("dependents"), min_value=0, max_value=15, value=1, key="f_dependents")
        existing_emis = d2.number_input(tr("existing_emis"), min_value=0.0, value=0.0, step=500.0, key="f_emis")
        credit_utilization_pct = d3.number_input(tr("util"), min_value=0.0, max_value=200.0,
                                                   value=20.0, step=5.0,
                                                   help=tr("util_help"), key="f_util")
        open_credit_lines = d4.number_input(tr("open_lines"), min_value=0, max_value=30, value=3, key="f_open_lines")

        e1, e2, e3, e4 = st.columns(4)
        real_estate_loans = e1.number_input(tr("mortgages"), min_value=0, max_value=10, value=0, key="f_mortgages")
        times_30_59 = e2.number_input(tr("late_30"), min_value=0, max_value=20, value=0, key="f_late30")
        times_60_89 = e3.number_input(tr("late_60"), min_value=0, max_value=20, value=0, key="f_late60")
        times_90 = e4.number_input(tr("late_90"), min_value=0, max_value=20, value=0, key="f_late90")

        st.markdown(f"##### {tr('doc_upload')}")
        uploaded_file = st.file_uploader(tr("upload_label"), type=["pdf"], key="f_upload")

        submitted = st.form_submit_button(tr("submit"), use_container_width=True)

    if uploaded_file is not None and st.session_state.doc_result is None:
        with st.spinner(tr("parsing")):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                r = requests.post(f"{API_URL}/upload-document", files=files, timeout=30)
                if r.status_code == 200:
                    st.session_state.doc_result = r.json()
                else:
                    st.error(tr("parse_failed", detail=r.json().get('detail', r.text)))
            except requests.exceptions.RequestException as e:
                st.error(tr("unreachable", err=e))

    if st.session_state.doc_result:
        doc = st.session_state.doc_result
        st.markdown(f"###### {tr('extracted')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("det_income"), f"₹{doc.get('avg_monthly_income', 0):,.0f}{tr('per_month')}")
        m2.metric(tr("employer"), doc.get("employer") or tr("not_detected"))
        m3.metric(tr("obligations"), f"₹{doc.get('existing_monthly_emi', 0):,.0f}{tr('per_month')}")
        m4.metric(tr("stability"), doc.get("account_stability", "—"))
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
    st.subheader(tr("proc_title"))

    progress_box = st.container()
    steps = [tr("step_doc"), tr("step_fraud"), tr("step_credit"), tr("step_final")]
    placeholders = [progress_box.empty() for _ in steps]

    for ph, label in zip(placeholders, steps):
        ph.markdown(f"⟳ **{tr('step_word')}:** {label} — {tr('processing')}")
        time.sleep(0.4)

    try:
        r = requests.post(f"{API_URL}/underwrite", json=st.session_state.form_data, timeout=60)
        r.raise_for_status()
        decision = r.json()
    except requests.exceptions.RequestException as e:
        st.error(tr("unreachable", err=e))
        if st.button(tr("back_form")):
            st.session_state.step = 1
            st.rerun()
        return

    doc_ok = st.session_state.form_data.get("document_id") is not None
    fraud = decision["module_outputs"]["fraud_detection"]
    credit = decision["module_outputs"]["credit_risk"]

    placeholders[0].markdown(
        f"✅ **{tr('step_word')} 1:** {tr('step_doc')}" if doc_ok else f"○ **{tr('step_word')} 1:** {tr('no_doc')}"
    )
    time.sleep(0.3)
    fraud_icon = "✅" if fraud["verdict"] == "APPROVE" else ("⚠️" if fraud["verdict"] == "REVIEW" else "🛑")
    placeholders[1].markdown(
        f"{fraud_icon} **{tr('step_word')} 2:** {tr('step_fraud')} — {verdict_label(fraud['verdict'])} "
        f"({tr('score')}: {fraud['fraud_probability']:.2f})"
    )
    time.sleep(0.3)
    placeholders[2].markdown(
        f"✅ **{tr('step_word')} 3:** {tr('step_credit')} — {credit['risk_band']} {tr('risk')} "
        f"({tr('default_prob')}: {credit['default_probability']:.2f})"
    )
    time.sleep(0.3)
    placeholders[3].markdown(
        f"✅ **{tr('step_word')} 4:** {tr('step_final')} — **{verdict_label(decision['final_verdict'])}**")
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
    banner_title = tr(f"banner_{verdict}") if f"banner_{verdict}" in _TX else verdict

    st.markdown(f"""
    <div class="decision-banner {banner_class}">
        <h1>{banner_title}</h1>
        <p>{tr('app_id')}: {decision['application_id']}</p>
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
        c1.metric(tr("m_amount"), f"₹{principal:,.0f}")
        c2.metric(tr("m_rate"), f"{annual_rate:.1f}%")
        c3.metric(tr("m_emi"), f"₹{emi:,.0f}{tr('per_month')}")

    st.markdown(f"##### {tr('explanation')}")
    render_explanation(decision["application_id"], decision["explanation"], current_lang())
    with st.expander(tr("tech_summary")):
        st.write(decision["explanation"])

    if decision.get("shap_top_drivers"):
        st.markdown(f"##### {tr('key_factors')}")
        shap_df = pd.DataFrame(decision["shap_top_drivers"])
        st.bar_chart(shap_df.set_index("feature")["shap_value"])

    if verdict == "MANUAL_REVIEW":
        st.warning(tr("review_time"))

    if decision.get("improvement_suggestions"):
        st.markdown(f"##### {tr('suggestions')}")
        for s in decision["improvement_suggestions"]:
            st.markdown(f"- {s}")

    st.markdown("---")
    render_letter_download(decision["application_id"], current_lang(), key="decision_letter")
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button(tr("go_tracker"), use_container_width=True):
            st.session_state.page = "track"
            st.session_state.step = 1
            st.rerun()
    with cc2:
        if st.button(tr("new_app"), use_container_width=True):
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
            return tr("chat_too_long")
        return tr("chat_unavailable", code=r.status_code)
    except requests.exceptions.RequestException:
        return tr("chat_unreachable")


def render_chat():
    lang = current_lang()
    with st.popover(tr("chat_btn")):
        box = st.container(height=320)
        with st.form("chat_form", clear_on_submit=True):
            msg = st.text_input("Message", max_chars=500, label_visibility="collapsed",
                                placeholder=tr("chat_ph"))
            sent = st.form_submit_button(tr("chat_send"), use_container_width=True)
        if sent and msg.strip():
            st.session_state.chat_history.append(("user", msg.strip()))
            with st.spinner(tr("chat_typing")):
                reply = _send_chat(msg.strip(), lang)
            st.session_state.chat_history.append(("assistant", reply))
        with box:
            st.chat_message("assistant").write(_CHAT_GREETING[lang])
            for role, text in st.session_state.chat_history:
                st.chat_message(role).write(text)
        st.caption(tr("chat_note"))


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