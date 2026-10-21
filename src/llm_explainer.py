# ============================================================
# LLM EXPLAINER — Module 5B
# ============================================================
# Turns a Decision Engine result (verdict + SHAP drivers + suggestions)
# into a short, empathetic, plain-language explanation, in English or
# Hindi, for the User Portal and for email/SMS notifications.
#
# TWO PATHS, ONE INTERFACE:
#   1. Gemini path  — used when GEMINI_API_KEY (or GOOGLE_API_KEY) is set
#      and the call succeeds.
#   2. Template path — deterministic, offline, used when there is no API
#      key, the call fails/times out/is rate-limited, or the model's reply
#      fails the safety check below. The platform therefore never shows a
#      blank or broken explanation because an external API was down.
#
# SAFETY / HONESTY RULES (enforced in code, not just in the prompt):
#   - The LLM is only a WRITER. It is told the verdict and may not change
#     it; the reply must open with a verdict tag ([APPROVED] / [UNDER
#     REVIEW] / [DECLINED]) matching the real verdict or it is discarded
#     and the template is used instead.
#   - Only a minimal fact sheet is sent to the API: first name, verdict,
#     loan amount/tenure/purpose, and the top driver *labels* with a
#     risk-up/risk-down direction. Raw income, bureau numbers, account
#     numbers and the full name are never sent.
#
# NOTE ON TESTING: the Gemini request/response shapes below follow
# Google's public REST API for `generateContent`, but this module was
# built in a sandbox with no route to generativelanguage.googleapis.com,
# so the live call has NOT been exercised end-to-end. Everything else
# (fact-sheet building, tag validation, templates, fallback behaviour) is
# unit-tested with the HTTP layer mocked — see tests/test_module5.py.
# ============================================================

import os
import re
import logging
from typing import Dict, List, Optional, Any

import requests

logger = logging.getLogger(__name__)

# Free-tier-eligible Flash model at the time of writing. Model names change
# often — override with GEMINI_MODEL=<name> instead of editing code.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

VERDICT_TAGS = {
    "APPROVE": "[APPROVED]",
    "MANUAL_REVIEW": "[UNDER REVIEW]",
    "DECLINE": "[DECLINED]",
}

# ── Friendly feature labels (English / Hindi) ─────────────────────
FEATURE_LABELS = {
    "revolving_utilization":      ("credit card / credit line utilization", "क्रेडिट कार्ड/क्रेडिट लाइन का उपयोग"),
    "debt_ratio":                 ("monthly debt compared to income", "आय की तुलना में मासिक कर्ज़"),
    "monthly_income":             ("monthly income", "मासिक आय"),
    "age":                        ("age", "आयु"),
    "open_credit_lines":          ("number of open credit lines", "खुली क्रेडिट लाइनों की संख्या"),
    "real_estate_loans":          ("number of property loans", "प्रॉपर्टी लोन की संख्या"),
    "dependents":                 ("number of dependents", "आश्रितों की संख्या"),
    "times_30_59_days_late":      ("record of payments 30-59 days late", "30-59 दिन देरी से भुगतान का रिकॉर्ड"),
    "times_60_89_days_late":      ("record of payments 60-89 days late", "60-89 दिन देरी से भुगतान का रिकॉर्ड"),
    "times_90_days_late":         ("record of payments 90+ days late", "90+ दिन देरी से भुगतान का रिकॉर्ड"),
    "total_late_incidents":       ("overall late-payment record", "देरी से भुगतान का कुल रिकॉर्ड"),
    "has_delinquency_history":    ("late-payment history", "देरी से भुगतान का इतिहास"),
    "has_severe_delinquency":     ("serious (90+ day) delinquency record", "गंभीर (90+ दिन) चूक का रिकॉर्ड"),
    "income_per_dependent":       ("income per dependent", "प्रति आश्रित आय"),
    "credit_lines_per_dependent": ("credit lines per dependent", "प्रति आश्रित क्रेडिट लाइनें"),
    "high_utilization_flag":      ("credit usage above the limit", "सीमा से अधिक क्रेडिट उपयोग"),
    "high_debt_ratio_flag":       ("debt above income", "आय से अधिक कर्ज़"),
    "has_real_estate_loan":       ("having a property loan", "प्रॉपर्टी लोन होना"),
    "debt_ratio_x_utilization":   ("combined debt and credit usage", "कर्ज़ और क्रेडिट उपयोग का संयुक्त प्रभाव"),
    "age_bin":                    ("age group", "आयु वर्ग"),
    "near_retirement":            ("nearing retirement age", "सेवानिवृत्ति की आयु के निकट"),
    "monthly_income_was_missing": ("income not reported", "आय की जानकारी उपलब्ध नहीं"),
    "dependents_was_missing":     ("dependents not reported", "आश्रितों की जानकारी उपलब्ध नहीं"),
    "late_count_was_sentinel":    ("unreliable late-payment record", "देरी से भुगतान का अविश्वसनीय रिकॉर्ड"),
}

# Hindi renderings of the engine's English improvement suggestions.
SUGGESTION_HI = {
    "Reduce existing EMIs before reapplying": "दोबारा आवेदन से पहले अपनी मौजूदा EMI कम करें",
    "Maintain 6 months of stable salary credits before reapplying": "दोबारा आवेदन से पहले 6 महीने तक नियमित वेतन जमा बनाए रखें",
    "Wait 6 months to improve credit history, then reapply": "क्रेडिट इतिहास सुधारने के लिए 6 महीने प्रतीक्षा करें, फिर आवेदन करें",
    "Apply with a co-applicant or after an income increase": "सह-आवेदक के साथ या आय बढ़ने के बाद आवेदन करें",
    "Reduce existing debt so monthly obligations stay well below your income":
        "मौजूदा कर्ज़ घटाएँ ताकि मासिक किस्तें आपकी आय से काफ़ी कम रहें",
    "Clear any overdue payments and keep every payment on time for 12-24 months before reapplying":
        "बकाया भुगतान चुकाएँ और दोबारा आवेदन से पहले 12-24 महीने तक हर भुगतान समय पर करें",
    "Apply with a co-applicant so household income is shared across fewer dependents":
        "सह-आवेदक के साथ आवेदन करें ताकि घर की आय का बोझ बँट सके",
    "Provide additional supporting documents if you believe this decision is incorrect":
        "यदि आपको लगता है कि निर्णय गलत है तो अतिरिक्त दस्तावेज़ जमा करें",
}


# ============================================================
# LOW-LEVEL GEMINI CLIENT (shared with chatbot.py / doc_generator.py)
# ============================================================

def get_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None


def gemini_available() -> bool:
    return get_api_key() is not None


def call_gemini(
    contents: List[Dict[str, Any]],
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.4,
    max_output_tokens: int = 800,
    timeout: float = 25.0,
) -> Optional[Dict[str, Any]]:
    """
    POSTs one generateContent request. Returns the parsed JSON response, or
    None on ANY failure (no key, network error, timeout, HTTP error incl.
    429 rate limit, non-JSON body). Never raises — callers fall back.
    The API key is sent in a header and is never logged.
    """
    key = get_api_key()
    if not key:
        return None

    body: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools

    try:
        resp = requests.post(
            GEMINI_ENDPOINT.format(model=GEMINI_MODEL),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning("Gemini HTTP %s: %s", resp.status_code, resp.text[:200])
            return None
        return resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning("Gemini call failed: %s", e)
        return None


def extract_text(response: Optional[Dict[str, Any]]) -> Optional[str]:
    """Concatenates the text parts of the first candidate; None if absent/blocked."""
    try:
        parts = response["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except (KeyError, IndexError, TypeError):
        return None


def generate_text(prompt: str, system: Optional[str] = None, **kwargs) -> Optional[str]:
    resp = call_gemini([{"role": "user", "parts": [{"text": prompt}]}], system=system, **kwargs)
    return extract_text(resp)


# ============================================================
# FACT SHEET (the ONLY thing sent to the LLM)
# ============================================================

def _label(feature: str, lang: str) -> str:
    en, hi = FEATURE_LABELS.get(feature, (feature.replace("_", " "), feature.replace("_", " ")))
    return hi if lang == "hi" else en


def build_fact_sheet(decision: Dict, application: Optional[Dict] = None, language: str = "en") -> Dict:
    app = application or {}
    first_name = (app.get("full_name") or "").strip().split(" ")[0] or None

    raising, lowering = [], []
    for d in decision.get("shap_top_drivers", []) or []:
        label = _label(d["feature"], language)
        (raising if d["shap_value"] > 0 else lowering).append(label)

    return {
        "first_name": first_name,
        "verdict": decision["final_verdict"],
        "engine_explanation": decision.get("explanation", ""),
        "factors_increasing_risk": raising,
        "factors_reducing_risk": lowering,
        "improvement_suggestions": decision.get("improvement_suggestions", []),
        "loan_amount": app.get("loan_amount_requested"),
        "tenure_months": app.get("tenure_months"),
        "loan_purpose": app.get("loan_purpose"),
    }


# ============================================================
# TEMPLATE FALLBACK
# ============================================================

def _template_explanation(facts: Dict, language: str) -> str:
    verdict = facts["verdict"]
    name = facts["first_name"]
    up, down, tips = facts["factors_increasing_risk"], facts["factors_reducing_risk"], facts["improvement_suggestions"]

    if language == "hi":
        hello = f"{name} जी, " if name else ""
        opening = {
            "APPROVE": "आपका ऋण आवेदन स्वीकृत कर दिया गया है।",
            "MANUAL_REVIEW": "आपका आवेदन समीक्षा के लिए हमारी अंडरराइटिंग टीम को भेजा गया है। निर्णय सामान्यतः 2-3 कार्य-दिवसों में बताया जाता है।",
            "DECLINE": "खेद है, हम इस समय आपका ऋण आवेदन स्वीकृत नहीं कर सके।",
        }[verdict]
        lines = [hello + opening]
        if up:
            lines.append("जोखिम बढ़ाने वाले मुख्य कारक: " + "; ".join(up) + "।")
        if down:
            lines.append("आपके पक्ष में रहे कारक: " + "; ".join(down) + "।")
        if not up and not down:
            lines.append("विस्तृत कारण आपके आवेदन के विवरण में उपलब्ध हैं।")
        if tips and verdict != "APPROVE":
            lines.append("आगे के लिए सुझाव: " + "; ".join(SUGGESTION_HI.get(t, t) for t in tips) + "।")
        return " ".join(lines)

    hello = f"Hi {name}, " if name else ""
    opening = {
        "APPROVE": "good news — your loan application has been approved.",
        "MANUAL_REVIEW": "your application has been sent to our underwriting team for a closer look. You can usually expect an update within 2-3 business days.",
        "DECLINE": "thank you for applying. We're sorry — we're unable to approve this application right now.",
    }[verdict]
    lines = [(hello + opening) if hello else opening[0].upper() + opening[1:]]
    if up:
        lines.append("The main factors that weighed against the application: " + "; ".join(up) + ".")
    if down:
        lines.append("Factors working in your favour: " + "; ".join(down) + ".")
    if not up and not down:
        lines.append(facts["engine_explanation"] or "Full details are available in your application summary.")
    if tips and verdict != "APPROVE":
        lines.append("What could help next time: " + "; ".join(tips) + ".")
    return " ".join(lines)


# ============================================================
# PROMPT + VALIDATION
# ============================================================

_SYSTEM_PROMPT = (
    "You write short, warm, professional loan-decision explanations for bank customers. "
    "You are a WRITER, not the decision-maker: the verdict is already final and you must never "
    "change, soften, or contradict it. Use ONLY the facts provided. Do not invent numbers, rates, "
    "dates, fees, or reasons that are not in the facts. Do not promise future approval. "
    "No legal or financial advice. Maximum 120 words, plain text, no markdown, no bullet symbols."
)


def _build_prompt(facts: Dict, language: str) -> str:
    lang_name = "Hindi (Devanagari script)" if language == "hi" else "English"
    tag = VERDICT_TAGS[facts["verdict"]]
    return (
        f"Write the explanation in {lang_name}.\n"
        f"The VERY FIRST characters of your reply must be the exact tag {tag} (in English, even when "
        f"writing Hindi), followed by a space and then the explanation.\n"
        f"Tone: empathetic and actionable"
        + (" — congratulate the customer briefly.\n" if facts["verdict"] == "APPROVE"
           else " — be kind, explain the reasons clearly, then give the improvement tips.\n")
        + f"FACTS (JSON): {facts}"
    )


def _validate_and_strip(text: str, verdict: str) -> Optional[str]:
    """Reply must open with the correct verdict tag; returns text without it, else None."""
    expected = VERDICT_TAGS[verdict]
    stripped = text.strip()
    if not stripped.upper().startswith(expected):
        return None
    for other_verdict, other_tag in VERDICT_TAGS.items():
        if other_verdict != verdict and other_tag in stripped.upper():
            return None
    # case-insensitive: the validation above upper-cases, so "[under review]" is accepted
    # and must also be stripped, or the raw tag would leak into the customer-facing text.
    body = re.sub(r"^\s*\[[A-Za-z ]+\]\s*", "", stripped, count=1).strip()
    return body or None


# ============================================================
# PUBLIC API
# ============================================================

def generate_explanation(decision: Dict, application: Optional[Dict] = None, language: str = "en") -> Dict:
    """
    Returns {"text": str, "source": "gemini" | "template", "language": "en"|"hi"}.
    Never raises and never returns empty text.
    """
    language = "hi" if language == "hi" else "en"
    facts = build_fact_sheet(decision, application, language)

    if gemini_available():
        raw = generate_text(_build_prompt(facts, language), system=_SYSTEM_PROMPT)
        if raw:
            body = _validate_and_strip(raw, decision["final_verdict"])
            if body:
                return {"text": body, "source": "gemini", "language": language}
            logger.warning("Gemini reply failed verdict-tag validation — using template instead")

    return {"text": _template_explanation(facts, language), "source": "template", "language": language}
