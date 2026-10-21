# ============================================================
# DOCUMENT GENERATOR — Module 5C
# ============================================================
# Builds the customer-facing PDF for a decision:
#   APPROVE        → approval letter + Loan Offer Summary + full EMI schedule
#   DECLINE        → decline letter with the reasons and improvement tips
#   MANUAL_REVIEW  → "under review" notice
#
# WHAT THE LLM WRITES vs WHAT IT NEVER TOUCHES
#   The narrative paragraph (why this outcome, what to do next) comes from
#   llm_explainer.generate_explanation() — Gemini when available, a
#   deterministic template otherwise. Every NUMBER on the page — loan
#   amount, interest rate, EMI, total interest, the amortization table —
#   is computed here by src/loan_math.py. The model is never asked to
#   produce or repeat a figure, so a hallucinated rate or EMI cannot end
#   up in a customer document.
#
# LIMITS (stated plainly)
#   - The Loan Offer is INDICATIVE: it is a demo using an illustrative
#     risk-based pricing curve (loan_math.risk_based_rate), and the PDF
#     says so. It is not a sanction letter.
#   - Amounts are printed as "Rs." — reportlab's built-in fonts have no
#     ₹ glyph (it renders as a black box).
#   - Hindi: reportlab's built-in fonts cannot draw Devanagari, so a
#     Hindi PDF is attempted only if a Devanagari-capable TTF is found
#     (HINDI_FONT_PATH env var, Windows Nirmala UI/Mangal, common Linux
#     Noto/Lohit paths). Without one the letter is produced in English and
#     the result says so (`language_used`). Even with a font, reportlab
#     does not do complex-script shaping by default, so conjuncts/matras
#     in a Hindi PDF may not look right — this path was NOT visually
#     verified in the build sandbox (no Devanagari font available there).
#     The Hindi explanation shown on screen and in chat/SMS is the
#     reliable Hindi path.
# ============================================================

import io
import os
import logging
from datetime import datetime
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from xml.sax.saxutils import escape

from src.llm_explainer import generate_explanation
from src.loan_math import risk_based_rate, calculate_emi, amortization_schedule, inr

logger = logging.getLogger(__name__)

NAVY = colors.HexColor("#0B3D91")
_HINDI_FONT_NAME = "HindiFont"
_HINDI_FONT_CANDIDATES = [
    os.getenv("HINDI_FONT_PATH", ""),
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\mangal.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/truetype/fonts-deva-extra/Sarai.ttf",
]


def _find_hindi_font() -> Optional[str]:
    for path in _HINDI_FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                if _HINDI_FONT_NAME not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(_HINDI_FONT_NAME, path))
                return _HINDI_FONT_NAME
            except Exception as e:
                logger.warning("Could not register Hindi font %s: %s", path, e)
    return None


# ============================================================
# OFFER FIGURES (deterministic — never LLM-generated)
# ============================================================

def compute_offer(decision: Dict, application: Dict) -> Dict:
    principal = float(application.get("loan_amount_requested", 0) or 0)
    months = int(application.get("tenure_months", 36) or 36)
    pd_ = decision["module_outputs"]["credit_risk"]["default_probability"]
    rate = risk_based_rate(pd_)
    emi = calculate_emi(principal, rate, months)
    total_payable = emi * months
    return {
        "principal": principal,
        "tenure_months": months,
        "annual_rate_pct": rate,
        "emi": round(emi, 2),
        "total_payable": round(total_payable, 2),
        "total_interest": round(total_payable - principal, 2),
    }


# ============================================================
# PDF BUILDING
# ============================================================

_STRINGS = {
    "en": {
        "title": {"APPROVE": "Loan Approval Letter", "DECLINE": "Loan Application Outcome",
                  "MANUAL_REVIEW": "Application Under Review"},
        "app_id": "Application ID", "date": "Date", "applicant": "Applicant",
        "decision": "Decision", "confidence": "Decision confidence",
        "suggestions": "What could help next time",
        "offer": "Loan Offer Summary (indicative)", "schedule": "EMI Schedule",
        "amount": "Loan amount", "rate": "Interest rate (p.a.)", "tenure": "Tenure",
        "emi": "Monthly EMI", "interest": "Total interest", "payable": "Total payable",
        "months": "months",
        "cols": ["Month", "Opening", "Interest", "Principal", "EMI", "Closing"],
        "offer_note": ("This offer is indicative and illustrative. It is subject to document "
                       "verification and final sanction, and the pricing shown comes from a demo "
                       "risk-based curve, not a real lender's rate card."),
        "footer": ("Generated by the AI Loan Underwriting Platform — a capstone project demo. "
                   "Not a legally binding document."),
        "verdict": {"APPROVE": "APPROVED", "DECLINE": "DECLINED", "MANUAL_REVIEW": "UNDER REVIEW"},
    },
    "hi": {
        "title": {"APPROVE": "ऋण स्वीकृति पत्र", "DECLINE": "ऋण आवेदन परिणाम",
                  "MANUAL_REVIEW": "आवेदन समीक्षाधीन"},
        "app_id": "आवेदन आईडी", "date": "दिनांक", "applicant": "आवेदक",
        "decision": "निर्णय", "confidence": "निर्णय की विश्वसनीयता",
        "suggestions": "अगली बार क्या मदद कर सकता है",
        "offer": "ऋण प्रस्ताव सारांश (संकेतात्मक)", "schedule": "EMI अनुसूची",
        "amount": "ऋण राशि", "rate": "ब्याज दर (वार्षिक)", "tenure": "अवधि",
        "emi": "मासिक EMI", "interest": "कुल ब्याज", "payable": "कुल देय",
        "months": "महीने",
        "cols": ["माह", "प्रारंभिक", "ब्याज", "मूलधन", "EMI", "शेष"],
        "offer_note": ("यह प्रस्ताव केवल संकेतात्मक है और दस्तावेज़ सत्यापन तथा अंतिम स्वीकृति के अधीन है।"),
        "footer": ("AI Loan Underwriting Platform द्वारा तैयार — कैपस्टोन प्रोजेक्ट डेमो। कानूनी रूप से बाध्यकारी दस्तावेज़ नहीं।"),
        "verdict": {"APPROVE": "स्वीकृत", "DECLINE": "अस्वीकृत", "MANUAL_REVIEW": "समीक्षाधीन"},
    },
}


def generate_decision_letter(
    decision: Dict,
    application: Optional[Dict] = None,
    language: str = "en",
) -> Dict:
    """
    Returns {"pdf": bytes, "language_used": "en"|"hi", "text_source": "gemini"|"template",
             "offer": dict|None}.
    """
    application = application or {}
    verdict = decision["final_verdict"]

    language_used = "en"
    font_regular, font_bold = "Helvetica", "Helvetica-Bold"
    if language == "hi":
        hindi_font = _find_hindi_font()
        if hindi_font:
            language_used, font_regular, font_bold = "hi", hindi_font, hindi_font
        else:
            logger.warning("Hindi PDF requested but no Devanagari font found — producing English PDF")

    s = _STRINGS[language_used]
    expl = generate_explanation(decision, application, language_used)

    offer = compute_offer(decision, application) if verdict == "APPROVE" and application.get("loan_amount_requested") else None

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=font_bold, textColor=NAVY, fontSize=18)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font_bold, textColor=NAVY, fontSize=13)
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=font_regular, fontSize=10.5, leading=15)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey, leading=11)

    verdict_color = {"APPROVE": "#1F8A4C", "MANUAL_REVIEW": "#C77700", "DECLINE": "#C0392B"}[verdict]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title=s["title"][verdict], author="AI Loan Underwriting Platform")

    name = application.get("full_name") or "Applicant"
    story = [
        Paragraph("AI Loan Underwriting Platform", h1),
        Paragraph(escape(s["title"][verdict]), h2),
        Spacer(1, 6),
        Paragraph(f"{s['app_id']}: <b>{escape(str(decision['application_id']))}</b>", body),
        Paragraph(f"{s['date']}: {datetime.now().strftime('%d %b %Y')}", body),
        Paragraph(f"{s['applicant']}: {escape(str(name))}", body),
        Spacer(1, 10),
        Paragraph(f"{s['decision']}: <font color='{verdict_color}'><b>{s['verdict'][verdict]}</b></font>", body),
        Paragraph(f"{s['confidence']}: {decision.get('confidence_score', 0):.0%}", body),
        Spacer(1, 10),
        Paragraph(escape(expl["text"]), body),
    ]

    if verdict != "APPROVE" and decision.get("improvement_suggestions"):
        story += [Spacer(1, 10), Paragraph(f"<b>{s['suggestions']}</b>", body)]
        for tip in decision["improvement_suggestions"]:
            story.append(Paragraph("• " + escape(tip), body))

    if offer:
        story += [PageBreak(), Paragraph(escape(s["offer"]), h2), Spacer(1, 6)]
        summary = [
            [s["amount"], inr(offer["principal"])],
            [s["rate"], f"{offer['annual_rate_pct']:.2f}%"],
            [s["tenure"], f"{offer['tenure_months']} {s['months']}"],
            [s["emi"], inr(offer["emi"], 2)],
            [s["interest"], inr(offer["total_interest"], 2)],
            [s["payable"], inr(offer["total_payable"], 2)],
        ]
        t = Table(summary, colWidths=[70 * mm, 60 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
            ("FONTSIZE", (0, 0), (-1, -1), 10.5),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3FC")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))
        story += [t, Spacer(1, 6), Paragraph(escape(s["offer_note"]), small), Spacer(1, 12),
                  Paragraph(escape(s["schedule"]), h2), Spacer(1, 4)]

        rows = [s["cols"]] + [
            [r["month"], f"{r['opening']:,.0f}", f"{r['interest']:,.0f}",
             f"{r['principal']:,.0f}", f"{r['emi']:,.0f}", f"{r['closing']:,.0f}"]
            for r in amortization_schedule(offer["principal"], offer["annual_rate_pct"], offer["tenure_months"])
        ]
        sched = Table(rows, repeatRows=1, colWidths=[16 * mm] + [28 * mm] * 5)
        sched.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6FA")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ]))
        story.append(sched)

    story += [Spacer(1, 16), Paragraph(escape(s["footer"]), small)]
    doc.build(story)

    return {"pdf": buf.getvalue(), "language_used": language_used,
            "text_source": expl["source"], "offer": offer}
