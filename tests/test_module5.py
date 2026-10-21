# ============================================================
# PYTEST — Module 5 (RAG, loan math, LLM explainer, letter PDF,
#                    chatbot, audit-trail storage, API endpoints)
# ============================================================
# ALL OFFLINE. The Gemini HTTP layer (`requests.post` inside
# src.llm_explainer) is always mocked, so these tests never touch the
# network and never need a real GEMINI_API_KEY.
#
# What these tests can NOT tell you (see README "Known Limitations"):
#   - that the live Gemini API accepts our request shapes,
#   - that a Hindi PDF looks right (no Devanagari font in CI),
#   - real-model behaviour: the API tests use a stub model that returns a
#     fixed probability, because training needs the full dataset.
# ============================================================

import io
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import requests
from unittest import mock

from src import rag_engine, loan_math, llm_explainer, doc_generator, chatbot, decision_engine
from src.rag_engine import chunk_text, KnowledgeBase, ingest_directory, get_knowledge_base, stem_analyzer
from src.loan_math import calculate_emi, amortization_schedule, inr, risk_based_rate
from src.llm_explainer import (
    build_fact_sheet, generate_explanation, call_gemini, _validate_and_strip, VERDICT_TAGS,
)
from src.doc_generator import generate_decision_letter, compute_offer
from src.chatbot import (
    LoanAssistantChatbot, eligibility_checker, emi_calculator, application_status,
    faq_retriever, run_tool, MAX_TOOL_ROUNDS, _TEXT,
)
from src.decision_engine import make_final_decision, load_audit_trail

KB_DIR = os.path.join(PROJECT_ROOT, "data", "knowledge_base")


# ============================================================
# SHARED FIXTURES / HELPERS
# ============================================================

@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    """No API key by default; audit trail goes to a temp dir; chatbot KB cache reset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(decision_engine, "AUDIT_TRAIL_DIR", str(tmp_path))
    chatbot.reset_kb_cache()
    yield
    chatbot.reset_kb_cache()


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")


def _credit(verdict="APPROVE", prob=0.08, band="LOW", rule=None):
    return {"module": "credit_risk", "default_probability": prob, "risk_band": band,
            "verdict": verdict, "rule_triggered": rule}


def _fraud(verdict="APPROVE", prob=0.01):
    return {"module": "fraud_detection", "fraud_probability": prob, "fraud_band": "LOW",
            "verdict": verdict, "rule_triggered": None}


APPLICATION = {"full_name": "Ravi Kumar Sharma", "loan_amount_requested": 500000.0,
               "tenure_months": 36, "loan_purpose": "Personal"}


def _decision(verdict="APPROVE", app_id="APP1A2B3C4D5E", with_application=True, **credit_kw):
    credit = _credit(verdict if verdict != "MANUAL_REVIEW" else "MANUAL_REVIEW", **credit_kw)
    shap = [{"feature": "revolving_utilization", "shap_value": 0.4},
            {"feature": "monthly_income", "shap_value": -0.3}]
    return make_final_decision(app_id, credit, _fraud(), shap_pairs=shap,
                               application=APPLICATION if with_application else None)


def _gemini_response(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _http(status=200, payload=None):
    r = mock.Mock()
    r.status_code = status
    r.text = json.dumps(payload) if payload is not None else "err"
    r.json.return_value = payload
    return r


def _pdf_text(pdf_bytes):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [p.get_text() for p in doc]
    doc.close()
    return pages


# ============================================================
# 1. CHUNKER
# ============================================================

class TestChunker:

    def test_heading_attaches_to_following_block(self):
        text = "4. DELINQUENCY HISTORY POLICY\n\n4.1 Three or more late payments cause a decline."
        chunks = chunk_text(text, "p.txt")
        assert len(chunks) == 1
        assert chunks[0]["text"].startswith("4. DELINQUENCY HISTORY POLICY")
        assert "Three or more late payments" in chunks[0]["text"]
        assert chunks[0]["source"] == "p.txt"

    def test_qa_pairs_are_atomic(self):
        text = "Q1: What is A? A1: It is alpha.\n\nQ2: What is B? A2: It is beta."
        chunks = chunk_text(text, "f.txt")
        assert len(chunks) == 2                      # not packed together although tiny
        assert "alpha" in chunks[0]["text"] and "beta" not in chunks[0]["text"]
        assert "beta" in chunks[1]["text"] and "alpha" not in chunks[1]["text"]

    def test_long_section_splits_on_sentences_and_keeps_heading(self):
        sentences = [f"Rule number {i} says that applicants must satisfy condition {i} in full." for i in range(12)]
        text = "7. PRICING POLICY\n\n" + " ".join(sentences)
        chunks = chunk_text(text, "p.txt", chunk_size=200)
        assert len(chunks) > 1
        for c in chunks:
            assert c["text"].startswith("7. PRICING POLICY")
            assert c["text"].endswith(".")                     # never cut mid-sentence
            assert len(c["text"]) <= 200

    def test_single_oversized_sentence_is_hard_split_with_overlap(self):
        text = "word " * 200                                    # one 1000-char "sentence", no punctuation
        chunks = chunk_text(text.strip() + ".", "p.txt", chunk_size=100, overlap=20)
        assert len(chunks) > 5
        assert all(len(c["text"]) <= 100 for c in chunks)

    def test_empty_text_gives_no_chunks(self):
        assert chunk_text("", "x.txt") == []
        assert chunk_text("\n\n   \n", "x.txt") == []

    def test_small_blocks_are_packed_together(self):
        text = "First short rule applies here.\n\nSecond short rule applies there."
        assert len(chunk_text(text, "x.txt")) == 1


# ============================================================
# 2. RETRIEVAL (TF-IDF)
# ============================================================

@pytest.fixture(scope="module")
def kb():
    return ingest_directory(KB_DIR)


# (question, substring that must appear in a retrieved chunk, how many top results to look at)
RETRIEVAL_CASES = [
    ("How is the EMI calculated?", "reducing-balance", 1),
    ("What documents do I need to apply?", "photo ID", 1),
    ("What does a CIBIL score mean?", "CIBIL", 1),
    ("Can I prepay my loan early?", "prepay", 1),
    ("How do I file a grievance or complaint?", "Grievance", 1),
    ("How many dependents trigger a manual review?", "6 or more dependents", 3),
    ("What happens if I have 3 serious delinquencies?", "Three or more instances of 90+ day", 3),
    ("What is the minimum income required?", "5,000", 3),
]


class TestRetrieval:

    @pytest.mark.parametrize("question,needle,top_k", RETRIEVAL_CASES)
    def test_known_question_retrieves_right_passage(self, kb, question, needle, top_k):
        hits = kb.query(question, top_k=top_k)
        assert any(needle in h.text for h in hits), (
            f"{question!r}: expected {needle!r} in top-{top_k}, got {[h.text[:60] for h in hits]}")

    def test_overall_hit_rate(self, kb):
        ok = sum(any(n in h.text for h in kb.query(q, top_k=k)) for q, n, k in RETRIEVAL_CASES)
        assert ok / len(RETRIEVAL_CASES) >= 0.85

    def test_stemmer_matches_plural_forms(self):
        assert "delinquency" in stem_analyzer("delinquencies")
        assert stem_analyzer("7 dependents") == stem_analyzer("dependents")   # pure digits dropped

    def test_save_and_load_roundtrip(self, kb, tmp_path):
        path = str(tmp_path / "idx.joblib")
        kb.save(path)
        loaded = KnowledgeBase.load(path)
        a = kb.query("prepayment penalty", top_k=2)
        b = loaded.query("prepayment penalty", top_k=2)
        assert [x.text for x in a] == [x.text for x in b]

    def test_missing_index_returns_none(self, tmp_path):
        assert get_knowledge_base(str(tmp_path / "nope.joblib")) is None

    def test_unfitted_query_and_empty_fit_raise(self):
        with pytest.raises(RuntimeError):
            KnowledgeBase().query("anything")
        with pytest.raises(ValueError):
            KnowledgeBase().fit([])

    def test_gibberish_returns_nothing(self, kb):
        assert kb.query("zzzqqq xylophone", top_k=3) == []


# ============================================================
# 3. LOAN MATH
# ============================================================

class TestLoanMath:

    def test_known_emi_value(self):
        assert round(calculate_emi(500000, 12, 36), 2) == 16607.15

    def test_zero_rate_emi_is_straight_division(self):
        assert calculate_emi(120000, 0, 12) == pytest.approx(10000)

    def test_invalid_tenure_raises(self):
        with pytest.raises(ValueError):
            calculate_emi(1000, 10, 0)

    def test_schedule_closes_to_zero_and_principal_adds_up(self):
        rows = amortization_schedule(500000, 12, 36)
        assert len(rows) == 36
        assert rows[-1]["closing"] == 0
        assert sum(r["principal"] for r in rows) == pytest.approx(500000, abs=1.0)
        assert rows[0]["opening"] == 500000
        assert rows[0]["interest"] == pytest.approx(5000, abs=0.01)

    @pytest.mark.parametrize("amount,decimals,expected", [
        (500000, 0, "Rs. 5,00,000"),
        (999, 0, "Rs. 999"),
        (1234567.5, 2, "Rs. 12,34,567.50"),
        (16607.15, 2, "Rs. 16,607.15"),
        (10000000, 0, "Rs. 1,00,00,000"),
        (-2500, 0, "-Rs. 2,500"),
    ])
    def test_inr_indian_grouping(self, amount, decimals, expected):
        assert inr(amount, decimals) == expected

    def test_risk_based_rate_bounds(self):
        assert risk_based_rate(0.0) == 10.5
        assert risk_based_rate(1.0) == 25.5
        assert risk_based_rate(5.0) == 25.5       # clipped
        assert risk_based_rate(-1) == 10.5


# ============================================================
# 4. LLM EXPLAINER (HTTP mocked)
# ============================================================

class TestFactSheet:

    def test_no_full_name_or_raw_income_in_fact_sheet(self):
        decision = _decision("DECLINE", prob=0.8, band="HIGH")
        app = dict(APPLICATION, monthly_income=45123.0, existing_emis=7777.0)
        facts = build_fact_sheet(decision, app)
        blob = json.dumps(facts)
        assert facts["first_name"] == "Ravi"
        assert "Sharma" not in blob and "Kumar" not in blob
        assert "45123" not in blob and "7777" not in blob

    def test_payload_actually_sent_to_gemini_has_no_pii(self, gemini_key):
        decision = _decision("APPROVE")
        app = dict(APPLICATION, monthly_income=45123.0)
        with mock.patch("src.llm_explainer.requests.post",
                        return_value=_http(200, _gemini_response("[APPROVED] Congratulations Ravi."))) as post:
            generate_explanation(decision, app, "en")
        sent = json.dumps(post.call_args.kwargs["json"])
        assert "Sharma" not in sent and "45123" not in sent
        # key travels in a header, never in the URL or body
        assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "test-key-123"
        assert "test-key-123" not in post.call_args.args[0]
        assert "test-key-123" not in sent

    def test_factors_split_by_shap_sign(self):
        facts = build_fact_sheet(_decision("DECLINE", prob=0.8, band="HIGH"), APPLICATION)
        assert "credit card / credit line utilization" in facts["factors_increasing_risk"]
        assert "monthly income" in facts["factors_reducing_risk"]


class TestGeminiExplanation:

    def test_correct_tag_is_accepted_and_stripped(self, gemini_key):
        with mock.patch("src.llm_explainer.requests.post",
                        return_value=_http(200, _gemini_response("[APPROVED] Great news, Ravi."))):
            out = generate_explanation(_decision("APPROVE"), APPLICATION)
        assert out["source"] == "gemini"
        assert out["text"] == "Great news, Ravi."

    @pytest.mark.parametrize("reply", [
        "[DECLINED] We are sorry.",                       # wrong verdict tag
        "Congratulations, you are approved!",             # tag missing
        "[APPROVED] Yes, but [DECLINED] maybe not.",      # contradicting second tag
    ])
    def test_bad_or_missing_tag_falls_back_to_template(self, gemini_key, reply):
        with mock.patch("src.llm_explainer.requests.post", return_value=_http(200, _gemini_response(reply))):
            out = generate_explanation(_decision("APPROVE"), APPLICATION)
        assert out["source"] == "template"
        assert "approved" in out["text"].lower()

    def test_no_key_uses_template_without_any_http_call(self):
        with mock.patch("src.llm_explainer.requests.post") as post:
            out = generate_explanation(_decision("DECLINE", prob=0.9, band="HIGH"), APPLICATION)
        post.assert_not_called()
        assert out["source"] == "template" and out["text"]

    @pytest.mark.parametrize("failure", [
        {"return_value": _http(429, {"error": "rate limited"})},
        {"return_value": _http(500, None)},
        {"side_effect": requests.exceptions.Timeout()},
        {"side_effect": requests.exceptions.ConnectionError()},
        {"return_value": mock.Mock(status_code=200, text="<html>", json=mock.Mock(side_effect=ValueError()))},
    ])
    def test_http_failures_fall_back_to_template(self, gemini_key, failure):
        with mock.patch("src.llm_explainer.requests.post", **failure):
            out = generate_explanation(_decision("MANUAL_REVIEW"), APPLICATION)
        assert out["source"] == "template"
        assert out["text"]

    def test_empty_candidates_fall_back(self, gemini_key):
        with mock.patch("src.llm_explainer.requests.post", return_value=_http(200, {"candidates": []})):
            assert generate_explanation(_decision("APPROVE"), APPLICATION)["source"] == "template"

    def test_hindi_template(self):
        out = generate_explanation(_decision("DECLINE", prob=0.9, band="HIGH"), APPLICATION, "hi")
        assert out["language"] == "hi" and out["source"] == "template"
        assert any("ऀ" <= ch <= "ॿ" for ch in out["text"])
        assert "Ravi जी" in out["text"]

    def test_unknown_language_defaults_to_english(self):
        assert generate_explanation(_decision("APPROVE"), APPLICATION, "fr")["language"] == "en"

    def test_validate_and_strip_helper(self):
        assert _validate_and_strip("[under review] Hold on.", "MANUAL_REVIEW") == "Hold on."
        assert _validate_and_strip("[APPROVED]", "APPROVE") is None       # tag only, no body
        assert set(VERDICT_TAGS) == {"APPROVE", "MANUAL_REVIEW", "DECLINE"}

    def test_call_gemini_never_raises_and_returns_none_without_key(self):
        assert call_gemini([{"role": "user", "parts": [{"text": "hi"}]}]) is None


# ============================================================
# 5. DECISION LETTER PDF
# ============================================================

class TestDecisionLetter:

    def test_approve_has_offer_and_schedule(self):
        out = generate_decision_letter(_decision("APPROVE"), APPLICATION)
        assert out["pdf"].startswith(b"%PDF")
        assert out["offer"] is not None and out["language_used"] == "en"
        pages = _pdf_text(out["pdf"])
        assert len(pages) == 3, f"expected letter + offer page + schedule overflow, got {len(pages)} pages"
        joined = "\n".join(pages)
        assert "Loan Offer Summary" in joined and "EMI Schedule" in joined
        assert "Rs. 5,00,000" in joined

    def test_offer_numbers_are_deterministic_and_consistent(self):
        offer = compute_offer(_decision("APPROVE", prob=0.08), APPLICATION)
        assert offer["annual_rate_pct"] == risk_based_rate(0.08)
        assert offer["total_payable"] == pytest.approx(offer["emi"] * 36, abs=1)
        assert offer["total_interest"] == pytest.approx(offer["total_payable"] - 500000, abs=1)

    def test_decline_has_no_offer(self):
        out = generate_decision_letter(_decision("DECLINE", prob=0.9, band="HIGH"), APPLICATION)
        assert out["offer"] is None
        joined = "\n".join(_pdf_text(out["pdf"]))
        assert "Loan Offer Summary" not in joined and "EMI Schedule" not in joined
        assert "DECLINED" in joined

    def test_manual_review_has_no_offer(self):
        out = generate_decision_letter(_decision("MANUAL_REVIEW", prob=0.45, band="MEDIUM"), APPLICATION)
        assert out["offer"] is None
        assert "UNDER REVIEW" in "\n".join(_pdf_text(out["pdf"]))

    def test_approve_without_application_still_builds_pdf_without_offer(self):
        out = generate_decision_letter(_decision("APPROVE", with_application=False), None)
        assert out["pdf"].startswith(b"%PDF") and out["offer"] is None

    def test_hindi_without_font_falls_back_to_english_and_says_so(self, monkeypatch):
        monkeypatch.setattr(doc_generator, "_find_hindi_font", lambda: None)
        out = generate_decision_letter(_decision("DECLINE", prob=0.9, band="HIGH"), APPLICATION, language="hi")
        assert out["language_used"] == "en"
        assert out["pdf"].startswith(b"%PDF")

    def test_hindi_with_a_font_uses_hindi_path(self, monkeypatch):
        # Logic test only: Helvetica stands in for a Devanagari TTF so no font file is needed.
        # It does NOT verify Hindi glyph rendering — that has to be checked by eye.
        monkeypatch.setattr(doc_generator, "_find_hindi_font", lambda: "Helvetica")
        out = generate_decision_letter(_decision("DECLINE", prob=0.9, band="HIGH"), APPLICATION, language="hi")
        assert out["language_used"] == "hi"

    def test_markup_in_name_cannot_break_pdf(self):
        app = dict(APPLICATION, full_name="<b>Evil & <i>Name")
        out = generate_decision_letter(_decision("DECLINE", prob=0.9, band="HIGH"), app)
        assert out["pdf"].startswith(b"%PDF")


# ============================================================
# 6. AUDIT TRAIL — application summary + ID safety
# ============================================================

class TestAuditTrail:

    def test_application_summary_stores_only_four_fields(self):
        app = dict(APPLICATION, monthly_income=45123.0, existing_emis=999.0, times_90_days_late=1,
                   document_id="DOC1", age=30)
        res = make_final_decision("APPSTORE0001", _credit(), _fraud(), application=app)
        stored = load_audit_trail("APPSTORE0001")
        assert stored["application"] == APPLICATION
        assert set(res["application"]) == {"full_name", "loan_amount_requested", "tenure_months", "loan_purpose"}
        blob = json.dumps(stored["application"])
        assert "45123" not in blob and "999" not in blob

    def test_no_application_means_no_key(self):
        make_final_decision("APPSTORE0002", _credit(), _fraud())
        assert "application" not in load_audit_trail("APPSTORE0002")

    def test_none_values_are_skipped(self):
        make_final_decision("APPSTORE0003", _credit(), _fraud(),
                            application={"full_name": None, "loan_amount_requested": 1000.0})
        assert load_audit_trail("APPSTORE0003")["application"] == {"loan_amount_requested": 1000.0}

    @pytest.mark.parametrize("bad_id", ["..\\x", "../x", "a/b", "a" * 65, "", "a b", "x.json", "..\\..\\secret"])
    def test_load_audit_trail_rejects_unsafe_ids(self, bad_id):
        assert load_audit_trail(bad_id) is None

    def test_load_audit_trail_rejects_non_string(self):
        assert load_audit_trail(None) is None
        assert load_audit_trail(123) is None

    def test_valid_id_roundtrips(self):
        make_final_decision("APPSTORE0004", _credit(), _fraud())
        assert load_audit_trail("APPSTORE0004")["application_id"] == "APPSTORE0004"


# ============================================================
# 7. CHATBOT TOOLS
# ============================================================

class TestChatbotTools:

    def test_emi_tool_matches_loan_math(self):
        r = emi_calculator(500000, 36, 12)
        assert r["monthly_emi"] == 16607.15 and r["rate_was_assumed"] is False

    def test_emi_tool_discloses_assumed_rate(self):
        r = emi_calculator(500000, 36)
        assert r["rate_was_assumed"] is True and r["annual_rate_pct"] == 12.0

    @pytest.mark.parametrize("args", [(0, 36), (1000, 0), (1000, 12, 80), (1000, 12, -1)])
    def test_emi_tool_rejects_bad_input(self, args):
        assert "error" in emi_calculator(*args)

    def test_eligibility_rules_mirror_engine(self):
        assert eligibility_checker(50000, 0)["indicative_outcome"] == "LIKELY_ELIGIBLE_FOR_FULL_ASSESSMENT"
        assert eligibility_checker(50000, 60000)["indicative_outcome"] == "LIKELY_DECLINE"
        assert eligibility_checker(50000, 0, times_90_days_late=3)["indicative_outcome"] == "LIKELY_DECLINE"
        assert eligibility_checker(50000, 0, times_90_days_late=2)["indicative_outcome"] != "LIKELY_DECLINE"
        assert eligibility_checker(3000, 0)["indicative_outcome"] == "LIKELY_MANUAL_REVIEW"
        assert eligibility_checker(50000, 0, dependents=6)["indicative_outcome"] == "LIKELY_MANUAL_REVIEW"
        assert "error" in eligibility_checker(0)

    def test_eligibility_accounts_for_new_emi(self):
        r = eligibility_checker(20000, 5000, loan_amount=2000000, tenure_months=24)
        assert r["indicative_outcome"] == "LIKELY_DECLINE"
        assert r["debt_ratio_after_new_emi_pct"] > 100
        assert "not a loan decision" in r["disclaimer"].lower()

    def test_application_status_found_missing_and_invalid(self):
        make_final_decision("APP1A2B3C4D5E", _credit("DECLINE", 0.9, "HIGH"), _fraud())
        assert application_status("app1a2b3c4d5e")["status"] == "DECLINED"
        assert application_status("APPFFFFFFFFFF")["found"] is False
        assert "error" in application_status("not-an-id")
        assert "error" in application_status("..\\..\\x")

    def test_run_tool_survives_bugs_and_bad_calls(self, monkeypatch):
        assert "error" in run_tool("does_not_exist", {})
        assert "error" in run_tool("emi_calculator", {"wrong_arg": 1})
        monkeypatch.setitem(chatbot.TOOL_FUNCTIONS, "emi_calculator",
                            mock.Mock(side_effect=RuntimeError("boom")))
        assert "tool failed" in run_tool("emi_calculator", {"principal": 1, "months": 1})["error"]

    def test_faq_retriever_without_kb(self, monkeypatch):
        monkeypatch.setattr(chatbot, "get_knowledge_base", lambda *a, **k: None)
        chatbot.reset_kb_cache()
        assert faq_retriever("how is emi calculated") == {"available": False, "passages": []}


# ============================================================
# 8. CHATBOT CONVERSATION (rules path)
# ============================================================

class TestChatbotRulesPath:

    @pytest.mark.parametrize("msg", ["tell me a joke", "what is the capital of France", "write me a poem about cats"])
    def test_off_topic_is_blocked_and_not_remembered(self, msg):
        bot = LoanAssistantChatbot()
        out = bot.chat(msg)
        assert out["blocked"] is True and out["tool_calls"] == []
        assert out["reply"] == _TEXT["en"]["off_topic"]
        assert len(bot.memory) == 0

    def test_off_topic_after_loan_talk_is_still_blocked(self):
        bot = LoanAssistantChatbot()
        bot.chat("What is the EMI for 5 lakh for 36 months at 12%?")
        out = bot.chat("tell me a joke")
        assert out["blocked"] is True

    def test_hindi_off_topic_reply(self):
        out = LoanAssistantChatbot().chat("tell me a joke", "hi")
        assert out["blocked"] and out["reply"] == _TEXT["hi"]["off_topic"] and out["language"] == "hi"

    def test_greeting_is_answered_but_not_remembered(self):
        bot = LoanAssistantChatbot()
        out = bot.chat("Hello!")
        assert out["reply"] == _TEXT["en"]["greet"] and out["blocked"] is False
        assert len(bot.memory) == 0

    def test_memory_is_capped_at_five_turns(self):
        bot = LoanAssistantChatbot()
        for i in range(8):
            bot.chat(f"What is the EMI for {i + 1} lakh for 36 months at 12%?")
        assert len(bot.memory) == 5
        assert "4 lakh" in bot.memory[0][0]            # turns 1-3 were dropped

    def test_overlong_message_rejected_and_not_remembered(self):
        bot = LoanAssistantChatbot()
        out = bot.chat("loan " * 120)                  # 600 chars
        assert out["reply"] == _TEXT["en"]["too_long"] and out["tool_calls"] == []
        assert len(bot.memory) == 0

    def test_exactly_500_chars_is_accepted(self):
        msg = ("loan " * 100)[:500]
        assert len(msg) == 500
        assert LoanAssistantChatbot().chat(msg)["reply"] != _TEXT["en"]["too_long"]

    def test_empty_message_gets_greeting(self):
        assert LoanAssistantChatbot().chat("   ")["reply"] == _TEXT["en"]["greet"]

    def test_three_90_day_lates_is_likely_decline(self):
        out = LoanAssistantChatbot().chat(
            "Am I eligible? my income is 50000 and existing EMIs are 5000, I have 3 payments 90 days late")
        call = out["tool_calls"][0]
        assert call["tool"] == "eligibility_checker" and call["args"]["times_90_days_late"] == 3
        assert "declined" in out["reply"].lower() and "indicative" in out["reply"].lower()
        assert out["source"] == "rules"

    def test_eligibility_without_numbers_asks_for_them(self):
        out = LoanAssistantChatbot().chat("Can I get a loan? am I eligible")
        assert out["reply"] == _TEXT["en"]["need_elig"]

    def test_emi_question_is_computed_by_the_tool(self):
        out = LoanAssistantChatbot().chat("What is the EMI for 5 lakh for 36 months at 12%?")
        assert "Rs. 16,607.15" in out["reply"]
        assert out["tool_calls"][0]["tool"] == "emi_calculator"

    def test_emi_with_years_and_no_rate_discloses_assumption(self):
        out = LoanAssistantChatbot().chat("calculate emi for 10 lakh for 5 years")
        assert out["tool_calls"][0]["args"]["months"] == 60
        assert "assumed" in out["reply"].lower()

    def test_followup_with_number_is_allowed_inside_live_conversation(self):
        bot = LoanAssistantChatbot()
        bot.chat("What is the EMI for 5 lakh for 36 months at 12%?")
        assert bot.chat("what about 60 months?")["blocked"] is False

    def test_tool_error_never_crashes_chat(self, monkeypatch):
        monkeypatch.setitem(chatbot.TOOL_FUNCTIONS, "emi_calculator", mock.Mock(side_effect=RuntimeError("boom")))
        out = LoanAssistantChatbot().chat("What is the EMI for 5 lakh for 36 months at 12%?")
        assert "tool failed" in out["reply"]

    def test_application_status_via_chat(self):
        make_final_decision("APP1A2B3C4D5E", _credit("DECLINE", 0.9, "HIGH"), _fraud())
        bot = LoanAssistantChatbot()
        assert "DECLINED" in bot.chat("status of APP1A2B3C4D5E please")["reply"]
        assert "No application found" in bot.chat("status of APPAAAAAAAAAA")["reply"]

    def test_faq_answer_comes_from_knowledge_base(self, kb, monkeypatch):
        monkeypatch.setattr(chatbot, "get_knowledge_base", lambda *a, **k: kb)
        chatbot.reset_kb_cache()
        out = LoanAssistantChatbot().chat("Can I prepay my loan early?")
        assert out["tool_calls"][0]["tool"] == "faq_retriever"
        assert "prepay" in out["reply"].lower() or "foreclos" in out["reply"].lower()


# ============================================================
# 9. CHATBOT — Gemini function-calling loop (HTTP mocked)
# ============================================================

def _fcall(name, args):
    return {"candidates": [{"content": {"parts": [{"functionCall": {"name": name, "args": args}}]}}]}


class TestChatbotGeminiPath:

    def test_two_step_tool_call_loop(self, gemini_key):
        responses = [
            _http(200, _fcall("emi_calculator", {"principal": 500000, "months": 36, "annual_rate_pct": 12})),
            _http(200, _gemini_response("Your EMI would be about Rs. 16,607 per month.")),
        ]
        with mock.patch("src.llm_explainer.requests.post", side_effect=responses) as post:
            out = LoanAssistantChatbot().chat("What would the EMI be for 5 lakh over 36 months at 12%?")
        assert out["source"] == "gemini"
        assert out["reply"] == "Your EMI would be about Rs. 16,607 per month."
        assert out["tool_calls"] == [{"tool": "emi_calculator",
                                      "args": {"principal": 500000, "months": 36, "annual_rate_pct": 12}}]
        assert post.call_count == 2
        # the tool RESULT (computed by our code) must have been fed back to the model
        second_body = post.call_args_list[1].kwargs["json"]
        fr = second_body["contents"][-1]["parts"][0]["functionResponse"]
        assert fr["name"] == "emi_calculator" and fr["response"]["result"]["monthly_emi"] == 16607.15
        assert "tools" in second_body and "systemInstruction" in second_body

    def test_plain_text_answer_without_tools(self, gemini_key):
        with mock.patch("src.llm_explainer.requests.post",
                        return_value=_http(200, _gemini_response("Please bring ID and address proof."))):
            out = LoanAssistantChatbot().chat("which documents are needed for a loan?")
        assert out["source"] == "gemini" and out["tool_calls"] == []

    def test_http_failure_falls_back_to_rules(self, gemini_key):
        with mock.patch("src.llm_explainer.requests.post", return_value=_http(429, {"error": "quota"})):
            out = LoanAssistantChatbot().chat("What is the EMI for 5 lakh for 36 months at 12%?")
        assert out["source"] == "rules" and "Rs. 16,607.15" in out["reply"]

    def test_endless_tool_calling_is_bounded(self, gemini_key):
        loop = _http(200, _fcall("emi_calculator", {"principal": 1000, "months": 12}))
        with mock.patch("src.llm_explainer.requests.post", return_value=loop) as post:
            out = LoanAssistantChatbot().chat("What is the EMI for 5 lakh for 36 months at 12%?")
        assert post.call_count <= MAX_TOOL_ROUNDS + 1
        assert out["source"] == "rules"                 # gave up on Gemini, answered itself

    def test_malformed_gemini_reply_falls_back(self, gemini_key):
        with mock.patch("src.llm_explainer.requests.post", return_value=_http(200, {"unexpected": True})):
            assert LoanAssistantChatbot().chat("What is the EMI for 5 lakh for 36 months at 12%?")["source"] == "rules"

    def test_unknown_tool_from_model_is_handled(self, gemini_key):
        responses = [_http(200, _fcall("delete_everything", {})), _http(200, _gemini_response("Sorry, I cannot do that."))]
        with mock.patch("src.llm_explainer.requests.post", side_effect=responses):
            out = LoanAssistantChatbot().chat("loan question please")
        assert out["source"] == "gemini"

    def test_off_topic_never_reaches_gemini(self, gemini_key):
        with mock.patch("src.llm_explainer.requests.post") as post:
            out = LoanAssistantChatbot().chat("tell me a joke")
        post.assert_not_called()
        assert out["blocked"] is True


# ============================================================
# 10. API ENDPOINTS (FastAPI TestClient, stub models)
# ============================================================

class StubModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        import numpy as np
        return np.array([[1 - self.p, self.p]] * len(X))


@pytest.fixture
def api(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import serving.api as api_module
    monkeypatch.setattr(api_module, "credit_model", StubModel(0.05))
    monkeypatch.setattr(api_module, "fraud_model", StubModel(0.01))
    monkeypatch.setattr(api_module, "credit_threshold", 0.5)
    monkeypatch.setattr(api_module, "fraud_threshold", 0.5)
    api_module._chat_sessions.clear()
    client = TestClient(api_module.app)
    client.module = api_module
    yield client
    api_module._chat_sessions.clear()


APPLICANT = {"full_name": "Ravi Kumar Sharma", "age": 32, "monthly_income": 60000.0,
             "loan_amount_requested": 500000.0, "tenure_months": 36, "loan_purpose": "Personal"}


def _underwrite(api):
    r = api.post("/underwrite", json=APPLICANT)
    assert r.status_code == 200, r.text
    return r.json()


class TestApiUnderwriteStoresApplication:

    def test_underwrite_persists_minimal_application_summary(self, api):
        d = _underwrite(api)
        assert d["final_verdict"] == "APPROVE"
        stored = api.get(f"/application/{d['application_id']}").json()
        assert stored["application"] == {"full_name": "Ravi Kumar Sharma", "loan_amount_requested": 500000.0,
                                         "tenure_months": 36, "loan_purpose": "Personal"}
        assert "60000" not in json.dumps(stored["application"])


class TestApiLetter:

    def test_letter_for_approved_application_contains_offer(self, api):
        app_id = _underwrite(api)["application_id"]
        r = api.get(f"/application/{app_id}/letter?language=en")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF")
        assert app_id in r.headers["content-disposition"]
        assert r.headers["x-language-used"] == "en"
        assert "Loan Offer Summary" in "\n".join(_pdf_text(r.content))

    def test_hindi_letter_without_font_reports_english(self, api, monkeypatch):
        monkeypatch.setattr(doc_generator, "_find_hindi_font", lambda: None)
        app_id = _underwrite(api)["application_id"]
        r = api.get(f"/application/{app_id}/letter?language=hi")
        assert r.status_code == 200 and r.headers["x-language-used"] == "en"

    def test_missing_application_is_404(self, api):
        assert api.get("/application/APPDOESNOT00/letter").status_code == 404

    @pytest.mark.parametrize("bad", ["..%5Cx", "a%2Fb", "x" * 65, "bad%20id"])
    def test_malformed_id_is_rejected(self, api, bad):
        assert api.get(f"/application/{bad}/letter").status_code in (400, 404)

    def test_invalid_language_is_422(self, api):
        app_id = _underwrite(api)["application_id"]
        assert api.get(f"/application/{app_id}/letter?language=fr").status_code == 422


class TestApiExplain:

    def test_explain_returns_template_without_key(self, api):
        app_id = _underwrite(api)["application_id"]
        r = api.post("/explain", json={"application_id": app_id, "language": "en"})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "template" and body["language"] == "en" and body["text"]
        assert body["text"].startswith("Hi Ravi")

    def test_explain_hindi(self, api):
        app_id = _underwrite(api)["application_id"]
        body = api.post("/explain", json={"application_id": app_id, "language": "hi"}).json()
        assert body["language"] == "hi" and any("ऀ" <= c <= "ॿ" for c in body["text"])

    def test_explain_missing_and_malformed(self, api):
        assert api.post("/explain", json={"application_id": "APPNOPE00000"}).status_code == 404
        assert api.post("/explain", json={"application_id": "..\\x"}).status_code == 400
        assert api.post("/explain", json={"application_id": "APP1", "language": "fr"}).status_code == 422


class TestApiChat:

    def test_chat_roundtrip_and_session_memory(self, api):
        r = api.post("/chat", json={"session_id": "s1", "message": "What is the EMI for 5 lakh for 36 months at 12%?"})
        assert r.status_code == 200 and "Rs. 16,607.15" in r.json()["reply"]
        api.post("/chat", json={"session_id": "s1", "message": "what about 60 months?"})
        assert len(api.module._chat_sessions["s1"].memory) == 2
        api.post("/chat", json={"session_id": "s2", "message": "What is the EMI for 5 lakh for 36 months at 12%?"})
        assert len(api.module._chat_sessions["s2"].memory) == 1          # sessions are isolated

    def test_off_topic_blocked_through_api(self, api):
        body = api.post("/chat", json={"session_id": "s1", "message": "tell me a joke"}).json()
        assert body["blocked"] is True

    def test_sessions_are_bounded_and_oldest_evicted(self, api, monkeypatch):
        monkeypatch.setattr(api.module, "MAX_CHAT_SESSIONS", 3)
        for sid in ("a", "b", "c"):
            api.post("/chat", json={"session_id": sid, "message": "loan emi"})
        api.post("/chat", json={"session_id": "a", "message": "loan emi"})        # touch "a" → now newest
        api.post("/chat", json={"session_id": "d", "message": "loan emi"})        # evicts oldest = "b"
        assert list(api.module._chat_sessions) == ["c", "a", "d"]

    def test_validation(self, api):
        assert api.post("/chat", json={"session_id": "bad id!", "message": "loan"}).status_code == 400
        assert api.post("/chat", json={"session_id": "s", "message": "x" * 2001}).status_code == 422
        assert api.post("/chat", json={"session_id": "s", "message": "loan", "language": "fr"}).status_code == 422
        assert api.post("/chat", json={"message": "loan"}).status_code == 422

    def test_message_between_501_and_2000_chars_gets_polite_rejection(self, api):
        body = api.post("/chat", json={"session_id": "s", "message": "loan " * 120}).json()
        assert body["reply"] == _TEXT["en"]["too_long"]

    def test_unexpected_chatbot_exception_becomes_500_not_crash(self, api, monkeypatch):
        monkeypatch.setattr(LoanAssistantChatbot, "chat", mock.Mock(side_effect=RuntimeError("x")))
        assert api.post("/chat", json={"session_id": "s", "message": "loan"}).status_code == 500
