# ============================================================
# DOCUMENT INTELLIGENCE PARSER — Module 3
# ============================================================
# Extracts underwriting-relevant signals from a bank-statement
# PDF:
#   • Monthly income        (average + stability: Good/Moderate/Poor)
#   • Employer name
#   • Existing EMI / loan obligations (monthly ₹)
#   • Account balance trend (month-end balances + stability)
#
# Pipeline:
#   1. PyMuPDF (fitz)  — extract raw text per page (works on the
#      text layer produced by any digitally-generated statement;
#      a scanned/photographed statement would additionally need
#      an OCR pre-pass, which is out of scope for this module).
#   2. Regex             — parse the structured ledger rows
#      (date / narration / amount / balance) and the header block.
#   3. spaCy NER         — secondary, best-effort cross-check that
#      looks for ORG entities in salary-credit narrations, used
#      only when the primary regex pattern ("SALARY CREDIT - X")
#      does not match. This is a soft dependency: if the spaCy
#      model isn't installed, the parser logs a warning and
#      falls back to regex-only extraction rather than failing.
# ============================================================

import re
import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

import fitz  # PyMuPDF

from src.config import (
    SALARY_KEYWORDS, EMI_KEYWORDS,
    BALANCE_STABILITY_GOOD_CV, BALANCE_STABILITY_MODERATE_CV,
)

logger = logging.getLogger(__name__)

# ── spaCy is a soft dependency — parser degrades gracefully without it ──
_NLP = None
_SPACY_TRIED = False


def _get_nlp():
    """Lazily loads the spaCy English model; caches None on failure so we
    only try once per process."""
    global _NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _NLP
    _SPACY_TRIED = True
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
        logger.info("spaCy NER model loaded (en_core_web_sm)")
    except Exception as exc:
        logger.warning(
            "spaCy model unavailable (%s) — employer NER cross-check disabled, "
            "falling back to regex-only extraction.", exc
        )
        _NLP = None
    return _NLP


DATE_RE = re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}$")
AMOUNT_RE = re.compile(r"^-?[\d,]+\.\d{2}$")


@dataclass
class ParsedTransaction:
    date: str
    narration: str
    amount: float
    balance: float
    direction: str   # "CREDIT" or "DEBIT"


@dataclass
class DocumentExtractionResult:
    account_holder: Optional[str]
    bank_name: Optional[str]
    account_number_masked: Optional[str]
    statement_period: Optional[str]
    employer: Optional[str]
    employer_source: str                 # "regex" | "spacy_ner" | "not_found"
    detected_monthly_income: List[float]
    avg_monthly_income: float
    income_stability: str                # Good / Moderate / Poor
    existing_monthly_emi: float
    month_end_balances: List[float]
    account_stability: str               # Good / Moderate / Poor
    n_transactions_parsed: int
    warnings: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def _extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _cv_to_label(cv: float) -> str:
    if cv <= BALANCE_STABILITY_GOOD_CV:
        return "Good"
    elif cv <= BALANCE_STABILITY_MODERATE_CV:
        return "Moderate"
    return "Poor"


def _parse_header(lines: List[str]) -> Dict:
    header = {
        "bank_name": lines[0].strip() if lines else None,
        "account_holder": None,
        "account_number_masked": None,
        "statement_period": None,
    }
    for line in lines[:8]:
        m = re.search(r"Account Holder:\s*(.+)", line)
        if m:
            header["account_holder"] = m.group(1).strip()

        m = re.search(r"Account Number:\s*(\S+)", line)
        if m:
            header["account_number_masked"] = m.group(1).strip()

        m = re.search(r"Statement Period:\s*(.+)", line)
        if m:
            header["statement_period"] = m.group(1).strip()

    return header


def _parse_transactions(lines: List[str]) -> List[ParsedTransaction]:
    """
    Ledger rows are laid out as 4 consecutive text lines:
        date, narration, amount, balance
    (only ONE of debit/credit ever has a value per row, so PyMuPDF's
    text layer always yields exactly one amount line per transaction —
    the sign/direction is recovered afterwards from balance deltas and
    narration keywords, not from column position.)
    """
    # Find where the ledger starts (first line matching the date pattern)
    start_idx = next((i for i, l in enumerate(lines) if DATE_RE.match(l.strip())), None)
    if start_idx is None:
        return []

    rows: List[ParsedTransaction] = []
    i = start_idx
    prev_balance = None

    while i < len(lines) - 2:
        date_line = lines[i].strip()
        if not DATE_RE.match(date_line):
            i += 1
            continue

        narration = lines[i + 1].strip()
        amount_line = lines[i + 2].strip() if i + 2 < len(lines) else ""
        balance_line = lines[i + 3].strip() if i + 3 < len(lines) else ""

        if not (AMOUNT_RE.match(amount_line) and AMOUNT_RE.match(balance_line)):
            i += 1
            continue

        amount  = float(amount_line.replace(",", ""))
        balance = float(balance_line.replace(",", ""))

        if prev_balance is None:
            direction = "CREDIT" if _is_credit_keyword(narration) else "DEBIT"
        else:
            delta = balance - prev_balance
            if _is_credit_keyword(narration):
                direction = "CREDIT"
            elif _is_debit_keyword(narration):
                direction = "DEBIT"
            else:
                direction = "CREDIT" if delta > 0 else "DEBIT"

        rows.append(ParsedTransaction(
            date=date_line, narration=narration,
            amount=amount, balance=balance, direction=direction
        ))
        prev_balance = balance
        i += 4

    return rows


def _is_credit_keyword(narration: str) -> bool:
    upper = narration.upper()
    return any(k in upper for k in SALARY_KEYWORDS) or "INTEREST CREDIT" in upper


def _is_debit_keyword(narration: str) -> bool:
    upper = narration.upper()
    return any(k in upper for k in EMI_KEYWORDS)


def _extract_employer(transactions: List[ParsedTransaction], full_text: str) -> (Optional[str], str):
    """
    Primary: regex on 'SALARY CREDIT - <EMPLOYER>' narrations.
    Fallback: spaCy NER ORG entity found in the same narration lines.
    """
    salary_narrations = [
        t.narration for t in transactions
        if t.direction == "CREDIT" and any(k in t.narration.upper() for k in SALARY_KEYWORDS)
    ]

    for narration in salary_narrations:
        m = re.search(r"SALARY\s+CREDIT\s*-\s*(.+)", narration, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().title(), "regex"

    # ── spaCy NER fallback ──────────────────────────────────────
    nlp = _get_nlp()
    if nlp is not None and salary_narrations:
        for narration in salary_narrations:
            doc = nlp(narration)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if orgs:
                return orgs[0].title(), "spacy_ner"

    return None, "not_found"


def _monthly_aggregate(transactions: List[ParsedTransaction], keep_fn) -> Dict[str, float]:
    """Sums matching transaction amounts per calendar month (Mon-YYYY)."""
    buckets: Dict[str, float] = defaultdict(float)
    for t in transactions:
        if keep_fn(t):
            month_key = datetime.strptime(t.date, "%d-%b-%Y").strftime("%b-%Y")
            buckets[month_key] += t.amount
    return dict(buckets)


def parse_bank_statement(pdf_path: str) -> DocumentExtractionResult:
    warnings: List[str] = []

    raw_text = _extract_text(pdf_path)
    lines = [l for l in raw_text.split("\n") if l.strip() != ""]

    header = _parse_header(lines)
    transactions = _parse_transactions(lines)

    if not transactions:
        warnings.append("No transactions could be parsed from this document.")

    # ── Monthly income (salary credits) ──────────────────────────
    income_by_month = _monthly_aggregate(
        transactions,
        keep_fn=lambda t: t.direction == "CREDIT" and any(
            k in t.narration.upper() for k in SALARY_KEYWORDS
        ),
    )
    monthly_income = list(income_by_month.values())

    if monthly_income:
        avg_income = round(statistics.mean(monthly_income), 2)
        income_cv = (statistics.pstdev(monthly_income) / avg_income) if avg_income else 0.0
        income_stability = _cv_to_label(income_cv)
    else:
        avg_income = 0.0
        income_stability = "Poor"
        warnings.append("No salary-credit narrations detected — income could not be verified.")

    # ── Existing EMI obligations ──────────────────────────────────
    emi_by_month = _monthly_aggregate(
        transactions,
        keep_fn=lambda t: t.direction == "DEBIT" and any(
            k in t.narration.upper() for k in EMI_KEYWORDS
        ),
    )
    existing_monthly_emi = round(statistics.mean(list(emi_by_month.values())), 2) if emi_by_month else 0.0

    # ── Employer ────────────────────────────────────────────────
    employer, employer_source = _extract_employer(transactions, raw_text)
    if employer is None:
        warnings.append("Employer name could not be identified from the statement.")

    # ── Month-end balances / account stability ────────────────────
    balances_by_month: Dict[str, float] = {}
    for t in transactions:
        month_key = datetime.strptime(t.date, "%d-%b-%Y").strftime("%b-%Y")
        balances_by_month[month_key] = t.balance   # last write per month wins (chronological order)

    month_end_balances = list(balances_by_month.values())
    if month_end_balances:
        bal_mean = statistics.mean(month_end_balances)
        bal_cv = (statistics.pstdev(month_end_balances) / bal_mean) if bal_mean else 0.0
        account_stability = _cv_to_label(bal_cv)
    else:
        account_stability = "Poor"
        warnings.append("No month-end balances detected — account stability could not be assessed.")

    result = DocumentExtractionResult(
        account_holder=header["account_holder"],
        bank_name=header["bank_name"],
        account_number_masked=header["account_number_masked"],
        statement_period=header["statement_period"],
        employer=employer,
        employer_source=employer_source,
        detected_monthly_income=monthly_income,
        avg_monthly_income=avg_income,
        income_stability=income_stability,
        existing_monthly_emi=existing_monthly_emi,
        month_end_balances=month_end_balances,
        account_stability=account_stability,
        n_transactions_parsed=len(transactions),
        warnings=warnings,
    )

    logger.info(
        "Parsed %s | txns=%d | avg_income=%.2f (%s) | emi=%.2f | balance=%s",
        pdf_path, result.n_transactions_parsed, result.avg_monthly_income,
        result.income_stability, result.existing_monthly_emi, result.account_stability
    )

    return result


if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_statements/SYN00001_statement.pdf"
    res = parse_bank_statement(path)
    print(json.dumps(res.to_dict(), indent=2))
