# ============================================================
# SYNTHETIC BANK STATEMENT GENERATOR — Module 3 (Document Parser)
# ============================================================
# Generates SYNTHETIC bank statement PDFs (no real customer data
# anywhere) that mimic the layout of an Indian bank statement:
# a header block + a running Date / Narration / Debit / Credit /
# Balance ledger. Faker drives names, employers and dates; the
# transaction economics (salary, EMIs, spends) are built from
# simple, documented random-walk rules so the ground truth used
# to validate document_parser.py is always known exactly.
#
# Every statement's ground truth (the numbers a real underwriter
# should have extracted) is saved alongside the PDF as a JSON
# sidecar file, so document_parser.py's extraction accuracy can
# be scored automatically in tests/.
#
# THESE STATEMENTS ARE 100% SYNTHETIC. No real account holder,
# bank, employer or transaction data is used anywhere in this
# module — see README "Synthetic Data Disclosure".
# ============================================================

import os
import json
import random
import logging
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Dict

from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.config import (
    SYNTHETIC_STATEMENT_COUNT, SYNTHETIC_STATEMENT_MONTHS,
    SYNTHETIC_SEED, SYNTHETIC_STATEMENTS_DIR,
)

logger = logging.getLogger(__name__)

fake = Faker("en_IN")

# ── Synthetic-only bank shells (fictitious names, not real banks) ──
SYNTHETIC_BANKS = [
    "Bharat Trust Bank", "Sunrise National Bank", "Unity Commercial Bank",
    "Nilgiri Co-operative Bank", "Vaayu Digital Bank",
]

EMI_LABELS = [
    "AUTO DEBIT-EMI HOME LOAN", "NACH EMI PERSONAL LOAN",
    "AUTO DEBIT-EMI CAR LOAN", "LOAN INSTALLMENT-CONSUMER DURABLE",
]

SPEND_NARRATIONS = [
    "UPI-SWIGGY", "UPI-AMAZON RETAIL", "POS PURCHASE-RELIANCE RETAIL",
    "UPI-ZOMATO", "ATM WDL", "ELECTRICITY BILL PAYMENT",
    "UPI-BIGBASKET", "MOBILE RECHARGE", "NETFLIX SUBSCRIPTION",
]


@dataclass
class StatementGroundTruth:
    account_holder: str
    employer: str
    account_number: str
    bank_name: str
    monthly_income: List[float]
    avg_monthly_income: float
    income_stability: str          # Good / Moderate / Poor (by design, for scoring)
    monthly_emi_total: float
    month_end_balances: List[float]
    balance_stability: str
    statement_months: List[str]


def _month_range(n_months: int) -> List[date]:
    """Returns the 1st of each of the last n_months, oldest first."""
    today = date.today().replace(day=1)
    months = []
    cursor = today
    for _ in range(n_months):
        months.append(cursor)
        # step back one month
        prev_month_end = cursor - timedelta(days=1)
        cursor = prev_month_end.replace(day=1)
    return list(reversed(months))


def _generate_one_profile(rng: random.Random) -> Dict:
    """
    Draws a synthetic applicant profile with an explicit, documented
    income-stability archetype so document_parser.py's classification
    can be evaluated against a known label.
    """
    archetype = rng.choices(
        population=["stable", "moderate", "unstable"],
        weights=[0.55, 0.30, 0.15],
        k=1
    )[0]

    base_income = rng.uniform(25000, 150000)

    if archetype == "stable":
        income_cv, balance_cv = 0.03, 0.15
    elif archetype == "moderate":
        income_cv, balance_cv = 0.15, 0.40
    else:
        income_cv, balance_cv = 0.35, 0.75

    return {
        "archetype":   archetype,
        "base_income": round(base_income, 2),
        "income_cv":   income_cv,
        "balance_cv":  balance_cv,
    }


def _cv_to_label(cv: float) -> str:
    if cv <= 0.25:
        return "Good"
    elif cv <= 0.50:
        return "Moderate"
    return "Poor"


def generate_statement_ground_truth(rng: random.Random) -> StatementGroundTruth:
    profile   = _generate_one_profile(rng)
    n_months  = SYNTHETIC_STATEMENT_MONTHS
    months    = _month_range(n_months)

    account_holder = fake.name()
    employer       = fake.company()
    account_number = fake.numerify("###########")
    bank_name      = rng.choice(SYNTHETIC_BANKS)

    # ── Monthly salary credits (with archetype-driven variance) ──
    monthly_income = [
        max(5000.0, round(rng.gauss(profile["base_income"],
                                     profile["base_income"] * profile["income_cv"]), 2))
        for _ in months
    ]
    avg_income = round(sum(monthly_income) / len(monthly_income), 2)
    income_cv_actual = (
        (sum((x - avg_income) ** 2 for x in monthly_income) / len(monthly_income)) ** 0.5
    ) / avg_income if avg_income else 0
    income_stability = _cv_to_label(income_cv_actual)

    # ── Monthly EMI obligation (fixed installment, 0-2 loans) ────
    n_loans   = rng.choices([0, 1, 2], weights=[0.35, 0.45, 0.20], k=1)[0]
    emi_total = round(sum(rng.uniform(2000, 18000) for _ in range(n_loans)), 2)

    # ── Simulated month-end balances ──────────────────────────────
    running_balance = rng.uniform(15000, 60000)
    month_end_balances = []
    for income in monthly_income:
        spend = rng.uniform(0.5, 0.9) * income
        noise = rng.gauss(0, running_balance * profile["balance_cv"] * 0.3)
        running_balance = max(1000.0, running_balance + income - emi_total - spend + noise)
        month_end_balances.append(round(running_balance, 2))

    bal_mean = sum(month_end_balances) / len(month_end_balances)
    bal_cv = (
        (sum((x - bal_mean) ** 2 for x in month_end_balances) / len(month_end_balances)) ** 0.5
    ) / bal_mean if bal_mean else 0
    balance_stability = _cv_to_label(bal_cv)

    return StatementGroundTruth(
        account_holder=account_holder,
        employer=employer,
        account_number=account_number,
        bank_name=bank_name,
        monthly_income=monthly_income,
        avg_monthly_income=avg_income,
        income_stability=income_stability,
        monthly_emi_total=emi_total,
        month_end_balances=month_end_balances,
        balance_stability=balance_stability,
        statement_months=[m.strftime("%b-%Y") for m in months],
    )


def _build_transaction_ledger(gt: StatementGroundTruth, rng: random.Random) -> List[List[str]]:
    """
    Builds a row-per-transaction ledger (Date, Narration, Debit, Credit, Balance)
    consistent with the ground-truth income/EMI/balance numbers above.
    """
    rows = [["Date", "Narration", "Debit (₹)", "Credit (₹)", "Balance (₹)"]]
    months = _month_range(SYNTHETIC_STATEMENT_MONTHS)
    running_balance = gt.month_end_balances[0] - gt.monthly_income[0] + gt.monthly_emi_total

    for i, month_start in enumerate(months):
        income   = gt.monthly_income[i]
        emi      = gt.monthly_emi_total
        end_bal  = gt.month_end_balances[i]

        # Salary credit — always the 1st working day of the month
        running_balance += income
        rows.append([
            month_start.strftime("%d-%b-%Y"),
            f"SALARY CREDIT - {gt.employer.upper()}",
            "", f"{income:,.2f}", f"{running_balance:,.2f}"
        ])

        # EMI debit(s) — around the 5th
        if emi > 0:
            emi_date = month_start + timedelta(days=4)
            running_balance -= emi
            rows.append([
                emi_date.strftime("%d-%b-%Y"),
                rng.choice(EMI_LABELS),
                f"{emi:,.2f}", "", f"{running_balance:,.2f}"
            ])

        # A handful of everyday spends spread across the month
        n_spends = rng.randint(4, 9)
        for _ in range(n_spends):
            day   = rng.randint(6, 27)
            spend = round(rng.uniform(200, 4500), 2)
            running_balance -= spend
            txn_date = month_start + timedelta(days=min(day, 27))
            rows.append([
                txn_date.strftime("%d-%b-%Y"),
                rng.choice(SPEND_NARRATIONS),
                f"{spend:,.2f}", "", f"{max(running_balance, 0):,.2f}"
            ])

        # Nudge the running total to land exactly on the target month-end
        # balance so the PDF ledger and the ground truth JSON always agree.
        drift = end_bal - running_balance
        if abs(drift) > 0.01:
            adj_date = month_start + timedelta(days=28)
            if drift > 0:
                rows.append([adj_date.strftime("%d-%b-%Y"), "INTEREST CREDIT", "",
                             f"{drift:,.2f}", f"{end_bal:,.2f}"])
            else:
                rows.append([adj_date.strftime("%d-%b-%Y"), "OTHER CHARGES / ADJ",
                             f"{abs(drift):,.2f}", "", f"{end_bal:,.2f}"])
            running_balance = end_bal

    return rows


def render_statement_pdf(gt: StatementGroundTruth, ledger_rows: List[List[str]], out_path: str):
    """Renders the synthetic statement as a formatted PDF using reportlab."""
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "Header", parent=styles["Heading1"], fontSize=16, spaceAfter=2,
        textColor=colors.HexColor("#0B3D91")
    )
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    disclosure_style = ParagraphStyle(
        "Disclosure", parent=styles["Normal"], fontSize=7.5,
        textColor=colors.HexColor("#B00020"), spaceBefore=8
    )

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=15 * mm, bottomMargin=15 * mm,
                             leftMargin=15 * mm, rightMargin=15 * mm)

    elements = [
        Paragraph(gt.bank_name, header_style),
        Paragraph("Account Statement", sub_style),
        Spacer(1, 6),
        Paragraph(f"Account Holder: {gt.account_holder}", styles["Normal"]),
        Paragraph(f"Account Number: XXXXXX{gt.account_number[-4:]}", styles["Normal"]),
        Paragraph(f"Statement Period: {gt.statement_months[0]} to {gt.statement_months[-1]}",
                   styles["Normal"]),
        Spacer(1, 10),
    ]

    table = Table(ledger_rows, colWidths=[22 * mm, 62 * mm, 26 * mm, 26 * mm, 30 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6FA")]),
        ("ALIGN", (2, 0), (4, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)

    elements.append(Paragraph(
        "⚠ SYNTHETIC DATA — This statement is machine-generated for capstone-project "
        "testing only. All names, account numbers and transactions are fictitious "
        "(Faker-generated) and do not correspond to any real person, bank or account.",
        disclosure_style
    ))

    doc.build(elements)


def generate_synthetic_statements(
    n: int = SYNTHETIC_STATEMENT_COUNT,
    out_dir: str = SYNTHETIC_STATEMENTS_DIR,
    seed: int = SYNTHETIC_SEED,
) -> List[str]:
    """
    Generates `n` synthetic bank-statement PDFs + ground-truth JSON
    sidecars. Returns the list of application IDs generated.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    Faker.seed(seed)

    app_ids = []
    for i in range(n):
        gt = generate_statement_ground_truth(rng)
        ledger_rows = _build_transaction_ledger(gt, rng)

        app_id = f"SYN{i + 1:05d}"
        pdf_path  = os.path.join(out_dir, f"{app_id}_statement.pdf")
        json_path = os.path.join(out_dir, f"{app_id}_ground_truth.json")

        render_statement_pdf(gt, ledger_rows, pdf_path)

        with open(json_path, "w") as f:
            json.dump({
                "application_id": app_id,
                **gt.__dict__,
            }, f, indent=2)

        app_ids.append(app_id)

        if (i + 1) % 100 == 0:
            logger.info("Generated %d/%d synthetic statements", i + 1, n)

    logger.info("Synthetic statement generation complete → %s (%d files)", out_dir, n)
    return app_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_synthetic_statements()
