# ============================================================
# LOAN MATH — shared by the chatbot (emi_calculator tool), the letter
# generator (loan offer summary + EMI schedule) and, in future, the portal.
# ============================================================
# Pure functions, no I/O, no model dependencies.
#
# The risk-based pricing curve below is ILLUSTRATIVE (same formula the
# User Portal's decision screen has used since Day 3): base 10.5% plus up
# to 15 points scaled by the model's predicted default probability. A
# real lender would use its treasury base rate + a risk premium table
# (see data/knowledge_base/bank_credit_policy_SYNTHETIC.txt, section 7).
# ============================================================

from typing import List, Dict

BASE_RATE_PCT = 10.5
MAX_RISK_PREMIUM_PCT = 15.0


def risk_based_rate(default_probability: float) -> float:
    """Annual interest rate (%) = base + premium proportional to default risk."""
    p = min(max(float(default_probability), 0.0), 1.0)
    return round(BASE_RATE_PCT + p * MAX_RISK_PREMIUM_PCT, 2)


def calculate_emi(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard reducing-balance EMI."""
    if months <= 0:
        raise ValueError("months must be positive")
    if principal <= 0:
        return 0.0
    r = annual_rate_pct / 1200.0
    if r == 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def amortization_schedule(principal: float, annual_rate_pct: float, months: int) -> List[Dict]:
    """Month-by-month schedule: opening balance, interest, principal, EMI, closing balance."""
    emi = calculate_emi(principal, annual_rate_pct, months)
    r = annual_rate_pct / 1200.0
    balance = principal
    rows = []
    for m in range(1, months + 1):
        interest = balance * r
        principal_part = emi - interest
        if m == months:                     # absorb rounding drift in the last instalment
            principal_part = balance
            emi_this = principal_part + interest
        else:
            emi_this = emi
        closing = balance - principal_part
        rows.append({
            "month": m,
            "opening": round(balance, 2),
            "interest": round(interest, 2),
            "principal": round(principal_part, 2),
            "emi": round(emi_this, 2),
            "closing": round(max(closing, 0.0), 2),
        })
        balance = closing
    return rows


def inr(amount: float, decimals: int = 0) -> str:
    """
    Indian digit grouping: 500000 -> '5,00,000'. Prefixed with 'Rs.' rather
    than the ₹ glyph because reportlab's built-in fonts (and many Windows
    fonts) have no ₹ glyph and render a black box instead.
    """
    negative = amount < 0
    amount = abs(float(amount))
    whole = int(amount)
    frac = f"{amount - whole:.{decimals}f}"[1:] if decimals else ""
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if negative else "") + "Rs. " + s + frac
