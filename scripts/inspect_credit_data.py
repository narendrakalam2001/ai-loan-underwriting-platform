# ============================================================
# INSPECT CREDIT DATA — read-only diagnostic for the real GMSC file
# ============================================================
# Prints exactly what the Day-5 "item 0" question needs, WITHOUT training:
#   1. sentinel values (96 / 98) in the three days-past-due columns
#   2. how often each hard rule in credit_engine would fire on this data
#   3. missing-income share and the income scale
#
# Usage:
#   python scripts/inspect_credit_data.py "D:\...\GiveMeSomeCredit\cs-training.csv"
# Paste the whole output back; it contains aggregate counts only, no rows.
# ============================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.data_loader import GMSC_COLUMN_RENAME
from src import config

LATE = ["times_30_59_days_late", "times_60_89_days_late", "times_90_days_late"]


def main(path: str):
    df = pd.read_csv(path)
    df = df.rename(columns={c: GMSC_COLUMN_RENAME[c.strip().lower()]
                            for c in df.columns if c.strip().lower() in GMSC_COLUMN_RENAME})
    n = len(df)
    print(f"rows = {n:,}   columns = {list(df.columns)}\n")

    print("=== 1. Days-past-due columns: value counts (top 12) ===")
    for c in LATE:
        vc = df[c].value_counts().sort_index()
        print(f"\n{c}:  max={df[c].max()}  rows with value >= 90: {(df[c] >= 90).sum():,}")
        print(vc.head(12).to_string())
        big = vc[vc.index >= 20]
        if len(big):
            print("  values >= 20:", dict(big))
    sent = ((df[LATE] >= 90).any(axis=1))
    print(f"\nRows with a 96/98-style value in ANY of the 3 columns: {sent.sum():,} "
          f"({sent.mean():.2%}) — default rate in those rows: "
          f"{df.loc[sent, 'serious_dlqin2yrs'].mean():.2%} (overall {df['serious_dlqin2yrs'].mean():.2%})")

    print("\n=== 2. Hard rules from src/config.py, as they fire on this raw data ===")
    inc = df["monthly_income"]
    print(f"MonthlyIncome missing:           {inc.isna().sum():,} ({inc.isna().mean():.2%})")
    print(f"MonthlyIncome median / p10 / p90: {inc.median():,.0f} / {inc.quantile(.1):,.0f} / {inc.quantile(.9):,.0f}")
    filled = inc.fillna(inc.median())
    print(f"HIGH_DEBT_RATIO  (debt_ratio > {config.HIGH_DEBT_RATIO_RULE}): {(df['debt_ratio'] > config.HIGH_DEBT_RATIO_RULE).sum():,} "
          f"({(df['debt_ratio'] > config.HIGH_DEBT_RATIO_RULE).mean():.2%})")
    miss = inc.isna()
    print(f"   ...of which income was missing: {((df['debt_ratio'] > config.HIGH_DEBT_RATIO_RULE) & miss).sum():,}   "
          f"(debt_ratio median where income missing: {df.loc[miss, 'debt_ratio'].median():,.1f}; where present: {df.loc[~miss, 'debt_ratio'].median():.2f})")
    print(f"SEVERE_DELINQUENCY (90+ late >= {config.SEVERE_DELINQUENCY_RULE}): {(df['times_90_days_late'] >= config.SEVERE_DELINQUENCY_RULE).sum():,} "
          f"({(df['times_90_days_late'] >= config.SEVERE_DELINQUENCY_RULE).mean():.2%})")
    print(f"   ...of which are 96/98 sentinel rows: {(df['times_90_days_late'] >= 90).sum():,}")
    print(f"LOW_INCOME (filled income < {config.MIN_INCOME_RULE_MODEL_UNITS:,.0f} model units = Rs. {config.MIN_INCOME_RULE:,}): {(filled < config.MIN_INCOME_RULE_MODEL_UNITS).sum():,} "
          f"({(filled < config.MIN_INCOME_RULE_MODEL_UNITS).mean():.2%})")
    print(f"MANY_DEPENDENTS (>= {config.HIGH_DEPENDENTS_RULE}): {(df['dependents'].fillna(0) >= config.HIGH_DEPENDENTS_RULE).sum():,}")

    print("\nDefault rate by rule group (is the rule justified by the label?):")
    for name, mask in [("debt_ratio > 1", df["debt_ratio"] > 1),
                       ("debt_ratio > 1 AND income present", (df["debt_ratio"] > 1) & ~miss),
                       ("90+ late >= 3 excluding sentinels", (df["times_90_days_late"] >= 3) & (df["times_90_days_late"] < 90)),
                       ("90+ late = 96/98 (sentinel)", df["times_90_days_late"] >= 90),
                       ("income < MIN_INCOME_RULE", filled < config.MIN_INCOME_RULE_MODEL_UNITS)]:
        if mask.sum():
            print(f"  {name:<40} n={mask.sum():>7,}  default rate={df.loc[mask, 'serious_dlqin2yrs'].mean():.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('usage: python scripts/inspect_credit_data.py "<path to cs-training.csv>"')
    main(sys.argv[1])
