# ============================================================
# DATA LOADER + FEATURE ENGINEERING — AI Loan Underwriting Platform
# ============================================================
# Loads and validates the two datasets that back Module 1
# (Credit Risk) and Module 2 (Fraud Detection). Both loaders
# follow the exact validation contract used in the reference
# projects so the rest of the pipeline (leakage_check,
# model_tuning, evaluation) can stay identical across modules.
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required columns per dataset ─────────────────────────────
# CREDIT_REQUIRED_COLS uses the clean names produced by
# GMSC_COLUMN_RENAME below (Kaggle "Give Me Some Credit" dataset).
CREDIT_REQUIRED_COLS = [
    "serious_dlqin2yrs", "revolving_utilization", "age", "debt_ratio",
    "monthly_income", "open_credit_lines", "times_90_days_late",
    "real_estate_loans", "dependents",
]
FRAUD_REQUIRED_COLS  = ["Time", "Amount", "Class"]

# Placeholder codes used in the three days-past-due columns of the real GMSC file.
LATE_SENTINEL_CODES = (96, 98)

# Kaggle's raw GMSC column names are CamelCase with no separators (and one
# hyphenated name), so the generic "replace non-alnum with underscore" used
# for the old dataset can't split them into readable snake_case on its own.
# Explicit rename map instead (matched case-insensitively against the raw
# header after stripping whitespace).
GMSC_COLUMN_RENAME = {
    "seriousdlqin2yrs":                     "serious_dlqin2yrs",
    "revolvingutilizationofunsecuredlines":  "revolving_utilization",
    "age":                                   "age",
    "numberoftime30-59dayspastduenotworse":  "times_30_59_days_late",
    "debtratio":                             "debt_ratio",
    "monthlyincome":                         "monthly_income",
    "numberofopencreditlinesandloans":       "open_credit_lines",
    "numberoftimes90dayslate":               "times_90_days_late",
    "numberrealestateloansorlines":          "real_estate_loans",
    "numberoftime60-89dayspastduenotworse":  "times_60_89_days_late",
    "numberofdependents":                    "dependents",
}


# ============================================================
# MODULE 1 — CREDIT RISK DATA VALIDATION
# ============================================================

def validate_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates Kaggle's "Give Me Some Credit" dataset used by the Credit Risk
    Scoring Engine. Target `serious_dlqin2yrs` (from SeriousDlqin2yrs) is a
    GENUINE default-risk label — 1 means the borrower was 90+ days past due
    within 2 years — unlike the Bank Personal Loan Modelling dataset this
    replaced, whose "personal_loan" label meant "accepted a marketing offer".

    Known, well-documented data-quality issues in this dataset that this
    function explicitly handles rather than silently passing through:
      - An unnamed index column (first column in cs-training.csv) → dropped.
      - `monthly_income` has ~20% missing values → median-imputed, with a
        `monthly_income_was_missing` flag kept as a feature (missingness
        itself is informative here — bureaus disproportionately fail to
        report income for higher-risk applicants).
      - `dependents` has a small fraction of missing values → imputed with 0
        (the modal value), with the same missingness-flag pattern.
      - 269 rows (0.18%) of the real file carry the codes 96 / 98 in ALL THREE
        days-past-due columns. They are not counts (nobody was "98 times late"
        in 2 years) — they are placeholder codes. Left alone they would
        (a) trigger the SEVERE_DELINQUENCY auto-decline rule for the wrong
        reason and (b) dominate every scaler/clipper. They are replaced by 0
        and flagged with `late_count_was_sentinel` (those rows default at
        ~55% vs 6.7% overall, so the flag is informative to the model).
        Genuine counts such as 5 or 11 are untouched.
      - `debt_ratio` is NOT a ratio when `monthly_income` is missing: in the
        real file those rows hold an absolute debt amount (median ~1,159), so
        `debt_ratio > 1` would auto-decline ~28k applicants (~19% of the file)
        whose default rate (~6.5%) is no worse than average. For those rows
        `debt_ratio` is replaced by the median ratio of rows that DO have
        income; `monthly_income_was_missing` still records the gap.
      - A handful of `age == 0` rows are known bad data (Kaggle's own data
        page documents this) → replaced with the median age.
      - `revolving_utilization` and `debt_ratio` contain extreme outliers
        (e.g. utilization in the thousands) from data-entry errors — these
        are intentionally left for `preprocessing.Clipper` to cap downstream
        rather than dropped here, consistent with how every other numeric
        feature in this pipeline is handled.
    """
    df.columns = [str(c).strip() for c in df.columns]

    # Drop the anonymous pandas-index column Kaggle ships as column 0
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        logger.info("Dropped unnamed index column(s): %s", unnamed_cols)

    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in GMSC_COLUMN_RENAME:
            rename_map[col] = GMSC_COLUMN_RENAME[key]
    df.rename(columns=rename_map, inplace=True)

    missing = [c for c in CREDIT_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after rename: {missing}. "
            f"Got columns: {list(df.columns)}. Expected the raw Kaggle "
            f"'Give Me Some Credit' headers (cs-training.csv)."
        )

    if not set(df["serious_dlqin2yrs"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target 'serious_dlqin2yrs' must contain only 0 and 1")

    if df.shape[0] < 100:
        raise ValueError("Credit dataset too small for training (< 100 rows)")

    # ── Known data-quality fixes ──────────────────────────────
    # 96 / 98 placeholder codes in the days-past-due columns (see docstring)
    late_present = [c for c in ("times_30_59_days_late", "times_60_89_days_late", "times_90_days_late")
                    if c in df.columns]
    sentinel_rows = df[late_present].isin(LATE_SENTINEL_CODES).any(axis=1) if late_present else pd.Series(False, index=df.index)
    df["late_count_was_sentinel"] = sentinel_rows.astype(int)
    if sentinel_rows.any():
        for c in late_present:
            df.loc[df[c].isin(LATE_SENTINEL_CODES), c] = 0
        logger.info("Replaced 96/98 placeholder codes in %d row(s) with 0 and flagged them "
                    "(late_count_was_sentinel)", int(sentinel_rows.sum()))

    income_missing = df["monthly_income"].isna()
    df["monthly_income_was_missing"] = income_missing.astype(int)

    # debt_ratio is an absolute amount, not a ratio, where income is missing (see docstring)
    valid_ratio_median = df.loc[~income_missing, "debt_ratio"].median()
    if income_missing.any() and pd.notna(valid_ratio_median):
        n_fixed = int((income_missing & (df["debt_ratio"] > 1.0)).sum())
        df.loc[income_missing, "debt_ratio"] = valid_ratio_median
        logger.info("debt_ratio replaced by median ratio (%.3f) for %d row(s) with missing income "
                    "(%d of them had debt_ratio > 1)", valid_ratio_median, int(income_missing.sum()), n_fixed)

    income_median = df["monthly_income"].median()
    df["monthly_income"] = df["monthly_income"].fillna(income_median)

    df["dependents_was_missing"] = df["dependents"].isna().astype(int)
    df["dependents"] = df["dependents"].fillna(0)

    bad_age_mask = df["age"] <= 0
    if bad_age_mask.any():
        age_median = df.loc[~bad_age_mask, "age"].median()
        logger.info("Fixed %d row(s) with age <= 0 (set to median=%s)", bad_age_mask.sum(), age_median)
        df.loc[bad_age_mask, "age"] = age_median

    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate credit rows", dropped)

    logger.info("Credit data validation passed | shape=%s | default_rate=%.4f",
                df.shape, df["serious_dlqin2yrs"].mean())

    return df


def add_credit_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for genuine default-risk scoring on GMSC-shaped data."""
    df = df.copy()

    late_cols = [c for c in ("times_30_59_days_late", "times_60_89_days_late", "times_90_days_late")
                 if c in df.columns]
    if late_cols:
        df["total_late_incidents"] = df[late_cols].sum(axis=1)
        df["has_delinquency_history"] = (df[late_cols].sum(axis=1) > 0).astype(int)
        df["has_severe_delinquency"] = (df.get("times_90_days_late", 0) > 0).astype(int)

    df["income_per_dependent"] = df["monthly_income"] / (df["dependents"] + 1)
    df["credit_lines_per_dependent"] = df["open_credit_lines"] / (df["dependents"] + 1)

    df["high_utilization_flag"] = (df["revolving_utilization"] > 1.0).astype(int)
    df["high_debt_ratio_flag"]  = (df["debt_ratio"] > 1.0).astype(int)

    df["has_real_estate_loan"] = (df["real_estate_loans"] > 0).astype(int)
    df["debt_ratio_x_utilization"] = df["debt_ratio"] * df["revolving_utilization"]

    df["age_bin"] = pd.cut(
        df["age"], bins=[0, 30, 40, 50, 60, 120],
        labels=[0, 1, 2, 3, 4], include_lowest=True
    ).astype(float)

    df["near_retirement"] = (df["age"] >= 60).astype(int)

    logger.info("Credit feature engineering done | total columns=%d", df.shape[1])
    return df


# ============================================================
# MODULE 2 — FRAUD DATA VALIDATION
# ============================================================

def validate_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the credit-card-transaction dataset (Time, V1..V28,
    Amount, Class) used by the Fraud Detection Engine. Mirrors the
    reference Fraud Detection project's validation contract.
    """
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in FRAUD_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Target 'Class' must contain only 0 and 1")

    if df.shape[0] < 100:
        raise ValueError("Fraud dataset too small for training (< 100 rows)")

    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate fraud rows", dropped)

    logger.info("Fraud data validation passed | shape=%s | fraud_rate=%.5f",
                df.shape, df["Class"].mean())

    return df


def add_fraud_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight, leakage-safe feature engineering on top of the
    anonymized PCA features (V1-V28) + Time + Amount.
    """
    df = df.copy()

    # Preserve the raw amount for the rule engine before scaling downstream
    df["Amount_original"] = df["Amount"]

    # Time of day in hours (Time is seconds since first transaction)
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # Log-scaled amount reduces the heavy right skew typical of transaction amounts
    df["amount_log"] = np.log1p(df["Amount"])

    logger.info("Fraud feature engineering done | total columns=%d", df.shape[1])
    return df


# ============================================================
# FEATURE TYPE DETECTION (shared by both modules)
# ============================================================

def detect_feature_types(df: pd.DataFrame, target_col: str, threshold: int = 10):
    """
    Auto-detects ordinal / continuous / binary columns, excluding
    the given target column. Shared by credit_engine and fraud_engine
    so both use the identical dual-ColumnTransformer preprocessing path.
    """
    ordinal_cols, continuous_cols, binary_cols = [], [], []

    for col in df.columns:
        if col == target_col:
            continue

        dtype_name = df[col].dtype.name
        n_unique   = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            ordinal_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types | ordinal=%d continuous=%d binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols


# ============================================================
# CONVENIENCE LOADERS
# ============================================================

def load_credit_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = validate_credit_data(df)
    df = add_credit_engineered_features(df)
    return df


def load_fraud_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = validate_fraud_data(df)
    df = add_fraud_engineered_features(df)
    return df


# ============================================================
# DEMO-ONLY MINORITY AUGMENTATION
# ============================================================
# The fraud sample shipped in this repo (sample_transactions.csv,
# 100 rows) is the same small sample committed in the reference
# Fraud Detection ZIP — it has only 1 fraud row, which is too few
# to run a 3-way stratified split (train_fit / calibration / test)
# or SMOTENC. In PRODUCTION, point train_fraud_model() at the full
# ~284,807-row Kaggle "Credit Card Fraud Detection" dataset (492
# frauds), where this function is a no-op (min_minority default is
# already satisfied). This function exists purely so the shipped
# repo sample can exercise the FULL pipeline end-to-end in CI/local
# demos without requiring that external download.
# ============================================================

def augment_minority_for_demo(
    df: pd.DataFrame,
    target_col: str = "Class",
    min_minority: int = 20,
    noise_scale: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    If the minority class has fewer than `min_minority` rows, bootstrap
    it up to `min_minority` by resampling-with-replacement and adding
    small Gaussian jitter (scaled to each numeric column's std) to the
    duplicated rows. The target label is never altered. This is a
    clearly-labeled DEMO/TEST utility, not a modeling technique — it
    does not substitute for SMOTE at train time and must not be used
    to inflate reported model metrics; it only exists to let the
    pipeline *run* on a tiny sample.
    """
    rng = np.random.RandomState(random_state)
    counts = df[target_col].value_counts()
    minority_class = counts.idxmin()
    minority_count = int(counts.min())

    if minority_count >= min_minority:
        return df

    n_needed = min_minority - minority_count
    minority_rows = df[df[target_col] == minority_class]
    numeric_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]

    sampled = minority_rows.sample(n=n_needed, replace=True, random_state=random_state).reset_index(drop=True)
    for col in numeric_cols:
        col_std = df[col].std() or 1.0
        sampled[col] = sampled[col] + rng.normal(0, noise_scale * col_std, size=len(sampled))

    logger.warning(
        "[DEMO ONLY] Minority class '%s' has only %d rows (< %d) — bootstrapped "
        "%d jittered synthetic rows so the pipeline can run end-to-end on this "
        "sample. Use the full production dataset for real training.",
        minority_class, minority_count, min_minority, n_needed
    )

    return pd.concat([df, sampled], ignore_index=True)
