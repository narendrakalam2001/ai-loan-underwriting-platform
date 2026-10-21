# ============================================================
# DATA SIMULATOR — AI Loan Underwriting Platform
# ============================================================
# Generates synthetic loan applications and sends them to the
# live /underwrite endpoint — useful for demoing the platform,
# generating traffic for the monitoring dashboard, and load
# sanity-checking the API. Follows the same pattern as both
# reference projects' simulators.
#
# Scenarios:
#   safe    — low-risk applicant profile (should mostly APPROVE)
#   risky   — high-risk applicant profile (should mostly DECLINE)
#   random  — mixed, realistic distribution of applications
# ============================================================

import os
import time
import random
import logging

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_API_BASE = os.getenv("UNDERWRITING_API_URL", "http://127.0.0.1:8000")
# 0.0.0.0 is a valid address for a SERVER to bind to ("listen on all interfaces"),
# but it is NOT a valid destination for a client on Windows (WinError 10049).
# Map it to loopback so a copy-pasted "http://0.0.0.0:8000" still works.
_API_BASE = _API_BASE.replace("0.0.0.0", "127.0.0.1").rstrip("/")
API_URL = _API_BASE + "/underwrite"

LOAN_PURPOSES = ["Personal", "Home Renovation", "Education", "Medical", "Business", "Wedding"]


# ============================================================
# GENERATE SYNTHETIC APPLICATION
# ============================================================

def generate_application(scenario: str = "random") -> dict:
    """
    Scenarios (calibrated to Kaggle "Give Me Some Credit"'s actual ranges —
    monthly_income median ≈ ₹5,400, mean ≈ ₹6,670 in the source dataset;
    scaled up here to plausible Indian monthly-income figures):
        safe    — high income, low utilization/debt, clean payment history
        risky   — low income, high utilization/debt, delinquency history
        random  — mixed realistic distribution
    """
    if scenario == "safe":
        base = {
            "age": random.randint(35, 60),
            "monthly_income": random.uniform(60_000, 150_000),
            "dependents": random.randint(0, 2),
            "existing_emis": random.uniform(0, 8_000),
            "credit_utilization_pct": random.uniform(5, 30),
            "open_credit_lines": random.randint(2, 6),
            "real_estate_loans": random.choice([0, 0, 1]),
            "times_30_59_days_late": 0,
            "times_60_89_days_late": 0,
            "times_90_days_late": 0,
            "loan_amount_requested": random.uniform(200_000, 800_000),
        }

    elif scenario == "risky":
        base = {
            "age": random.randint(22, 32),
            "monthly_income": random.uniform(10_000, 25_000),
            "dependents": random.randint(3, 6),
            "existing_emis": random.uniform(8_000, 20_000),
            "credit_utilization_pct": random.uniform(80, 150),
            "open_credit_lines": random.randint(4, 10),
            "real_estate_loans": 0,
            "times_30_59_days_late": random.randint(1, 4),
            "times_60_89_days_late": random.randint(0, 2),
            "times_90_days_late": random.randint(0, 3),
            "loan_amount_requested": random.uniform(800_000, 3_000_000),
        }

    else:  # random
        base = {
            "age": random.randint(23, 65),
            "monthly_income": random.uniform(15_000, 120_000),
            "dependents": random.randint(0, 5),
            "existing_emis": random.uniform(0, 15_000),
            "credit_utilization_pct": random.uniform(5, 120),
            "open_credit_lines": random.randint(1, 8),
            "real_estate_loans": random.choice([0, 0, 0, 1]),
            "times_30_59_days_late": random.choice([0, 0, 0, 1, 2]),
            "times_60_89_days_late": random.choice([0, 0, 0, 1]),
            "times_90_days_late": random.choice([0, 0, 0, 0, 1]),
            "loan_amount_requested": random.uniform(100_000, 1_500_000),
        }

    base.update({
        "full_name": f"Simulated Applicant {random.randint(1000, 9999)}",
        "employment_type": random.choice(["Salaried", "Self-Employed", "Business Owner"]),
        "years_employment": round(random.uniform(0.5, 20.0), 1),
        "loan_purpose": random.choice(LOAN_PURPOSES),
        "tenure_months": random.choice([12, 24, 36, 48, 60]),
    })

    return base


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_application(application: dict, idx: int):
    try:
        response = requests.post(API_URL, json=application, timeout=15)

        if response.status_code == 200:
            result = response.json()
            credit = result["module_outputs"]["credit_risk"]
            fraud = result["module_outputs"]["fraud_detection"]
            logger.info(
                "[%3d] income=₹%.0f loan=₹%.0f dependents=%d → credit=%.3f(%s) fraud=%s → %s (conf=%.2f)",
                idx + 1, application["monthly_income"], application["loan_amount_requested"],
                application["dependents"], credit["default_probability"], credit["risk_band"],
                fraud["verdict"], result["final_verdict"], result["confidence_score"],
            )
        else:
            logger.error("[%3d] API error: HTTP %d — %s", idx + 1, response.status_code, response.text[:200])

    except requests.exceptions.RequestException as e:
        logger.error("[%3d] Connection error: %s", idx + 1, e)


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_applications(n: int = 30, scenario: str = "random", delay: float = 0.3):
    logger.info("Simulating %d loan applications | scenario=%s | target=%s", n, scenario, API_URL)
    logger.info("-" * 70)

    for i in range(n):
        app = generate_application(scenario)
        send_application(app, i)
        time.sleep(delay)

    logger.info("-" * 70)
    logger.info("Simulation complete")


def simulate_all_scenarios(n_per_scenario: int = 15):
    """Runs all 3 scenarios in sequence — good for populating the ops dashboard demo data."""
    for scenario in ("safe", "risky", "random"):
        simulate_applications(n=n_per_scenario, scenario=scenario)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate loan applications against the live API")
    parser.add_argument("--n", type=int, default=30, help="Number of applications to simulate")
    parser.add_argument("--scenario", choices=["safe", "risky", "random", "all"], default="random")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
    args = parser.parse_args()

    if args.scenario == "all":
        simulate_all_scenarios(n_per_scenario=max(1, args.n // 3))
    else:
        simulate_applications(n=args.n, scenario=args.scenario, delay=args.delay)
