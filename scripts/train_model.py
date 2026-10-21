# ============================================================
# RUN TRAINING — runner script
# ============================================================
# Usage:
#   python scripts/train_model.py                 # train both engines (default sample paths)
#   python scripts/train_model.py --engine credit  # credit risk only
#   python scripts/train_model.py --engine fraud   # fraud only
#
#   # Point at your OWN full-size dataset files (no need to move/rename them):
#   python scripts/train_model.py --engine credit --credit-path "D:\path\to\Bank_Personal_Loan_Modelling.csv"
#   python scripts/train_model.py --engine fraud   --fraud-path  "D:\path\to\creditcard.csv"
#   python scripts/train_model.py --credit-path "..." --fraud-path "..."   # both, custom paths
# ============================================================

import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.training_pipeline import train_credit_model, train_fraud_model, run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the underwriting platform's models")
    parser.add_argument("--engine", choices=["credit", "fraud", "both"], default="both")
    parser.add_argument("--credit-path", default="data/raw/credit_risk_data.csv",
                         help="Path to the credit-risk CSV (default: the small sample shipped in the repo)")
    parser.add_argument("--fraud-path", default="data/raw/fraud_transactions_data.csv",
                         help="Path to the fraud CSV (default: the small sample shipped in the repo)")
    args = parser.parse_args()

    if args.engine == "credit":
        train_credit_model(data_path=args.credit_path)
    elif args.engine == "fraud":
        train_fraud_model(data_path=args.fraud_path)
    else:
        train_credit_model(data_path=args.credit_path)
        train_fraud_model(data_path=args.fraud_path)
