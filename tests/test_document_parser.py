# ============================================================
# PYTEST — synthetic_data_gen + document_parser round trip
# ============================================================
# These tests actually generate a synthetic PDF and verify the
# parser recovers the SAME numbers the generator produced —
# proving Module 3's extraction pipeline works end-to-end, not
# just that its functions run without raising.
# ============================================================

import sys
import os
import shutil
import json
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest

from src.synthetic_data_gen import generate_synthetic_statements
from src.document_parser import parse_bank_statement

TEST_STATEMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic_statements", "_pytest_tmp")


@pytest.fixture(scope="module")
def demo_statements():
    os.makedirs(TEST_STATEMENTS_DIR, exist_ok=True)
    ids = generate_synthetic_statements(n=3, out_dir=TEST_STATEMENTS_DIR, seed=123)
    yield ids
    shutil.rmtree(TEST_STATEMENTS_DIR, ignore_errors=True)


class TestSyntheticStatementRoundTrip:

    def test_pdf_and_json_created_for_each_id(self, demo_statements):
        for app_id in demo_statements:
            assert os.path.exists(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
            assert os.path.exists(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_ground_truth.json"))

    def test_income_extraction_matches_ground_truth(self, demo_statements):
        for app_id in demo_statements:
            gt = json.load(open(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_ground_truth.json")))
            res = parse_bank_statement(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
            assert abs(res.avg_monthly_income - gt["avg_monthly_income"]) < 1.0

    def test_employer_extracted_via_regex(self, demo_statements):
        for app_id in demo_statements:
            gt = json.load(open(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_ground_truth.json")))
            res = parse_bank_statement(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
            assert res.employer is not None
            assert res.employer.lower() == gt["employer"].lower()
            assert res.employer_source == "regex"

    def test_income_stability_label_matches(self, demo_statements):
        for app_id in demo_statements:
            gt = json.load(open(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_ground_truth.json")))
            res = parse_bank_statement(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
            assert res.income_stability == gt["income_stability"]

    def test_no_warnings_on_clean_synthetic_statement(self, demo_statements):
        for app_id in demo_statements:
            res = parse_bank_statement(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
            assert res.warnings == []

    def test_transactions_parsed_is_positive(self, demo_statements):
        for app_id in demo_statements:
            res = parse_bank_statement(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
            assert res.n_transactions_parsed > 0

    def test_to_dict_is_json_serializable(self, demo_statements):
        app_id = demo_statements[0]
        res = parse_bank_statement(os.path.join(TEST_STATEMENTS_DIR, f"{app_id}_statement.pdf"))
        json.dumps(res.to_dict())   # must not raise
