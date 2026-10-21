# Knowledge Base — Module 5A (RAG / Policy Intelligence Engine)

Every `.txt` and `.pdf` file in this folder is picked up automatically by
`python scripts/ingest_knowledge_base.py` and indexed for retrieval by the
Loan Assistant Chatbot's `faq_retriever` tool (and optionally by
`decision_engine.py` to ground an explanation against policy text).

## Files shipped here (all SYNTHETIC — see disclosure below)

| File | What it is |
|---|---|
| `rbi_lending_guidelines_SYNTHETIC.txt` | A synthetic summary of general-knowledge lending principles (fair-practices style disclosures, KYC, grievance redressal) — **not the real RBI Master Circular** |
| `cibil_score_interpretation_SYNTHETIC.txt` | Synthetic CIBIL-band interpretation guide |
| `bank_credit_policy_SYNTHETIC.txt` | Synthetic internal credit policy for a fictitious bank |
| `loan_faqs_SYNTHETIC.txt` | 50 synthetic loan FAQ Q&A pairs |

## ⚠️ Important: these are placeholders, not real regulatory text

I could not download the actual RBI Master Circular PDF into this
repository — my build sandbox's network allowlist doesn't include
`rbi.org.in` (or any Kaggle/general-download domain), so I could not fetch
it the way I did for your training datasets. Writing out the real circular
from memory would risk citing incorrect/outdated regulatory text, which is
worse than not having it at all for something regulatory.

**To use the real RBI guidelines:**

1. Download the current RBI Master Direction on Loans and Advances from
   https://www.rbi.org.in (search "Master Direction – Reserve Bank of India
   (Loans and Advances) Directions") as a PDF.
2. Drop that PDF into this folder.
3. Re-run `python scripts/ingest_knowledge_base.py`.

The synthetic `rbi_lending_guidelines_SYNTHETIC.txt` will keep being
indexed alongside it unless you delete it — you may want to delete it once
you've added the real document, to avoid the chatbot citing synthetic
"guidance" next to real regulatory text.

## Adding your own policy documents

Any `.txt` or `.pdf` dropped here gets indexed the same way — e.g. your
actual bank's credit policy, an updated FAQ list, or product-specific
terms. No code changes needed, just re-run the ingest script.
