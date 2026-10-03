import os
import sys
import json
from typing import List, Optional

from backend.core.exceptions import CustomException
from backend.core.logger import logging

# Fields a bank statement can plausibly evidence. Everything else in the
# dataset (gender, occupation, mortgage_status, credit_card_utilization_pct,
# preferred_loan_term_months, existing_loans_count, months_credit_history)
# is not reliably visible in a checking-account statement and should come
# from a known-fields form instead, or be left as null for the model's
# imputer to handle.
EXTRACTABLE_FIELDS = {
    "annual_salary_gbp": (
        "Estimate annual salary in GBP from recurring salary/payroll deposits. "
        "If monthly salary deposits are visible, multiply by 12. Return a number or null."
    ),
    "on_time_payment_ratio_pct": (
        "Estimate the percentage (0-100) of recurring bill/direct-debit payments "
        "that went through successfully without being returned, bounced, or "
        "followed by an overdraft/insufficient-funds fee. Return a number or null."
    ),
    "rent_payment_monthly_gbp": (
        "If a recurring monthly payment to a landlord or letting agency is visible, "
        "return its amount in GBP. Otherwise return 0 or null."
    ),
    "rent_on_time_rate_pct": (
        "Estimate the percentage (0-100) of rent payments made on or before their "
        "due date, if due dates and payment dates for rent are inferable. Return a "
        "number or null."
    ),
    "utility_bills_on_time_pct": (
        "Estimate the percentage (0-100) of utility bill payments (electricity, gas, "
        "water, internet) made successfully without being returned or late. Return a "
        "number or null."
    ),
    "mobile_money_active": (
        "True if any mobile money service transactions (e.g. PayPal, Venmo, M-Pesa, "
        "Cash App) appear in the statement, else False."
    ),
    "mobile_money_tx_monthly": (
        "Count of mobile money transactions in this statement period. Return an "
        "integer or null."
    ),
}


def build_extraction_prompt(chunks: List[str]) -> str:
    context = "\n---\n".join(chunks)
    field_descriptions = "\n".join(
        f'- "{field}": {desc}' for field, desc in EXTRACTABLE_FIELDS.items()
    )

    return f"""You are extracting structured financial features from a bank statement for a credit scoring model.

Below are the most relevant excerpts from the statement:

{context}

Extract these fields. For each, use the excerpts above only — do not guess
beyond what the text supports. If a field cannot be determined from the
text, return null for it.

{field_descriptions}

Respond with ONLY a JSON object with exactly these keys: {list(EXTRACTABLE_FIELDS.keys())}.
No preamble, no markdown formatting, no explanation — just the raw JSON object."""


def extract_features_from_chunks(chunks: List[str]) -> dict:
    """
    Sends retrieved statement chunks to Claude and asks for a strict JSON
    object of extracted feature values. Any field the model can't support
    from the text comes back as null, which the caller merges with
    user-supplied known fields and ultimately with NaN for the model's
    imputer to fill in.
    """
    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        client = anthropic.Anthropic(api_key=api_key)

        prompt = build_extraction_prompt(chunks)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        # Defensive cleanup in case the model wraps the JSON in a code fence
        # despite being asked not to.
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        extracted = json.loads(raw_text)
        logging.info(f"Extracted fields from statement: {extracted}")
        return extracted

    except Exception as e:
        raise CustomException(e, sys)
