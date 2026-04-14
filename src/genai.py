"""GenAI utilities for personalised customer recovery emails.

Uses the OpenAI Python SDK (gpt-4o-mini) to generate emails and order summaries
from rows in the intervention payload produced by src/model.py.

Required environment variable
------------------------------
OPENAI_API_KEY  — your OpenAI secret key
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Look-up tables
# ---------------------------------------------------------------------------

_PAYMENT_LABELS: dict[str, str] = {
    "credit_card": "credit card",
    "boleto": "boleto",
    "voucher": "voucher",
    "debit_card": "debit card",
}

_TONE_DESCRIPTIONS: dict[str, str] = {
    "empathetic": (
        "warm, sincere, and apologetic — the customer should feel genuinely heard "
        "and valued"
    ),
    "formal": "professional, concise, and respectful — keep it brief and to the point",
    "friendly": "casual, upbeat, and positive — make the customer smile while still being clear",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Return a configured OpenAI client, raising clear errors on misconfiguration."""
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "openai is not installed. Run: pip install 'openai>=1.0.0'"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running the app:\n"
            "  export OPENAI_API_KEY='your_key_here'"
        )
    return OpenAI(api_key=api_key)


def _fmt_payment(raw: object) -> str:
    """Convert a snake_case payment type to a human-readable label."""
    return _PAYMENT_LABELS.get(str(raw).lower(), str(raw).replace("_", " "))


def _fmt_category(raw: object) -> str:
    """Convert a snake_case product category to Title Case with spaces."""
    return str(raw).replace("_", " ").title() if raw else "your recent order"


def _int_val(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _float_val(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recovery_email(
    customer_row: dict,
    tone: str = "empathetic",
) -> str:
    """Generate a personalised recovery email for one at-risk customer.

    Parameters
    ----------
    customer_row:
        A single row from ``intervention_payload_high_risk_late.csv`` as a dict.
        Expected keys: ``customer_state``, ``payment_type``,
        ``payment_installments``, ``product_category``, ``delay_days``,
        ``total_order_value``, ``recommended_coupon_pct``.
    tone:
        Writing style — ``"empathetic"`` (default), ``"formal"``, or
        ``"friendly"``.

    Returns
    -------
    str
        The email body only (no subject line).
    """
    client = _get_client()

    tone_desc = _TONE_DESCRIPTIONS.get(tone.lower(), _TONE_DESCRIPTIONS["empathetic"])
    payment_label = _fmt_payment(customer_row.get("payment_type", ""))
    installments = _int_val(customer_row.get("payment_installments", 1), default=1)
    category = _fmt_category(customer_row.get("product_category", ""))
    state = customer_row.get("customer_state") or "Brazil"
    delay_days = _int_val(customer_row.get("delay_days", 0))
    order_value = _float_val(customer_row.get("total_order_value", 0.0))
    coupon_pct = _int_val(customer_row.get("recommended_coupon_pct", 10), default=10)

    installment_phrase = (
        f"in {installments} installments"
        if installments > 1
        else "in a single payment"
    )

    system_prompt = (
        "You are a customer success specialist at Olist, Brazil's largest "
        "e-commerce marketplace. Your role is to write personalised recovery emails "
        "to customers whose orders arrived late. "
        f"Write in a tone that is {tone_desc}. "
        "Write in English. Return only the email body — no subject line, no metadata."
    )

    user_prompt = (
        f"Write a recovery email for a customer with these order details:\n\n"
        f"- State: {state}\n"
        f"- Product category: {category}\n"
        f"- Payment: {payment_label} {installment_phrase}\n"
        f"- Order value: R$ {order_value:.2f}\n"
        f"- Days late: {delay_days} day{'s' if delay_days != 1 else ''}\n"
        f"- Discount offer: {coupon_pct}% off their next purchase\n\n"
        "The email must include:\n"
        "1. A personalised opening referencing their specific order "
        "(product category, payment method, and how many days late it was)\n"
        "2. A genuine apology for the late delivery\n"
        f"3. The discount offer ({coupon_pct}% off their next purchase) "
        "with a clear call to action\n"
        "4. A warm sign-off from the Olist Customer Success team\n\n"
        "Keep the email under 150 words."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=350,
    )
    return response.choices[0].message.content.strip()


def build_customer_summary(customer_row: dict) -> str:
    """Return a short human-readable bullet summary of a customer's order.

    Intended for display in the Streamlit app's info box above the email.

    Parameters
    ----------
    customer_row:
        A single row from the intervention payload as a dict.

    Returns
    -------
    str
        3–4 bullet points as a plain string (markdown-compatible).
    """
    order_value = _float_val(customer_row.get("total_order_value", 0.0))
    delay_days = _int_val(customer_row.get("delay_days", 0))
    risk_pct = _float_val(customer_row.get("risk_pct", 0.0))
    category = _fmt_category(customer_row.get("product_category", ""))
    payment_label = _fmt_payment(customer_row.get("payment_type", ""))
    installments = _int_val(customer_row.get("payment_installments", 1), default=1)
    state = customer_row.get("customer_state") or "—"

    installment_phrase = (
        f"{installments} installments" if installments > 1 else "single payment"
    )

    return "\n".join(
        [
            f"• **Order value:** R$ {order_value:,.2f}  |  **State:** {state}",
            f"• **Days late:** {delay_days} day{'s' if delay_days != 1 else ''}",
            f"• **Churn risk score:** {risk_pct:.1f}%",
            f"• **Product:** {category}  |  **Payment:** {payment_label} ({installment_phrase})",
        ]
    )
