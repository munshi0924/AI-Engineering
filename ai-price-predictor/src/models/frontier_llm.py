
import os
import re
import math

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.5")


def build_prompt(description: str) -> str:
    return (
        "Estimate the listing price of this used car "
        "in US dollars.\n"
        "Use only the information provided.\n"
        "Return one number, without a dollar sign, "
        "commas, or explanation.\n\n"
        f"Car description:\n{description}"
    )


def parse_price(response_text: str) -> float:
    """
    Accept:
        '14500'
        '$14,500'
        '14500.00'

    Reject explanations and ambiguous answers.
    """
    cleaned = response_text.strip()
    cleaned = cleaned.replace("$", "").replace(",", "")

    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        raise ValueError(
            f"Invalid LLM price response: {response_text!r}"
        )

    price = float(cleaned)

    if not math.isfinite(price) or price <= 0:
        raise ValueError("Price must be positive and finite")

    return price


class LLMPricePredictor:
    def __init__(self, model=MODEL_NAME):
        self.model = model
        self.client = OpenAI(timeout=30.0, max_retries=2)

    def predict(self, description: str) -> float:
        prompt = build_prompt(description)

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You estimate used-car listing prices. "
                "Return only a positive numeric price in USD. "
                "Do not include explanations."
            ),
            input=prompt,
            max_output_tokens=100,
            store=False,
        )

        return parse_price(response.output_text)
