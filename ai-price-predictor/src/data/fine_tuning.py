
import json
import math


SYSTEM_INSTRUCTION = (
    "You estimate used-car listing prices in US dollars. "
    "Return only a positive numeric price. "
    "Do not include dollar signs, commas, or explanations."
)


def create_training_example(description, price):
    """
    Convert one labeled car into a chat training example.
    """
    if not isinstance(description, str):
        raise ValueError("Description must be text")

    description = description.strip()

    if not description:
        raise ValueError("Description cannot be empty")

    price = float(price)

    if not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid training price")

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": (
                    "Estimate the used-car listing price "
                    "in US dollars based only on this "
                    "description:\n\n"
                    + description
                ),
            },
            {
                "role": "assistant",
                "content": f"{price:.2f}",
            },
        ]
    }


def write_jsonl(examples, output_path):
    """
    Write one JSON object per line.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(
                json.dumps(example, ensure_ascii=False)
                + "\n"
            )
