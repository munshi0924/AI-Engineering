
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.fine_tuning import (
    create_training_example,
    write_jsonl,
)


DATA_DIR = ROOT / "data/processed"
OUTPUT_DIR = ROOT / "data/fine_tuning"


def convert_split(input_file, output_file):
    df = pd.read_csv(input_file)

    required = {"llm_description", "price"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    examples = []

    for _, row in df.iterrows():
        description = row["llm_description"]
        price = row["price"]

        if pd.isna(description) or pd.isna(price):
            continue

        example = create_training_example(
            description,
            price,
        )

        examples.append(example)

    write_jsonl(examples, output_file)

    print(
        f"Saved {len(examples)} examples "
        f"to {output_file}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    convert_split(
        DATA_DIR / "train.csv",
        OUTPUT_DIR / "train.jsonl",
    )

    convert_split(
        DATA_DIR / "validation.csv",
        OUTPUT_DIR / "validation.jsonl",
    )


if __name__ == "__main__":
    main()
