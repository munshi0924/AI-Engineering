
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.curate import curate_cars


RAW_PATH = ROOT / "data/raw/used_cars.jsonl"
OUTPUT_DIR = ROOT / "data/curated"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading raw dataset...")

    raw = pd.read_json(
        RAW_PATH,
        lines=True,
    )

    print(f"Raw records: {len(raw):,}")

    print("Cleaning and validating records...")

    curated, rejected = curate_cars(raw)

    curated_path = (
        OUTPUT_DIR / "used_cars_curated.csv"
    )

    rejected_path = (
        OUTPUT_DIR / "rejected_records.csv"
    )

    curated.to_csv(
        curated_path,
        index=False,
    )

    rejected.to_csv(
        rejected_path,
        index=False,
    )

    print("\n=== CURATION SUMMARY ===")
    print(f"Raw records:      {len(raw):,}")
    print(f"Accepted records: {len(curated):,}")
    print(f"Rejected records: {len(rejected):,}")

    if len(raw) > 0:
        acceptance_rate = (
            len(curated) / len(raw) * 100
        )
        print(
            f"Acceptance rate: {acceptance_rate:.2f}%"
        )

    print("\n=== REJECTION REASONS ===")
    print(
        rejected["rejection_reason"]
        .value_counts()
    )

    print("\n=== CURATED DATA SAMPLE ===")
    print(
        curated[
            [
                "brand",
                "model",
                "model_year",
                "mileage",
                "price",
            ]
        ].head()
    )

    print("\nFiles saved:")
    print(curated_path)
    print(rejected_path)


if __name__ == "__main__":
    main()
