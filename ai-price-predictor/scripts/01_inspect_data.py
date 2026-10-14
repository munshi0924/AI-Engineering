
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RAW_FILE = Path("data/raw/used_cars.jsonl")
REPORT_DIR = Path("data/reports")


def inspect_data():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Load the raw data
    df = pd.read_json(
        RAW_FILE,
        lines=True,
    )

    # 2. Inspect the dataset
    print("\n=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== FIRST FIVE RECORDS ===")
    print(df.head().to_string())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    # 3. Find missing values
    missing = df.isna().sum()

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing.values,
        "missing_percent": (
            missing.values / len(df) * 100
        ).round(2),
    })

    print("\n=== MISSING VALUE REPORT ===")
    print(report.to_string(index=False))

    report.to_csv(
        REPORT_DIR / "missing_values.csv",
        index=False,
    )

    # 4. Count exact duplicates
    duplicates = df.astype(str).duplicated().sum()

    print("\n=== DUPLICATES ===")
    print(f"Exact duplicate rows: {duplicates}")

    # 5. Inspect price values
    print("\n=== RAW PRICE EXAMPLES ===")
    print(df["price"].head(10).tolist())

    # Create numeric prices for exploration only.
    # The raw JSONL file remains unchanged.
    numeric_prices = pd.to_numeric(
        df["price"]
        .astype("string")
        .str.replace(r"[$,]", "", regex=True)
        .str.strip(),
        errors="coerce",
    )

    print("\n=== PRICE STATISTICS ===")
    print(numeric_prices.describe())

    print("\n=== INVALID PRICES ===")
    print(numeric_prices.isna().sum())

    # 6. Plot a histogram
    valid_prices = numeric_prices[
        numeric_prices.between(500, 150000)
    ]

    if not valid_prices.empty:
        plt.figure(figsize=(10, 6))

        plt.hist(
            valid_prices,
            bins=40,
            edgecolor="black",
        )

        plt.title("Used Car Price Distribution")
        plt.xlabel("Listing Price (USD)")
        plt.ylabel("Number of Cars")
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / "price_distribution.png"
        )

        plt.close()

    print("\nInspection complete.")


if __name__ == "__main__":
    inspect_data()
