
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import engineer_features
from src.data.preprocess import (
    make_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


INPUT_PATH = (
    ROOT / "data/curated/used_cars_curated.csv"
)

OUTPUT_DIR = ROOT / "data/processed"
ARTIFACT_DIR = ROOT / "artifacts/preprocessors"


def create_llm_description(row):
    """
    Build a natural-language car description
    without revealing its price.
    """

    def readable(value):
        return (
            "unknown"
            if pd.isna(value) or str(value).strip() == ""
            else str(value)
        )

    return (
        f"A {readable(row['model_year'])} "
        f"{readable(row['brand'])} "
        f"{readable(row['model'])} with "
        f"{readable(row['mileage'])} miles. "
        f"Fuel type: {readable(row.get('fuel_type'))}. "
        f"Transmission: {readable(row.get('transmission'))}. "
        f"Engine: {readable(row.get('engine'))}. "
        f"Accident history: {readable(row.get('accident'))}. "
        f"Title status: {readable(row.get('clean_title'))}."
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Load curated data
    df = pd.read_csv(INPUT_PATH)

    # 2. Create new features
    df = engineer_features(df)

    # 3. Ensure optional columns exist
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = pd.NA

    # 4. Create LLM descriptions
    df["llm_description"] = df.apply(
        create_llm_description,
        axis=1,
    )

    # 5. Split data: 80% train, 10% val, 10% test
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=42,
    )

    # 6. Fit preprocessing on training data ONLY
    feature_columns = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    preprocessor = make_preprocessor()

    preprocessor.fit(
        train_df[feature_columns]
    )

    # 7. Save splits
    for name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        path = OUTPUT_DIR / f"{name}.csv"

        split_df.to_csv(
            path,
            index=False,
        )

        print(
            f"{name}: {len(split_df):,} rows"
        )

    # 8. Save fitted preprocessor
    joblib.dump(
        preprocessor,
        ARTIFACT_DIR / "preprocessor.joblib",
    )

    print("\nPreprocessing completed.")
    print("Data splits and preprocessor saved.")


if __name__ == "__main__":
    main()
