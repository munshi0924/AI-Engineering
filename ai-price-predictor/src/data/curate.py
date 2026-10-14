
import re
from datetime import datetime

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "brand",
    "model",
    "model_year",
    "milage",
    "price",
]


def parse_number(value):
    """
    Convert values such as:
    '$12,500' -> 12500.0
    '65,000 mi.' -> 65000.0

    Return NaN for invalid values.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    # Remove common formatting characters.
    text = re.sub(
        r"[$,\s]",
        "",
        text,
    )

    # Remove a trailing mileage unit.
    text = re.sub(
        r"(mi\.?|miles?)$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    try:
        return float(text)
    except ValueError:
        return np.nan


def curate_cars(df: pd.DataFrame):
    """
    Clean and validate raw car listings.

    Returns:
        curated: Accepted listings
        rejected: Rejected listings with reasons
    """
    df = df.copy()

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Keep an identifier linking records
    # to their original raw-data position.
    df["source_row_id"] = np.arange(len(df))

    # Standardize column names.
    df = df.rename(
        columns={"milage": "mileage"}
    )

    # Convert numerical fields.
    df["price"] = df["price"].map(parse_number)
    df["mileage"] = df["mileage"].map(parse_number)

    df["model_year"] = pd.to_numeric(
        df["model_year"],
        errors="coerce",
    )

    # Clean brand and model strings.
    for column in ["brand", "model"]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # Create validation masks.
    current_year = datetime.now().year

    invalid_price = (
        df["price"].isna()
        | ~np.isfinite(df["price"])
        | (df["price"] <= 0)
    )

    invalid_mileage = (
        df["mileage"].isna()
        | ~np.isfinite(df["mileage"])
        | (df["mileage"] < 0)
    )

    invalid_year = (
        df["model_year"].isna()
        | ~np.isfinite(df["model_year"])
        | (df["model_year"] < 1886)
        | (df["model_year"] > current_year + 1)
        | (df["model_year"] % 1 != 0)
    )

    missing_identity = (
        df["brand"].isna()
        | df["model"].isna()
        | df["brand"].eq("").fillna(True)
        | df["model"].eq("").fillna(True)
    )

    # Assign a primary rejection reason.
    df["rejection_reason"] = pd.NA

    rules = [
        (invalid_price, "invalid_price"),
        (invalid_mileage, "invalid_mileage"),
        (invalid_year, "invalid_model_year"),
        (missing_identity, "missing_brand_or_model"),
    ]

    for mask, reason in rules:
        df.loc[
            mask & df["rejection_reason"].isna(),
            "rejection_reason",
        ] = reason

    rejected = df[
        df["rejection_reason"].notna()
    ].copy()

    curated = df[
        df["rejection_reason"].isna()
    ].copy()

    # Remove exact duplicate car records,
    # excluding bookkeeping columns.
    comparison_columns = [
        col
        for col in curated.columns
        if col not in [
            "source_row_id",
            "rejection_reason",
        ]
    ]

    duplicates = curated.duplicated(
        subset=comparison_columns,
        keep="first",
    )

    duplicate_rows = curated.loc[duplicates].copy()
    duplicate_rows["rejection_reason"] = "duplicate_record"

    rejected = pd.concat(
        [rejected, duplicate_rows],
        ignore_index=True,
    )

    curated = curated.loc[~duplicates].copy()

    curated = curated.drop(
        columns=["rejection_reason"]
    )

    return curated, rejected
