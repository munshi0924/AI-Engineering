
import re

import numpy as np
import pandas as pd


REFERENCE_YEAR = 2026


def extract_engine_size(value):
    """Extract engine displacement in liters."""
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*L\b",
        str(value),
        flags=re.IGNORECASE,
    )

    return float(match.group(1)) if match else np.nan


def extract_horsepower(value):
    """Extract horsepower from an engine string."""
    if pd.isna(value):
        return np.nan

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*HP\b",
        str(value),
        flags=re.IGNORECASE,
    )

    return float(match.group(1)) if match else np.nan


def engineer_features(df):
    """Create additional car features."""
    df = df.copy()

    df["car_age"] = (
        REFERENCE_YEAR - df["model_year"]
    )

    engine = (
        df["engine"]
        if "engine" in df.columns
        else pd.Series("", index=df.index)
    )

    df["engine_liters"] = engine.map(
        extract_engine_size
    )

    df["horsepower"] = engine.map(
        extract_horsepower
    )

    return df
