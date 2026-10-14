
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


NUMERIC_FEATURES = [
    "model_year",
    "car_age",
    "mileage",
    "engine_liters",
    "horsepower",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "accident",
    "clean_title",
]


def make_preprocessor():
    """
    Create a preprocessing pipeline.

    Numerical columns:
        Missing values -> Median
        Scaling -> StandardScaler

    Categorical columns:
        Missing values -> Most frequent
        Encoding -> OneHotEncoder
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )
