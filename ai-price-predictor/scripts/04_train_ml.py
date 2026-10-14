
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.models.traditional_ml import create_models
from src.evaluation.metrics import regression_metrics


DATA_DIR = ROOT / "data/processed"
MODEL_DIR = ROOT / "artifacts/models"
REPORT_DIR = ROOT / "data/reports"

PREPROCESSOR_PATH = (
    ROOT / "artifacts/preprocessors/preprocessor.joblib"
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Load the saved data splits
    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    # 2. Separate inputs from target
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["price"].to_numpy()

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["price"].to_numpy()

    # 3. Load the fitted preprocessor
    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # 4. Transform using the SAME preprocessor
    X_train_processed = preprocessor.transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    print(
        "Training feature shape:",
        X_train_processed.shape,
    )

    # 5. Train and evaluate each model
    results = []
    best_name = None
    best_model = None
    best_mae = float("inf")

    for name, model in create_models().items():
        print(f"\nTraining: {name}")

        # Random Forest requires dense input in some
        # environments; use dense data for all models
        # only when the transformed matrix is small.
        X_train_model = X_train_processed
        X_val_model = X_val_processed

        model.fit(
            X_train_model,
            y_train,
        )

        predictions = model.predict(
            X_val_model
        )

        # Car prices cannot be negative.
        predictions = np.maximum(
            predictions,
            0,
        )

        metrics = regression_metrics(
            y_val,
            predictions,
        )

        results.append({
            "model": name,
            **metrics,
        })

        print(
            f"MAE: ${metrics['mae']:,.2f}"
        )
        print(
            f"RMSE: ${metrics['rmse']:,.2f}"
        )
        print(
            f"R²: {metrics['r2']:.4f}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_model = model

    # 6. Save model comparison
    results_df = (
        pd.DataFrame(results)
        .sort_values("mae")
    )

    results_df.to_csv(
        REPORT_DIR / "ml_comparison.csv",
        index=False,
    )

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    # 7. Save the best model
    joblib.dump(
        best_model,
        MODEL_DIR / "best_ml_model.joblib",
    )

    # Save its name separately
    (MODEL_DIR / "best_ml_model.txt").write_text(
        best_name,
        encoding="utf-8",
    )

    print(
        f"\nBest model: {best_name}"
    )
    print(
        f"Validation MAE: ${best_mae:,.2f}"
    )


if __name__ == "__main__":
    main()
