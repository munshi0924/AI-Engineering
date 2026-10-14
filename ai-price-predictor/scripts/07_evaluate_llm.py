
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.frontier_llm import LLMPricePredictor
from src.evaluation.metrics import regression_metrics


load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data/processed/validation.csv"
REPORT_DIR = ROOT / "data/reports"

SAMPLE_SIZE = 5


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Fixed sample so we can reproduce the experiment.
    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42,
    )

    predictor = LLMPricePredictor(
        model=os.getenv("OPENAI_MODEL", "gpt-5.5")
    )

    results = []

    for index, row in sample.iterrows():
        description = row["llm_description"]
        actual_price = float(row["price"])

        print(f"\nPredicting car {index}...")

        try:
            predicted_price = predictor.predict(
                description
            )

            error = abs(actual_price - predicted_price)
            status = "success"

            print(f"Actual:    ${actual_price:,.2f}")
            print(f"Predicted: ${predicted_price:,.2f}")
            print(f"Error:     ${error:,.2f}")

        except Exception as exc:
            predicted_price = np.nan
            error = np.nan
            status = "failed"

            print(f"Prediction failed: {exc}")

        results.append({
            "source_row_id": row.get("source_row_id"),
            "description": description,
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "absolute_error": error,
            "status": status,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "llm_predictions.csv",
        index=False,
    )

    successful = results_df[
        results_df["status"] == "success"
    ]

    print(
        f"\nSuccessful predictions: "
        f"{len(successful)}/{len(results_df)}"
    )

    if len(successful) >= 2:
        metrics = regression_metrics(
            successful["actual_price"],
            successful["predicted_price"],
        )

        pd.DataFrame([{
            "model": predictor.model,
            "sample_size": len(results_df),
            "successful_predictions": len(successful),
            **metrics,
        }]).to_csv(
            REPORT_DIR / "llm_validation_metrics.csv",
            index=False,
        )

        print("\n=== VALIDATION METRICS ===")

        for name, value in metrics.items():
            print(f"{name}: {value:,.2f}")

    else:
        print("Not enough successful predictions to evaluate.")


if __name__ == "__main__":
    main()
