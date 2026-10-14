
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics
from src.evaluation.predictors import (
    predict_traditional_ml,
    predict_neural_network,
)


TEST_PATH = ROOT / "data/processed/test.csv"
REPORT_DIR = ROOT / "data/reports"


def evaluate_predictions(
    model_name,
    actual,
    predictions,
    elapsed_seconds,
):
    """
    Evaluate one model's predictions.

    Failed predictions are excluded from error metrics
    but counted against prediction coverage.
    """
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if actual.shape != predictions.shape:
        raise ValueError("Actual/prediction shapes differ")

    valid = np.isfinite(predictions) & (predictions >= 0)

    coverage = float(valid.mean()) if len(valid) else 0.0

    result = {
        "model": model_name,
        "n_requested": len(actual),
        "n_successful": int(valid.sum()),
        "coverage": coverage,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_car": (
            elapsed_seconds / len(actual)
            if len(actual) else np.nan
        ),
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
    }

    if valid.sum() >= 2:
        metrics = regression_metrics(
            actual[valid],
            predictions[valid],
        )
        result.update(metrics)

    return result


def predict_llm_batch(
    descriptions,
    model_name,
):
    """
    Predict sequentially using the Phase 5 LLM client.

    Failed requests become NaN, not fabricated prices.
    """
    from src.models.frontier_llm import LLMPricePredictor

    predictor = LLMPricePredictor(model=model_name)

    predictions = []

    for i, description in enumerate(descriptions, 1):
        print(f"  LLM prediction {i}/{len(descriptions)}")

        try:
            price = predictor.predict(str(description))
        except Exception as exc:
            print(f"  Failed: {exc}")
            price = np.nan

        predictions.append(price)

    return np.asarray(predictions, dtype=float)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Enable paid LLM API predictions",
    )

    parser.add_argument(
        "--llm-limit",
        type=int,
        default=25,
        help="Maximum test cars sent to LLM APIs",
    )

    args = parser.parse_args()

    if args.llm_limit < 1:
        raise ValueError("--llm-limit must be positive")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_PATH)

    actual = test_df["price"].to_numpy(dtype=float)

    # Use a stable identifier for every test record.
    # The row position is sufficient for joining results.
    prediction_table = pd.DataFrame({
        "test_row": np.arange(len(test_df)),
        "actual_price": actual,
    })

    if "source_row_id" in test_df.columns:
        prediction_table["source_row_id"] = (
            test_df["source_row_id"].to_numpy()
        )

    comparison = []

    # MODEL 1: TRADITIONAL ML
    print("\nEvaluating traditional ML...")

    start = time.perf_counter()

    ml_predictions = predict_traditional_ml(
        test_df,
        ROOT,
    )

    elapsed = time.perf_counter() - start

    prediction_table["traditional_ml"] = ml_predictions

    comparison.append(
        evaluate_predictions(
            "traditional_ml",
            actual,
            ml_predictions,
            elapsed,
        )
    )

    # MODEL 2: PYTORCH
    print("\nEvaluating PyTorch neural network...")

    start = time.perf_counter()

    dl_predictions = predict_neural_network(
        test_df,
        ROOT,
    )

    elapsed = time.perf_counter() - start

    prediction_table["pytorch_mlp"] = dl_predictions

    comparison.append(
        evaluate_predictions(
            "pytorch_mlp",
            actual,
            dl_predictions,
            elapsed,
        )
    )

    # OPTIONAL: LLM MODELS
    if args.include_llm:
        if "llm_description" not in test_df.columns:
            raise ValueError(
                "test.csv has no llm_description column"
            )

        if test_df["llm_description"].isna().any():
            raise ValueError(
                "Some test descriptions are missing"
            )

        if not os.getenv("OPENAI_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not configured"
            )

        # A fixed, small test subset for API cost control.
        # IMPORTANT: Evaluate all compared models on
        # this same subset when reporting LLM rankings.
        subset = test_df.sample(
            n=min(args.llm_limit, len(test_df)),
            random_state=42,
        )

        subset_positions = subset.index.to_numpy()

        descriptions = subset["llm_description"].tolist()

        base_model_name = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.5",
        )

        llm_models = {
            "pretrained_llm": base_model_name,
        }

        ft_path = (
            ROOT / "artifacts/models/fine_tuned_model.txt"
        )

        if ft_path.exists():
            ft_name = ft_path.read_text(
                encoding="utf-8"
            ).strip()

            if ft_name:
                llm_models["fine_tuned_llm"] = ft_name

        for label, model_name in llm_models.items():
            print(f"\nEvaluating {label}: {model_name}")

            start = time.perf_counter()

            subset_predictions = predict_llm_batch(
                descriptions,
                model_name,
            )

            elapsed = time.perf_counter() - start

            full_predictions = np.full(
                len(test_df),
                np.nan,
            )

            full_predictions[
                subset_positions
            ] = subset_predictions

            prediction_table[label] = full_predictions

            # Evaluate on the requested subset,
            # not the entire test set.
            comparison.append(
                evaluate_predictions(
                    label,
                    actual[subset_positions],
                    subset_predictions,
                    elapsed,
                )
            )

    # Save all individual predictions
    prediction_table.to_csv(
        REPORT_DIR / "final_predictions.csv",
        index=False,
    )

    # Save summary metrics
    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        REPORT_DIR / "final_comparison.csv",
        index=False,
    )

    print("\n=== FINAL RESULTS ===")
    print(comparison_df.to_string(index=False))

    print("\nReports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
