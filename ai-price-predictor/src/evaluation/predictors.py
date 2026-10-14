
from pathlib import Path

import joblib
import numpy as np
import torch

from src.data.preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.models.neural_network import CarPriceNetwork


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df, preprocessor):
    """
    Apply the exact preprocessor fitted on train.csv.
    """
    X = df[FEATURE_COLUMNS]
    return preprocessor.transform(X)


def predict_traditional_ml(df, root: Path):
    preprocessor = joblib.load(
        root / "artifacts/preprocessors/preprocessor.joblib"
    )

    model = joblib.load(
        root / "artifacts/models/best_ml_model.joblib"
    )

    X = prepare_features(df, preprocessor)

    predictions = model.predict(X)

    return np.maximum(
        np.asarray(predictions, dtype=float),
        0.0,
    )


def predict_neural_network(df, root: Path):
    preprocessor = joblib.load(
        root / "artifacts/preprocessors/preprocessor.joblib"
    )

    X = prepare_features(df, preprocessor)

    # PyTorch expects dense numeric tensors.
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    checkpoint_path = (
        root / "artifacts/models/best_neural_network.pt"
    )

    # Load only checkpoint weights and metadata.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if X.shape[1] != checkpoint["input_size"]:
        raise ValueError(
            "Input feature count does not match "
            "the saved neural network."
        )

    model = CarPriceNetwork(
        input_size=checkpoint["input_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        log_predictions = model(
            torch.from_numpy(X)
        ).numpy().flatten()

    # Reverse the log1p target transformation
    predictions = np.expm1(
        np.clip(log_predictions, -20, 20)
    )

    return np.maximum(predictions, 0.0)
