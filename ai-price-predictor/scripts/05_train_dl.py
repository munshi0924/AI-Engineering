
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.models.neural_network import CarPriceNetwork
from src.evaluation.metrics import regression_metrics


DATA_DIR = ROOT / "data/processed"
MODEL_DIR = ROOT / "artifacts/models"
REPORT_DIR = ROOT / "data/reports"

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 12
SEED = 42


def prepare_tensors(df, preprocessor):
    """
    Transform car features into PyTorch tensors.
    """
    X = preprocessor.transform(
        df[FEATURE_COLUMNS]
    )

    # One-hot encoding may produce a sparse matrix.
    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    # Target: log(price + 1)
    y = np.log1p(
        df["price"].to_numpy(dtype=np.float32)
    )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).reshape(-1, 1)

    return X_tensor, y_tensor


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Select hardware
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Load the same data splits as Phase 3
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")

    # Load the preprocessor fitted in Phase 2
    preprocessor = joblib.load(
        ROOT / "artifacts/preprocessors/preprocessor.joblib"
    )

    X_train, y_train = prepare_tensors(
        train_df, preprocessor
    )

    X_val, y_val = prepare_tensors(
        val_df, preprocessor
    )

    print("Training features:", X_train.shape)
    print("Validation features:", X_val.shape)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Build the network
    model = CarPriceNetwork(
        input_size=X_train.shape[1]
    ).to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):

        # TRAINING
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_function(
                predictions,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(X_batch)
            )

        average_loss = (
            total_loss / len(X_train)
        )

        # VALIDATION
        model.eval()

        with torch.no_grad():
            val_predictions_log = model(
                X_val.to(device)
            ).cpu().numpy().flatten()

        # Convert predictions back into dollars.
        # Clipping prevents exponential overflow.
        val_predictions = np.maximum(
            np.expm1(
                np.clip(val_predictions_log, -20, 20)
            ),
            0,
        )

        actual_prices = np.expm1(
            y_val.numpy().flatten()
        )

        metrics = regression_metrics(
            actual_prices,
            val_predictions,
        )

        val_mae = metrics["mae"]

        history.append({
            "epoch": epoch,
            "training_log_mse": average_loss,
            "validation_mae": val_mae,
            "validation_rmse": metrics["rmse"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Val MAE: ${val_mae:,.2f}"
        )

        # Save best model using validation MAE
        if val_mae < best_mae:
            best_mae = val_mae
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": X_train.shape[1],
                    "epoch": epoch,
                    "validation_mae": best_mae,
                },
                MODEL_DIR / "best_neural_network.pt",
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(
        REPORT_DIR / "dl_training_history.csv",
        index=False,
    )

    print(
        f"\nBest validation MAE: ${best_mae:,.2f}"
    )


if __name__ == "__main__":
    main()
