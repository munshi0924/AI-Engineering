
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred):
    """
    Calculate price prediction metrics.

    Both inputs must contain prices in dollars.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction shapes do not match")

    if not np.isfinite(y_true).all():
        raise ValueError("Invalid actual prices")

    if not np.isfinite(y_pred).all():
        raise ValueError("Invalid predicted prices")

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }
