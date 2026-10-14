
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

history = pd.read_csv(
    ROOT / "data/reports/dl_training_history.csv"
)

plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["validation_mae"],
    marker="o",
    markersize=3,
)

plt.xlabel("Epoch")
plt.ylabel("Validation MAE ($)")
plt.title("Neural Network Validation Performance")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "data/reports/dl_learning_curve.png"
)

plt.show()
