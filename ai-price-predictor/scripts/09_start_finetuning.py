
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(override=True)

TRAIN_PATH = (
    ROOT / "data/fine_tuning/train.jsonl"
)

VAL_PATH = (
    ROOT / "data/fine_tuning/validation.jsonl"
)


def main():
    model_name = os.getenv(
        "FINETUNE_BASE_MODEL"
    )

    if (
        not model_name
        or model_name == "your_supported_fine_tuning_model"
    ):
        raise ValueError(
            "Set FINETUNE_BASE_MODEL to a "
            "fine-tuning-supported model first."
        )

    client = OpenAI()

    print("Uploading training data...")

    with open(TRAIN_PATH, "rb") as file:
        train_file = client.files.create(
            file=file,
            purpose="fine-tune",
        )

    print("Uploading validation data...")

    with open(VAL_PATH, "rb") as file:
        val_file = client.files.create(
            file=file,
            purpose="fine-tune",
        )

    print("Training file ID:", train_file.id)
    print("Validation file ID:", val_file.id)

    print("\nStarting fine-tuning job...")

    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=model_name,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": 3
                }
            },
        },
    )

    print("Job ID:", job.id)
    print("Status:", job.status)

    output_dir = ROOT / "artifacts/models"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "fine_tuning_job.txt").write_text(
        job.id,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
