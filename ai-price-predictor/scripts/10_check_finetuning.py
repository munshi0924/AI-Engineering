
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_DIR = ROOT / "artifacts/models"

JOB_ID_PATH = MODEL_DIR / "fine_tuning_job.txt"


def main():
    if not JOB_ID_PATH.exists():
        raise FileNotFoundError(
            "No saved fine-tuning job ID."
        )

    job_id = JOB_ID_PATH.read_text(
        encoding="utf-8"
    ).strip()

    client = OpenAI()

    job = client.fine_tuning.jobs.retrieve(
        job_id
    )

    print("Job ID:", job.id)
    print("Status:", job.status)

    if job.status == "succeeded":
        model_name = job.fine_tuned_model

        print("Fine-tuned model:", model_name)

        (MODEL_DIR / "fine_tuned_model.txt").write_text(
            model_name,
            encoding="utf-8",
        )

    elif job.status == "failed":
        print("Training failed:", job.error)

    else:
        print("Training is not complete.")


if __name__ == "__main__":
    main()
