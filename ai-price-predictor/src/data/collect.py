
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "Carson-Shively/used-car-price"


def download_raw_data(output_path: Path) -> None:
    """
    Download used-car listings from Hugging Face.

    Save the original records without cleaning them.
    """

    print(f"Downloading: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(f"Downloaded {len(dataset):,} records")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_json(
        str(output_path),
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"Saved raw data to: {output_path}")


if __name__ == "__main__":
    output = Path("data/raw/used_cars.jsonl")
    download_raw_data(output)
