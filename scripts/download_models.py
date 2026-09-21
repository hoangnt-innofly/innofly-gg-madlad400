"""Download google/madlad400-3b-mt weights into ./models."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPO_ID = "google/madlad400-3b-mt"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODELS / "madlad400-3b-mt"),
    )
    print(path)


if __name__ == "__main__":
    main()
