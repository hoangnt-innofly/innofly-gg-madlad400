import sys
from pathlib import Path

# Run from anywhere: make the repo root importable as the `app` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
