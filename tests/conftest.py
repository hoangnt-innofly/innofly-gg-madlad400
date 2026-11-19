import os
import sys
from pathlib import Path

import pytest

# Run from anywhere: make the repo root importable as the `app` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def token_len():
    """Token counter for '<2xx> text': the real MADLAD spiece.model when MADLAD_SPIECE
    points at it (pip install sentencepiece), else a rough length-based estimate."""
    path = os.environ.get("MADLAD_SPIECE", "")
    if path and Path(path).is_file():
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=path)
        return lambda s: len(sp.encode("<2en> " + s)) + 1
    return lambda s: len(s) // 2 + 4


@pytest.fixture(scope="session")
def para_ja() -> str:
    """The reported paragraph: a title line + 7 Japanese sentences."""
    return (DATA / "para_ja.txt").read_text(encoding="utf-8").strip()
