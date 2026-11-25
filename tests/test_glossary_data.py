"""The shipped data/glossary.json: radical names and the words deliberately left out."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.glossary import Glossary

GLOSSARY = Glossary(Path(__file__).resolve().parents[1] / "data" / "glossary.json")


@pytest.mark.parametrize(
    "text, source, target, expected",
    [
        ("にんべん", "ja", "vi", "bộ Nhân đứng (亻)"),  # model: "Linh dương"
        ("「にんべん」", "ja", "vi", "bộ Nhân đứng (亻)"),
        ("にんべん", "ja", "en", "person radical (亻)"),  # model: "Ninben"
        ("人偏", "zh", "vi", "bộ Nhân đứng (亻)"),  # kana-less text is detected as zh
        ("さんずい", "ja", "vi", "bộ Thủy (氵)"),
        ("くさかんむり", "ja", "vi", "bộ Thảo (艹)"),
        ("テーマ", "ja", "vi", "Chủ đề"),
    ],
)
def test_radical_names(text, source, target, expected):
    assert GLOSSARY.lookup(text, source_language=source, target_language=target) == expected


@pytest.mark.parametrize("word", ["れんが", "なべぶた", "しんにゅう", "りっとう", "かいへん", "頁", "下心"])
def test_ordinary_words_are_not_radicals(word):
    for source in ("ja", "zh"):
        assert GLOSSARY.lookup(word, source_language=source, target_language="vi") is None
