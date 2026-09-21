from __future__ import annotations


def mock_translate(text: str, *, target_language: str) -> str:
    """Echo with a tag so the API contract can be tested without a GPU."""
    cleaned = (text or "").strip()
    return f"[mock:{target_language}] {cleaned}"
