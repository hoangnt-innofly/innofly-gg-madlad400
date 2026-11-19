"""Fixed translations for terms the model keeps getting wrong.

File format (JSON, UTF-8). Keys are "<target>" (any source) or "<source>-><target>":

    {
      "en->vi": {"kite": "con diều", "kit": "bộ dụng cụ"},
      "vi": {"MADLAD": "MADLAD"}
    }

A glossary entry only fires when the *whole* segment (the full input, a whole line, a
heading label such as テーマ of "テーマ：…", or one translation unit of a line) matches
the key, case-insensitively and ignoring
surrounding quotes / trailing punctuation. The file is re-read when it changes, so
entries can be added without restarting the API.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.services.short_text import match_surface, normalize

logger = logging.getLogger("madlad400-mt-api")


class Glossary:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._bad_mtime: float | None = None
        self._entries: dict[tuple[str, str], dict[str, str]] = {}

    def lookup(self, text: str, *, source_language: str | None, target_language: str) -> str | None:
        entries = self._load()
        if not entries:
            return None
        key = normalize(text)
        if not key:
            return None
        target = target_language.lower()
        for scope in ((source_language or "").lower(), "*"):
            if not scope:
                continue
            hit = entries.get((scope, target), {}).get(key)
            if hit is not None:
                return match_surface(text, hit)
        return None

    def _load(self) -> dict[tuple[str, str], dict[str, str]]:
        if self.path is None:
            return {}
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except FileNotFoundError:
                self._entries, self._mtime = {}, None
                return {}
            except OSError as exc:
                logger.warning("Glossary %s unreadable: %s", self.path, exc)
                return self._entries
            if mtime in (self._mtime, self._bad_mtime):
                return self._entries
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
                self._entries = self._parse(raw)
                self._mtime, self._bad_mtime = mtime, None
                logger.info(
                    "Loaded glossary %s (%d entries)",
                    self.path,
                    sum(len(v) for v in self._entries.values()),
                )
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                # Keep serving the previous good version while the file is being edited,
                # and do not re-parse the same broken save on every lookup.
                self._bad_mtime = mtime
                logger.warning("Glossary %s invalid, keeping previous version: %s", self.path, exc)
            return self._entries

    @staticmethod
    def _parse(raw: object) -> dict[tuple[str, str], dict[str, str]]:
        if not isinstance(raw, dict):
            raise TypeError("glossary root must be a JSON object")
        parsed: dict[tuple[str, str], dict[str, str]] = {}
        for scope, terms in raw.items():
            if str(scope).startswith("_"):
                continue  # "_comment" etc.
            if not isinstance(terms, dict):
                raise TypeError(f"glossary section {scope!r} must be an object")
            if "->" in scope:
                source, target = (part.strip().lower() for part in scope.split("->", 1))
            else:
                source, target = "*", scope.strip().lower()
            if not target:
                raise ValueError(f"glossary section {scope!r} has no target language")
            bucket = parsed.setdefault((source or "*", target), {})
            for term, translation in terms.items():
                key = normalize(str(term))
                if key and isinstance(translation, str) and translation.strip():
                    bucket[key] = translation.strip()
        return parsed
