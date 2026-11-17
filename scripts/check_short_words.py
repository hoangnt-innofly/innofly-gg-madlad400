"""Smoke-test single-word translation against a running API.

Usage (on the server, API already running):
    python scripts/check_short_words.py
    python scripts/check_short_words.py --url http://127.0.0.1:8002 --key change-me --target vi
    python scripts/check_short_words.py kite kit cape        # custom words

Words listed in data/glossary.json bypass the model. To see what the model itself does
for those, start the API with MT_GLOSSARY_PATH= (empty). The server log line
"Short-text rerank ..." shows every candidate with its forward (f) / backward (b) score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

# Short, ambiguous English words that sentence-level MT often gets wrong in isolation.
DEFAULT_WORDS = [
    "kite", "kit", "cape", "bat", "bow", "fan", "pen", "pin", "tin", "can",
    "bank", "match", "mouse", "spring", "seal", "crane", "pitch", "date", "nail", "bark",
    "ring", "tie", "jam", "kid", "pool", "first aid kit", "Google", "NASA",
]


def translate(url: str, key: str, text: str, source: str, target: str) -> dict:
    body = urllib.parse.urlencode(
        {"text": text, "source_language": source, "target_language": target}
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/translate",
        data=body,
        headers={"api-key": key, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("words", nargs="*", help="words to test (default: built-in list)")
    parser.add_argument("--url", default=os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--key", default=os.environ.get("SECRET_API_KEY", "change-me"))
    parser.add_argument("--source", default="auto")
    parser.add_argument("--target", default="vi")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    words = args.words or DEFAULT_WORDS
    width = max(len(w) for w in words)
    for word in words:
        started = time.perf_counter()
        try:
            job = translate(args.url, args.key, word, args.source, args.target)
            out = job.get("translation")
            note = job.get("progress", "")
        except Exception as exc:  # keep going so one failure does not hide the rest
            out, note = None, f"ERROR {exc}"
        took = time.perf_counter() - started
        print(f"{word:<{width}}  ->  {out!s:<30} {took:5.1f}s  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
