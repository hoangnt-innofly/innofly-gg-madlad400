"""Regression probes against a running API: reported bugs + pivot / routing checks.

Usage (API running):
    python scripts/probe_translations.py --url http://127.0.0.1:8002 --key <SECRET_API_KEY>
    python scripts/probe_translations.py --url https://... --key ... --compare   # also pivot=false, side by side
    python scripts/probe_translations.py --key ... --only tanaka nin           # probes whose name contains these

Each probe has words the translation must contain / must not contain (case-insensitive).
KNOWN marks a model limit that is not fixed yet: printed, never counted as a failure.
Exit code 1 when a probe fails. Every request is one job on the shared GPU: the full set
is ~16 requests (~32 with --compare).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROBES: list[dict] = [
    {
        "name": "reported-paragraph",
        "source": "auto",
        "target": "vi",
        "text": (
            "テーマ：日本の大学生活  日本の大学生活は、高校までの生活とは大きく違います。"
            "大学生になると、自分で授業を選び、時間割を作ることができます。"
            "また、勉強だけでなく、アルバイトやサークル活動を通して、さまざまな経験をすることができます。"
            "大学生活は自由な時間が多い一方で、自分の行動に責任を持つことも大切です。"
            "私は、大学生活を充実させるためには、勉強と遊びのバランスを取りながら、"
            "新しいことに積極的に挑戦することが必要だと思います。"
        ),
        "must": ["Chủ đề:", "Tôi"],
        "must_not": ["các trường đại học cần", "Sau khi tốt nghiệp", "quy mô lớn nhất"],
    },
    {"name": "nin-hiragana", "source": "auto", "target": "vi", "text": "にんべん", "must": ["bộ Nhân đứng"]},
    {"name": "nin-brackets", "source": "auto", "target": "vi", "text": "「にんべん」", "must": ["bộ Nhân đứng"]},
    {"name": "nin-kanji", "source": "auto", "target": "vi", "text": "人偏", "must": ["bộ Nhân đứng"]},
    {
        "name": "vi-with-kanji-to-ja",
        "source": "auto",
        "target": "ja",
        "text": "Từ 休み trong tiếng Nhật nghĩa là ngày nghỉ, hãy dịch câu này giúp tôi.",
        "must_change": True,
        "must_not": ["nghĩa là"],
    },
    {
        "name": "mixed-study-sheet",
        "source": "ja",
        "target": "vi",
        "text": "休みの日に何をしますか。\nGhi chú: đây là câu hỏi thường gặp trong bài học.",
        "must": ["Ghi chú: đây là câu hỏi thường gặp trong bài học."],
    },
    {
        "name": "english-line-in-ja",
        "source": "ja",
        "target": "vi",
        "text": "毎日日本語を勉強しています。\nPlease read page 12 before class.",
        "must": ["12"],
    },
    {
        "name": "tanaka",
        "source": "ja",
        "target": "vi",
        "text": "田中さんは来週の月曜日に大阪へ出張する予定です。",
        "must": ["Tanaka", "Osaka"],
        "must_not": ["2011"],
    },
    {
        "name": "zh-first-person",
        "source": "zh",
        "target": "vi",
        "text": "我认为大学生应该在学习和娱乐之间找到平衡。",
        "must": ["Tôi"],
    },
    {
        "name": "zh-numbers",
        "source": "zh",
        "target": "vi",
        "text": "据新华社报道，2024年3月15日，上海的气温达到了28摄氏度。",
        "must": ["2024", "28", "15"],
    },
    {
        "name": "ko-to-vi",
        "source": "ko",
        "target": "vi",
        "text": "저는 내일 서울에서 친구를 만날 거예요.",
        "must": ["Seoul"],
    },
    {
        "name": "vi-to-ja-numbers",
        "source": "vi",
        "target": "ja",
        "text": "Chúng tôi sẽ gửi báo giá cho quý công ty trước ngày 15 tháng 12 năm 2024.",
        "must": ["15", "2024"],
    },
    {"name": "fr-to-vi", "source": "fr", "target": "vi", "text": "Je pense que c'est une très bonne idée.", "must": ["Tôi"]},
    {"name": "en-short-word", "source": "en", "target": "vi", "text": "kite", "must": ["diều"]},
    {
        "name": "title-alone",
        "source": "ja",
        "target": "vi",
        "text": "日本の大学生活",
        "must": ["cuộc sống"],
        "known": "short titles lose the head noun ('Đại học ở Nhật Bản')",
    },
]


def call(url: str, key: str, probe: dict, pivot: bool | None) -> tuple[dict, float]:
    fields = {"text": probe["text"], "source_language": probe["source"], "target_language": probe["target"]}
    if pivot is not None:
        fields["pivot"] = "true" if pivot else "false"
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/translate",
        data=urllib.parse.urlencode(fields).encode(),
        headers={"api-key": key, "Content-Type": "application/x-www-form-urlencoded"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=660) as resp:
        body = json.load(resp)
    return body, time.perf_counter() - started


def check(probe: dict, out: str) -> list[str]:
    low = (out or "").casefold()
    problems = [f"missing {w!r}" for w in probe.get("must", []) if w.casefold() not in low]
    problems += [f"contains {w!r}" for w in probe.get("must_not", []) if w.casefold() in low]
    if probe.get("must_change") and (out or "").strip() == probe["text"].strip():
        problems.append("returned the input untranslated")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("MT_API_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--key", default=os.environ.get("MT_API_KEY") or os.environ.get("SECRET_API_KEY"))
    parser.add_argument("--compare", action="store_true", help="also run every probe with pivot=false")
    parser.add_argument("--only", nargs="*", default=[], help="run probes whose name contains one of these")
    args = parser.parse_args()
    if not args.key:
        parser.error("--key (or MT_API_KEY / SECRET_API_KEY) is required")
    sys.stdout.reconfigure(encoding="utf-8")

    probes = [p for p in PROBES if not args.only or any(o in p["name"] for o in args.only)]
    failed = 0
    for probe in probes:
        runs = [("rule", None)] + ([("direct", False)] if args.compare else [])
        for label, pivot in runs:
            try:
                body, secs = call(args.url, args.key, probe, pivot)
            except urllib.error.HTTPError as exc:
                print(f"ERROR {probe['name']} [{label}]: HTTP {exc.code} {exc.read()[:200]!r}")
                failed += label == "rule"
                continue
            out = body.get("translation") or ""
            problems = check(probe, out)
            if label != "rule":
                status = "info"
            elif not problems:
                status = "PASS"
            elif probe.get("known"):
                status = "KNOWN"
            else:
                status = "FAIL"
                failed += 1
            print(
                f"{status:5} {probe['name']} [{label}] {secs:5.1f}s "
                f"src={body.get('source_language')} pivot={body.get('pivot_language')} chunks={body.get('chunks')}"
            )
            print(f"      {out}")
            if problems:
                print(f"      -> {'; '.join(problems)}" + (f" (known: {probe['known']})" if probe.get("known") else ""))
    print(f"\n{len(probes) - failed}/{len(probes)} probes passed" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
