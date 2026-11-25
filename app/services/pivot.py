"""How each translation unit is routed: copied, translated directly, or through English.

Pure python, no model needed.

MADLAD-400 3B is strong into and out of English but weak between two other languages.
Live ja->vi probes, sentence by sentence: the direct output invented content ("大学生に
なると、自分で授業を選び…" -> "Sau khi tốt nghiệp, sinh viên có thể tự lựa chọn ngành học")
and replaced the subject of "私は…と思います" with "các trường đại học", while ja->en was
exact and en->vi of that English kept the meaning and the subject ("Tôi nghĩ…"). Over 12
live pairs (ja/zh/ko/de/ru/th/fr->vi, vi->ja/zh/ko, ja<->zh) direct made 40 serious errors,
the pivot 9 (fr->vi was a tie: no serious error either way).

The route is decided per unit, from the unit's own script, not once per request: a
ja->vi study sheet has Vietnamese explanation lines ("Ví dụ: …") that must be copied, not
paraphrased through English, and English terms / romaji lines that need no English step.
"""

from __future__ import annotations

import re
from typing import Literal

from app.core.languages import looks_vietnamese, script_language
from app.services.segmenter import target_base

PIVOT_LANGUAGE = "en"

Route = Literal["copy", "direct", "pivot"]

# A unit counts as already written in the target only with evidence no other language
# gives: Vietnamese letters / accented Vietnamese syllables (``looks_vietnamese``; plain
# "can you go" never counts), kana, Hangul. Chinese is never assumed (Japanese kanji
# look the same).
_WRITTEN_IN = {
    "vi": looks_vietnamese,
    "ja": re.compile(r"[぀-ヿ]").search,
    "ko": re.compile(r"[가-힣]").search,
}
# Sources not written in Latin script: a Latin unit inside such a document is English,
# romaji or a code / product name, not the source language.
_NON_LATIN_SOURCES = frozenset(
    "ja zh yue ko th lo km my ar fa ur ps he yi ru uk bg sr mk be kk ky tg mn hi mr ne "
    "bn as ta te kn ml gu pa or si ka hy am ti el bo dz".split()
)


def route_unit(
    text: str,
    *,
    source_language: str | None,
    target_language: str,
    requested: bool | None = None,
) -> Route:
    """Route one unit.

    * ``copy``: the unit is decisively written in the target language (a Vietnamese
      note in a ja->vi document), whatever ``requested`` says;
    * ``direct``: ``requested`` is False, the target is English, or the unit's language
      is unknown, English or the target itself;
    * ``pivot``: ``requested`` is True, or the unit's language (its own script, else the
      request's ``source_language`` for Latin text of a Latin-script source) is neither
      English nor the target.

    ``source_language`` must be a trustworthy request source (explicit or a reliable
    detection), else None.
    """
    tgt = target_base(target_language)
    src = target_base(source_language or "")
    if src == "auto":
        src = ""
    lang, decisive = script_language(text)
    if lang == "vi" and not looks_vietnamese(text):
        lang = None  # é / à / ô of French, Portuguese ...: Latin of unknown language
    src_latin = bool(src) and src not in _NON_LATIN_SOURCES
    tgt_latin = tgt not in _NON_LATIN_SOURCES
    written = _WRITTEN_IN.get(tgt)
    # Never between two Latin-script languages: "Sí", "Não sei", "I live in Hà Nội" are
    # the source language naming / resembling Vietnamese, not Vietnamese notes.
    if (
        decisive
        and lang == tgt
        and src != tgt
        and not (src_latin and tgt_latin)
        and written is not None
        and written(text)
    ):
        return "copy"
    if src_latin and lang in (None, "vi") and src != tgt:
        lang = src  # Latin text of a Latin-script source is in the source language
    if requested is False or not tgt or tgt == PIVOT_LANGUAGE:
        return "direct"
    if requested:
        return "pivot"
    unit_src = lang or ""
    if not unit_src or unit_src in (PIVOT_LANGUAGE, tgt):
        return "direct"
    return "pivot"
