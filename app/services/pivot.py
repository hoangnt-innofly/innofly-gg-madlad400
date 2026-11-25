"""When to translate through English (source -> en -> target).

Pure python, no model needed.

MADLAD-400 3B is strong into and out of English but weak between two other languages.
Live ja->vi probes, sentence by sentence: the direct output invented content ("大学生に
なると、自分で授業を選び…" -> "Sau khi tốt nghiệp, sinh viên có thể tự lựa chọn ngành học")
and replaced the subject of "私は…と思います" with "các trường đại học", while ja->en was
exact and en->vi of that English kept the meaning and the subject ("Tôi nghĩ…"). Over 12
live pairs (ja/zh/ko/de/ru/th/fr->vi, vi->ja/zh/ko, ja<->zh) direct made 40 serious errors,
the pivot 9; no pair was better direct, so the rule has no exceptions and no setting.
"""

from __future__ import annotations

from app.services.segmenter import target_base

PIVOT_LANGUAGE = "en"


def choose_pivot(
    source_language: str | None,
    target_language: str,
    *,
    requested: bool | None = None,
) -> str | None:
    """``PIVOT_LANGUAGE``, or None for direct translation.

    * ``requested`` False (form field ``pivot=false``): direct;
    * the target is English: direct (nothing to pivot through);
    * ``requested`` True: pivot whatever the source (also an unknown one);
    * otherwise pivot only when the source is known and is neither English nor the
      target language.
    """
    if requested is False:
        return None
    tgt = target_base(target_language)
    if not tgt or tgt == PIVOT_LANGUAGE:
        return None
    if requested:
        return PIVOT_LANGUAGE
    src = target_base(source_language or "")
    if not src or src in ("auto", PIVOT_LANGUAGE, tgt):
        return None
    return PIVOT_LANGUAGE
