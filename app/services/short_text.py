"""Helpers for translating isolated words / very short phrases with MADLAD-400.

MADLAD is a sentence-level model. On a lone word ("kite", "kit", "cape") greedy
decoding often degenerates ("đi đi"), picks an unrelated common word ("đi"), or
copies the source untouched. The engine therefore generates several candidates
for short input and reranks them; this module holds the model-free parts
(candidate cleanup, penalties, segmentation) so they can be tested without a GPU.
"""

from __future__ import annotations

import re
import unicodedata

_EDGE_PUNCT = " \t\r\n\"'`“”‘’«»()[]{}"
_TRAILING_PUNCT = ".,;:!?…。！？、，；："
_WORD = re.compile(r"\w+", re.UNICODE)
# "Hi. Go" (Latin needs a space after the stop, so "U.S." / "3.5" stay short) or "好。走" (CJK).
_SENTENCE_END_INSIDE = re.compile(r"[.!?…]\s+\S|[。！？]\s*\S")
# Scripts written without spaces between words: str.split() cannot count their words.
_NO_SPACE_SCRIPT = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿"  # kana, CJK
    r"฀-๿຀-໿ក-៿က-႟]"  # Thai, Lao, Khmer, Myanmar
)
_CJK_LIKE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힣]")
# Max characters for a "lone word" in a no-space script (猫, 風箏, ว่าว).
_NO_SPACE_MAX_CHARS = 4
# Plain dictionary word(s): letters only, optionally joined by space / hyphen / apostrophe.
_PLAIN_WORDS = re.compile(r"[^\W\d_]+(?:[ '\-’][^\W\d_]+)*")
_TO_FULLWIDTH = {".": "。", "!": "！", "?": "？", ",": "，", ";": "；", ":": "："}
_TO_ASCII = {v: k for k, v in _TO_FULLWIDTH.items()} | {"、": ","}


def normalize(text: str) -> str:
    """Casefolded, accent-preserving key used for copy detection and glossary lookups."""
    cleaned = unicodedata.normalize("NFC", text or "").strip(_EDGE_PUNCT)
    cleaned = cleaned.rstrip(_TRAILING_PUNCT).strip(_EDGE_PUNCT)
    return re.sub(r"\s+", " ", cleaned).casefold()


def is_short_text(text: str, max_words: int, max_chars: int = 48) -> bool:
    """A lone word or tiny phrase without sentence structure."""
    stripped = (text or "").strip()
    if not stripped or max_words <= 0 or len(stripped) > max_chars or "\n" in stripped:
        return False
    core = stripped.rstrip(_TRAILING_PUNCT)
    if _SENTENCE_END_INSIDE.search(core):
        return False
    if _NO_SPACE_SCRIPT.search(core):
        # 我今天去商店买了很多东西 is one "word" to str.split(); only accept a real lone word.
        return len(core.replace(" ", "")) <= _NO_SPACE_MAX_CHARS
    words = core.split()
    return 0 < len(words) <= max_words


def is_plain_word(text: str) -> bool:
    """Letters-only word or phrase (no digits, URLs, e-mails, paths, identifiers)."""
    return _PLAIN_WORDS.fullmatch(normalize(text)) is not None


def input_variants(text: str) -> list[str]:
    """Deterministic rephrasings of the source that steer beam search to different modes."""
    stripped = text.strip()
    variants = [stripped]
    first = stripped[:1]
    if first.isalpha() and first.lower() != first.upper():
        toggled = (first.lower() if first.isupper() else first.upper()) + stripped[1:]
        variants.append(toggled)
    if stripped and stripped[-1] not in _TRAILING_PUNCT:
        variants.append(stripped + ("。" if _CJK_LIKE.search(stripped[-1]) else "."))
    seen: set[str] = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


_ENGLISH_ARTICLE = re.compile(r"(?:the|a|an|to)\s", re.IGNORECASE)


def english_context(text: str) -> str | None:
    """'cap' -> 'the cap': pins the source language to English.

    MADLAD has no source-language tag, so a bare word that is also a word elsewhere is
    often read in the other language (Catalan "cap" = "none" -> "không có", French
    "pain" = bread, German "gift" = poison). An English article removes that ambiguity
    and adds candidates from the English reading. Lower-case plain words only.
    """
    core = normalize(text)
    if not core or not is_plain_word(core) or core != text.strip().rstrip(_TRAILING_PUNCT).strip():
        return None
    if _ENGLISH_ARTICLE.match(core):
        return None
    return f"the {core}"


def match_surface(source: str, candidate: str) -> str:
    """Make a candidate's trailing punctuation / leading case follow the source.

    Punctuation is converted to the candidate's script ("你好！" -> "Xin chào!", not "Xin chào！").
    """
    cand = unicodedata.normalize("NFC", candidate or "").strip()
    src = (source or "").strip()
    if not cand:
        return cand
    cand_core = cand.rstrip(_TRAILING_PUNCT).rstrip()
    if not cand_core:
        return cand
    src_end = src[-1:] if src and src[-1] in _TRAILING_PUNCT else ""
    if src_end:
        if _CJK_LIKE.search(cand_core[-1]):
            src_end = _TO_FULLWIDTH.get(src_end, src_end)
        else:
            src_end = _TO_ASCII.get(src_end, src_end)
    cand = cand_core + src_end
    # Only ever upper-case: lower-casing would break German nouns, proper names, etc.
    if src[:1].isupper() and cand[:1].isalpha() and not cand[:1].isupper():
        cand = cand[:1].upper() + cand[1:]
    return cand


def clean_candidates(source: str, raw: list[str]) -> list[str]:
    """Strip, harmonise surface form, and de-duplicate while keeping beam order."""
    out: list[str] = []
    seen: set[str] = set()
    for cand in raw:
        surfaced = match_surface(source, cand)
        key = normalize(surfaced)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(surfaced)
    return out


def _tokens(text: str) -> list[str]:
    """Whitespace tokens without edge punctuation; 'anak-anak' / 'kupu-kupu' stay one token."""
    return [t for t in (w.strip(_EDGE_PUNCT + _TRAILING_PUNCT).casefold() for w in text.split()) if t]


def repetition_level(text: str, siblings: set[str] | frozenset[str] = frozenset()) -> int:
    """How degenerate a candidate's repetition looks.

    2 = clearly degenerate: a token 3+ times in a row, or a multi-token phrase repeated
        back-to-back ("con mèo con mèo").
    1 = suspicious: exactly "X X" while plain "X" is also a candidate ("đi đi" next to "đi").
    0 = none, or legitimate reduplication ("từ từ" with no "từ" candidate, "anak-anak").
    """
    toks = _tokens(text)
    for i in range(len(toks) - 2):
        if toks[i] == toks[i + 1] == toks[i + 2]:
            return 2
    for size in range(2, len(toks) // 2 + 1):
        for i in range(len(toks) - 2 * size + 1):
            if toks[i : i + size] == toks[i + size : i + 2 * size]:
                return 2
    if len(toks) == 2 and toks[0] == toks[1] and toks[0] in siblings:
        return 1
    return 0


def contains_source(source: str, candidate: str) -> bool:
    """Candidate copies the source words verbatim (whole or inside a longer phrase)."""
    src = _WORD.findall(normalize(source))
    cand = _WORD.findall(normalize(candidate))
    if not src or len(src) > len(cand):
        return False
    return any(cand[i : i + len(src)] == src for i in range(len(cand) - len(src) + 1))


def candidate_penalty(
    source: str,
    candidate: str,
    *,
    siblings: set[str] | frozenset[str] = frozenset(),
    translating_across_languages: bool,
    copy_penalty: float,
    repeat_penalty: float,
    backward_scored: bool = True,
) -> float:
    """Heuristic penalty (in avg-log-prob units) subtracted from the model scores.

    ``siblings`` are the normalized keys of all candidates (used to spot "đi đi" next to "đi").
    ``copy_penalty`` applies to exact copies only; the caller decides whether the source is a
    word that ought to be translated at all. ``backward_scored`` says whether a
    back-translation score is part of the ranking.
    """
    penalty = 0.0
    if repeat_penalty and repetition_level(source) == 0:
        level = repetition_level(candidate, siblings)
        if level == 2 or (level == 1 and not backward_scored):
            penalty += repeat_penalty
        elif level == 1:
            # Could be real reduplication; the back-translation score does most of the work.
            penalty += repeat_penalty / 4
    if copy_penalty and translating_across_languages:
        if normalize(candidate) == normalize(source):
            penalty += copy_penalty
        elif contains_source(source, candidate):
            # Half-translated ("áo cape"). Smaller penalty: loanwords ("bánh pizza") exist.
            penalty += copy_penalty / 2
    return penalty


def is_word_list(paragraph: str, max_words: int) -> bool:
    """Several lines that are each a short term (e.g. a vocabulary list, one word per line)."""
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    return len(lines) > 1 and all(is_short_text(line, max_words) for line in lines)
