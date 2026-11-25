"""Helpers for translating isolated words / very short phrases with MADLAD-400.

MADLAD is a sentence-level model. On a lone word ("kite", "kit", "cape") greedy
decoding often degenerates ("đi đi"), picks an unrelated common word ("đi"), or
copies the source untouched. The engine therefore generates several candidates
for short input and reranks them; this module holds the model-free parts
(candidate cleanup, penalties, segmentation) so they can be tested without a GPU.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger("madlad400-mt-api")

_EDGE_PUNCT = " \t\r\n\"'`“”‘’«»()[]{}　「」『』（）【】〈〉《》［］"
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


def _has_case_seam(text: str) -> bool:
    """A lower-case letter directly followed by an upper-case one ("iPhone", "userName")."""
    return any(a.islower() and b.isupper() for a, b in zip(text, text[1:]))


def is_plain_word(text: str) -> bool:
    """Letters-only word or phrase (no digits, URLs, e-mails, paths, identifiers).

    camelCase names ("iPhone", "YouTube", "PlayStation") are not plain: normalize()
    casefolds, so without the seam check they got the copy handling meant for words that
    ought to be translated, and their correct copies lost to junk ("iPhoneName").
    """
    return not _has_case_seam(text or "") and _PLAIN_WORDS.fullmatch(normalize(text)) is not None


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


# A capitalised ASCII word piece at the very end of a candidate ("Name", "ID").
_GLUED_TAIL = re.compile(r"[A-Z](?:[a-z]+|[A-Z]*)$")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def _cased_run(text: str) -> str:
    """Trailing run of cased letters: 'giao tiếp' -> 'tiếp', '使用Power' -> 'Power', '交流' -> ''."""
    i = len(text)
    while i > 0 and (text[i - 1].islower() or text[i - 1].isupper()):
        i -= 1
    return text[i:]


def _letters_key(text: str) -> str:
    """Casefolded letters/digits without accents or separators: 'Việt Nam' -> 'vietnam'."""
    return "".join(c for c in unicodedata.normalize("NFD", (text or "").casefold()) if c.isalnum())


def strip_glued_tail(
    source: str, candidate: str, siblings: set[str] | frozenset[str] = frozenset()
) -> str:
    """Cut an identifier piece the model glued onto a candidate ("giao tiếpName" -> "giao tiếp").

    On a bare word MADLAD sometimes reads the input as a UI / i18n key and continues the
    last word like "userName": "Name" is one SentencePiece token without a word-start
    marker, so "communication" -> ['▁giao', '▁tiếp', 'Name'] and "iPhone" -> ['▁iPhone',
    'Name']. Length-normalised scores barely punish one extra confident token, and a glued
    copy ("iPhoneName") back-translates as easily as the copy itself, so such junk can win.

    A capitalised ASCII piece at the end, glued to a lower-case or caseless letter, is cut
    when the source does not spell the glued word that way ("JavaScript", "YouTube" for
    "you tube", "userName" stay), more than a one-letter prefix precedes it ("iPhone",
    "eBay", "mRNA", Irish "an tSeirbhís" stay), and
    * what is left is the source ("iPhoneName", "TikTokName"), or
    * what is left is another candidate ("HoaName" next to "Hoa"), unless that is a lone
      ASCII word and the source has no Latin letters ("Git" next to "GitHub" for 깃허브 may
      just be a truncated beam), or
    * the piece is glued to a lower-case word with non-ASCII letters, which is never real
      spelling ("tiếpName", "khoảnData"; "ViệtNam", "ÄrztInnen" start with a capital).
    Stacked pieces ("tiếpUserName") are peeled one by one. Anything else is returned as is.
    """
    text = unicodedata.normalize("NFC", candidate or "").strip()
    core = text.rstrip(_TRAILING_PUNCT).rstrip()
    src_key, src_letters = normalize(source), _letters_key(source)
    latin_source = _LATIN_LETTER.search(source or "") is not None
    head, best = core, None
    while (match := _GLUED_TAIL.search(head)) and match.start() > 0:
        rest = head[: match.start()]
        if not rest[-1].isalpha() or rest[-1].isupper():
            break  # "IJsselmeer", spaced "giao tiếp Name"
        run = _cased_run(rest)
        if len(run) == 1 or _letters_key(run + match.group()) in src_letters:
            break  # "eBay"; the source has it ("JavaScript")
        head = rest
        key = normalize(head)
        sibling = key in siblings and (latin_source or " " in key or not key.isascii())
        if key == src_key or sibling or (run.islower() and not run.isascii()):
            best = head
    return candidate if best is None else best + text[len(core) :]


def clean_candidates(source: str, raw: list[str]) -> list[str]:
    """Strip, harmonise surface form, cut glued junk tails, and de-duplicate in beam order.

    A candidate with a glued tail collapses into its head ("giao tiếpName" -> "giao tiếp"),
    so the junk filter never empties the list.
    """
    surfaced = [match_surface(source, cand) for cand in raw]
    siblings = frozenset(key for key in map(normalize, surfaced) if key)
    out: list[str] = []
    seen: set[str] = set()
    cut: dict[str, str] = {}
    for cand in surfaced:
        stripped = strip_glued_tail(source, cand, siblings)
        if stripped != cand:
            cut.setdefault(cand, stripped)
        key = normalize(stripped)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    if cut:
        logger.info(
            "Short-text %r: cut glued tails %s",
            source,
            ", ".join(f"{junk!r} -> {head!r}" for junk, head in cut.items()),
        )
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
