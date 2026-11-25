"""Sentence segmentation and re-joining for MADLAD-400 (sentence-level MT).

Pure python, no model needed.

    units = split_text(text, token_len=engine_token_len, max_tokens=480)
    outs  = [translate(u.text) if u.translate else u.text for u in units]
    text  = join_units(units, outs, target_language)

MADLAD-400 is trained on sentence pairs. Fed a title line plus several sentences as one
input it drops sentences (or, with a "\\n" inside, writes new text in the source
language), so every line is its own unit and CJK-like text is cut at every sentence stop
even without a following space ("。次の文"). Sentences of spaced scripts (en, vi, ...) on
the same line may be re-packed up to ``pack_tokens`` because MADLAD keeps them all and
uses the context; the engine re-translates a pack sentence by sentence when its output
looks incomplete (``Unit.parts``).

Principle: a wrong split cuts a sentence in two (the model then mistranslates both
halves); a missed split only merges two sentences (the pre-fix behaviour). So every
ambiguous case resolves to "do not split".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable

# --------------------------------------------------------------------------- tables

# Ends a sentence even with no space after it (CJK, Indic, Ethiopic, Burmese, Khmer...).
STRONG_STOPS = frozenset(
    "。｡！？‼⁇⁈⁉"  # CJK full/half width, doubled marks
    "।॥"  # Devanagari / Bengali / Gurmukhi danda, double danda
    "۔؟"  # Urdu full stop, Arabic question mark
    "։"  # Armenian full stop
    "።፧፨"  # Ethiopic full stop, question mark, paragraph separator
    "။"  # Myanmar section mark (sentence end)
    "។៕"  # Khmer khan, bariyoosan
    "༎"  # Tibetan nyis shad
    "᠃᠉"  # Mongolian full stop, Manchu full stop
    "܀"  # Syriac end of paragraph
    "꓿"  # Lisu
    "᙮"  # Canadian syllabics full stop
    "．"  # fullwidth full stop (not between digits: "３．５")
)
# Ends a sentence only when followed by whitespace / end of line (Latin-like).
WEAK_STOPS = frozenset(".!?…")
# Tibetan shad marks clause *or* sentence end: split only before whitespace.
WEAK_STRONG = frozenset("།")
CLOSERS = frozenset("\"'”’»›)]}）］｝〕〉》」』】〙〛〗〟＂＇")
OPENERS = frozenset("\"'“‘«‹([{（［｛〔〈《「『【〘〚〖〝¿¡")
# Bracket pairs tracked for depth (ASCII " toggles; ' is never tracked: don't, 'n').
PAIRS = {"“": "”", "«": "»", "(": ")", "（": "）", "「": "」", "『": "』", "【": "】", "《": "》", "〈": "〉", "[": "]", "［": "］", "〝": "〟"}
CLOSE_TO_OPEN = {v: k for k, v in PAIRS.items()}
_OPENER_CHARS = "".join(OPENERS)

# Latin abbreviations after which "." never ends a sentence (lower-cased, no trailing ".").
NO_SPLIT_ABBR = frozenset(
    """
    mr mrs ms mx dr prof st sr sra srta dra jr sen gov gen col lt capt sgt rev hon pres
    vs cf approx eds bzw vgl ggf evtl inkl ggü hr fr mme mlle ud uds dott sig avv
    mt ave co corp ltd inc
    tp ts ths gs pgs bs
    гг ул им тт проф др напр
    """.split()
)
# Reference abbreviations: no split only before a number ("No. 5", "Fig. 3", "S. 12"),
# including month names ("Jan. 5", "5 déc. 2024"; "in Jan. The ..." still splits).
NUMERIC_ABBR = frozenset(
    """
    no nos nr nº n° nro núm num vol vols p pp ch chap sec art arts para fig figs eq eqs ref
    tel ca cap pág s bd abs ziff stk rd rs стр рис см т г
    jan feb mar apr jun jul aug sep sept oct nov dec
    janv févr avr juil déc
    """.split()
)
# "24. Dezember", "Das 21. Jahrhundert", "1. Dünya": a bare 1-3 digit number + "." is an
# ordinal in de / cs / tr / da / ... ; 4-digit years ("in 1990. Then") still end sentences.
_ORDINAL = re.compile(r"\d{1,3}")
# English sentence openers that are not nouns / month names in the ordinal languages:
# "I counted to 3. Then we left." ends a sentence after the number.
_NOT_AFTER_ORDINAL = frozenset(
    "the then it its i we he she they this that these those there but and so you my our "
    "his her their after before when if what why how where".split()
)
# "e.g.", "U.S.", "z.B.", "т.е.", "พ.ศ.": letters with inner dots.
DOTTED_ABBR = re.compile(r"(?:[^\W\d_]{1,4}\.)+[^\W\d_]{1,4}", re.UNICODE)
# Line-initial list marker: "1." "12)" "a." "iv." "(3)" "•" "-" "①", markdown "## ".
LIST_MARKER = re.compile(
    r"^(?:#{1,6}|[-*•・●◦▪▫■□◆◇►▶→※]|\(?\d{1,3}[.)．]|\(?[a-z][.)]|\(?[A-Z]\)|\(?[ivx]{1,4}[.)]|\(?[IVX]{1,4}\)|[①-⑳⑴-⒇❶-❿])(?=\s)"
)
# Bullet symbols glued to the text ("・テーマ：…", "■注意", "※例"): copied as the line's
# lead like a list marker. Only non-ASCII symbols: "-5", "*bold*" and "#tag" are text.
GLUED_BULLET = re.compile(r"^[・●○◎◦▪▫■□◆◇►▶▷→⇒※★☆♪♦❖✓✔➤➢•‣]+(?=[^\W_]|[「『（(【〈《“\"])")
# Clause punctuation for splitting an over-long sentence.
CLAUSE_BREAK = re.compile(r"(?<=[、，,；;：:])\s*|(?<=\s[-–—])\s+")

# Targets written without spaces between sentences.
NO_SPACE_TARGETS = frozenset({"ja", "zh", "yue", "wuu", "lzh", "gan", "hak", "cmn"})

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ　-〿！-｠]")
_HAN_KANA = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
# Han / Kana / Hangul: scripts that never write Latin-style abbreviations ("Mr.", "U.S.").
_CJK_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ가-힣]")
_QUOTATIVE = re.compile(r"(?:と|って|っ|など|とか|みたい)")
_HIRAGANA = re.compile(r"[぀-ゟ]")
# Scripts with no sentence punctuation in practice (Thai, Lao): space separates clauses.
_THAI_LAO = re.compile(r"[฀-໿]")
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_ARMENIAN = re.compile(r"[԰-֏]")
_INDIC_DANDA_SCRIPTS = re.compile(r"[ऀ-੿଀-୿]")  # Deva, Beng, Guru, Orya
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class Unit:
    text: str  # what is sent to the model (or passed through)
    sep: str  # source whitespace that followed it ("" at the end, "\n" etc. for layout)
    lead: str = ""  # list marker / bullet kept verbatim in front of the translation
    translate: bool = True  # False: symbols / numbers only, copied as-is
    # The unit is a whole line, or the whole content of a heading line ("kite" of
    # "Word: kite"): eligible for the short-text rerank.
    whole_line: bool = True
    line: int = 0  # index of the source line (for line-level glossary lookups)
    # Packed unit (several spaced-script sentences of one line): the sentences it is
    # made of, so the engine can fall back to translating them one by one.
    parts: tuple["Unit", ...] = ()
    block: int = 0  # index of the blank-line separated block the line belongs to
    # Heading label ("テーマ" of "テーマ：環境問題…"); ``sep`` holds the source colon.
    label: bool = False


# --------------------------------------------------------------------------- helpers


def _is_emoji(ch: str) -> bool:
    return unicodedata.category(ch) == "So" or ch in "‍️"


def _is_lower(ch: str) -> bool:
    return ch.islower() and ch.upper() != ch


def _is_upper(ch: str) -> bool:
    return ch.isupper() and ch.lower() != ch


def _first_letterish(text: str, start: int) -> str:
    """First char after ``start`` skipping openers (¿ ¡ « “ ( 「 ...)."""
    i = start
    while i < len(text) and text[i] in OPENERS:
        i += 1
    return text[i] if i < len(text) else ""


def _word_before(text: str, end: int) -> str:
    """The token ending at ``end`` (exclusive), without leading openers."""
    i = end
    while i > 0 and not text[i - 1].isspace() and text[i - 1] not in OPENERS:
        i -= 1
    return text[i:end]


def _abbr_word(text: str, end: int) -> str:
    """The Latin-like word ending at ``end``: ``_word_before`` cut after its last Han /
    Kana / Hangul character. CJK text has no spaces, so "彼はU.S" must be checked as
    "U.S" and "我好.你好" (two sentences) must not look like a dotted abbreviation."""
    return _CJK_SCRIPT.split(_word_before(text, end))[-1]


def _is_abbreviation(word: str, following: str) -> bool:
    """Is the "." after ``word`` part of it rather than a sentence end before ``following``
    (the text after the stop, from its first non-opener character)?

    Shared by the "stop + space" and the "stop glued to CJK" cases so "Mr. 田中" and
    "Mr.田中" behave alike.
    """
    core = word.rstrip(".")
    if not core:
        return False
    nxt = following[:1]
    low = core.casefold()
    if low in NO_SPLIT_ABBR:
        return True
    if low in NUMERIC_ABBR and nxt[:1].isdigit():
        return True
    if len(core) == 1 and _LETTER.match(core):
        return True  # initials: "J. K. Rowling", "M. Dupont", "د. أحمد", "Plan B. Then" (accepted miss)
    if DOTTED_ABBR.fullmatch(core):
        return True  # "U.S. Army", "e.g. Paris", "z.B. Berlin", "พ.ศ. 2500"
    if _ORDINAL.fullmatch(core) and nxt and _LETTER.match(nxt):
        # "am 24. Dezember", "5. května", "1. Dünya Savaşı"; not "to 3. Then".
        return _next_word(following).casefold() not in _NOT_AFTER_ORDINAL
    return False


def _next_word(text: str) -> str:
    m = re.match(r"[^\W\d_]+", text)
    return m.group() if m else ""


# ------------------------------------------------------------------ sentence split


def _boundaries(line: str, *, track_brackets: bool) -> tuple[list[int], int]:
    """Indices where a new sentence starts (after stops+closers+spaces). Returns (cuts, depth)."""
    cuts: list[int] = []
    n = len(line)
    depth = 0
    ascii_quote_open = False
    greek = bool(_GREEK.search(line))
    armenian = bool(_ARMENIAN.search(line))
    danda_script = bool(_INDIC_DANDA_SCRIPTS.search(line))
    i = 0
    while i < n:
        ch = line[i]
        if track_brackets:
            if ch in PAIRS:
                depth += 1
            elif ch in CLOSE_TO_OPEN and depth > 0:
                depth -= 1
            elif ch == '"':
                if ascii_quote_open:
                    depth -= 1
                else:
                    depth += 1
                ascii_quote_open = not ascii_quote_open

        is_strong = ch in STRONG_STOPS
        is_weak = ch in WEAK_STOPS or ch in WEAK_STRONG
        if not is_strong and not is_weak:
            if greek and ch in ";;":
                is_weak = True
            elif armenian and ch == ":":
                is_weak = True
            elif danda_script and ch == "|":
                is_strong = True
        if not (is_strong or is_weak):
            i += 1
            continue

        # "３．５", "3.5", "1,000.5"
        if ch in ".．" and i > 0 and i + 1 < n and line[i - 1].isdigit() and line[i + 1].isdigit():
            i += 1
            continue

        stop_start = i
        j = i + 1
        while j < n and (line[j] in STRONG_STOPS or line[j] in WEAK_STOPS or line[j] in WEAK_STRONG):
            if line[j] in STRONG_STOPS:
                is_strong = True
            j += 1
        stop_run = line[stop_start:j]
        depth_at_stop = depth
        k = j
        while True:
            # French « Bonjour. » : spaces (incl. NBSP) before a closing guillemet.
            sp = k
            while sp < n and line[sp] in "   " and sp + 1 < n and line[sp + 1] in "»›":
                sp += 1
            if not (sp < n and (line[sp] in CLOSERS or _is_emoji(line[sp]))):
                break
            k = sp
            c = line[k]
            if track_brackets:
                if c in CLOSE_TO_OPEN and depth > 0:
                    depth -= 1
                elif c == '"' and ascii_quote_open:
                    depth -= 1
                    ascii_quote_open = False
            k += 1
        closers = line[j:k]
        m = k
        while m < n and line[m].isspace():
            m += 1
        ws = line[k:m]
        if m >= n:  # end of line: nothing to split off
            i = m
            continue
        nxt = _first_letterish(line, m)

        split = _decide(
            line, stop_start, stop_run, closers, ws, nxt, is_strong,
            depth_at_stop=depth_at_stop, depth_after=depth,
            next_is_opener=line[m] in OPENERS,
            following=line[m:].lstrip(_OPENER_CHARS),
        )
        if split:
            cuts.append(m)
        i = max(m, i + 1) if split else k if k > i else i + 1
    return cuts, depth


def _decide(line, at, stop_run, closers, ws, nxt, strong, *, depth_at_stop, depth_after, next_is_opener, following) -> bool:
    if depth_after > 0:
        return False  # still inside a quote / bracket
    if not ws and not next_is_opener and nxt and _LETTER.match(nxt) and any(c in CLOSERS for c in closers):
        # '"Hi!"라고 말했다.', "'좋아.'하고 웃었다.", "“我来了。”他说。": a closing quote glued to
        # the next word is followed by the speech tag of the same sentence (any script;
        # ' is never depth-tracked, so this cannot rely on the bracket depth).
        return False
    if depth_at_stop > 0:
        # 「行こう。」と言った / "Stop!" she said: the quote closes the sentence only if
        # a space, another quote or a capital follows.
        if not ws and not next_is_opener and (not nxt or _HIRAGANA.match(nxt) or _is_lower(nxt)):
            return False  # quotative と / って / など ...
        if nxt and _is_lower(nxt):
            return False
        return True
    if strong:
        if not ws and stop_run[-1] in "！？!?‼⁇⁈⁉" and _QUOTATIVE.match(line[at + len(stop_run) + len(closers):]):
            return False  # なぜ？と思った / 本当！って言った
        if stop_run == "．" and not ws and not (nxt and _HAN_KANA.match(nxt)):
            return False  # "ver．2", "Ｕ．Ｓ．"
        return True
    # Weak (Latin-like) stops.
    if not ws:
        return _glued_weak_split(line, at, stop_run, closers, nxt)
    if nxt and _is_lower(nxt) and not re.match(r"[a-z]+[A-Z]", line[line.index(nxt, at):]):
        return False  # "e.g. apples", "Well... maybe", "Yahoo! is"
    if stop_run[0] in "…" or stop_run.startswith(".."):
        return not nxt.isdigit()
    if stop_run[0] != "." or len(stop_run) > 1:
        return True  # ! ? ?! ;(el) :(hy) |(hi)
    word = _word_before(line, at)
    if not word.rstrip("."):
        return True
    if LIST_MARKER.match(line[: at + 1] + " ") and line[: at + 1].strip() == word + ".":
        return False  # "1. Introduction", "a. first"
    return not _is_abbreviation(_abbr_word(line, at), following)


def _glued_weak_split(line: str, at: int, stop_run: str, closers: str, nxt: str) -> bool:
    """ASCII ".!?" with no space after it: a sentence end only between CJK / Hangul text.

    "我很好.你呢?" splits; "Mr.田中さん", "Dr.スミス", "彼はU.S.出身", "Apple Inc.の製品",
    "Mr.김이" and "Yahoo!ニュース" do not: the character before the stop is Latin, and the
    abbreviation checks of the spaced case run on the Latin word too.
    """
    if stop_run[0] not in ".!?" or "…" in stop_run or stop_run.startswith(".."):
        return False
    rest = line[at + len(stop_run) + len(closers):]
    if _QUOTATIVE.match(rest):
        return False  # "本当!って"
    if stop_run == "." and _is_abbreviation(_abbr_word(line, at), rest.lstrip(_OPENER_CHARS)):
        return False
    if rest[:1] in ("¿", "¡"):
        return True  # "Hola.¿Qué tal?"
    prev = line[at - 1] if at > 0 else ""
    return bool(_CJK_SCRIPT.match(prev)) and bool(nxt) and bool(_CJK_SCRIPT.match(nxt))


# A run of 2+ spaces / tabs / U+3000 between Han / Kana text: Japanese and Chinese put no
# spaces inside a sentence, so such a gap is a line break that was lost when the text was
# pasted ("テーマ：日本の大学生活  日本の大学生活は、…" = a title, then the first sentence).
# Sent as one input, MADLAD merged the title into the sentence and invented content. One
# space is left alone: names and addresses ("山田　太郎", "東京都 港区") use it.
_HAN_KANA_CHARS = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
_GAP = re.compile(
    r"(?<=[%s」』）】〉》〕\]\)])[ \t　]{2,}(?=[%s「『（【〈《〔\[\(])"
    % (_HAN_KANA_CHARS, _HAN_KANA_CHARS)
)


_HAN_KANA_1 = re.compile("[%s]" % _HAN_KANA_CHARS)
_GAP_CLOSERS = "」』）】〉》〕])"
_GAP_OPENERS = "「『（【〈《〔[("
_GAP_SEP = re.compile(r"[ \t　]{2,}")


def _depths(line: str) -> list[int]:
    """Bracket / quote depth before each character (same rules as ``_boundaries``)."""
    out: list[int] = []
    depth = 0
    quote = False
    for ch in line:
        out.append(depth)
        if ch in PAIRS:
            depth += 1
        elif ch in CLOSE_TO_OPEN and depth > 0:
            depth -= 1
        elif ch == '"':
            depth += -1 if quote else 1
            quote = not quote
    return out + [depth]


def _gap_cuts(line: str) -> list[int]:
    """Ends of the ``_GAP`` runs that separate two pieces of Han / Kana text.

    Brackets around the gap are stepped over ("（注）  本文", "「はい」  「いいえ」"), but
    the text next to them must be Han / Kana on both sides: "See figure (a)  (b)" is
    one sentence. A gap inside 「…」 / (…) is not a break ("「東京  大阪」に行った。"),
    unless the line's brackets are unbalanced (then they are ignored, as in
    ``split_sentences``).
    """
    matches = list(_GAP.finditer(line))
    if not matches:
        return []
    depth = _depths(line)
    balanced = depth[-1] == 0
    cuts: list[int] = []
    for m in matches:
        left = line[: m.start()].rstrip(_GAP_CLOSERS)[-1:]
        right = line[m.end() :].lstrip(_GAP_OPENERS)[:1]
        if not (_HAN_KANA_1.match(left) and _HAN_KANA_1.match(right)):
            continue
        if balanced and depth[m.start()] > 0:
            continue
        cuts.append(m.end())
    return cuts


def _split_gaps(line: str) -> list[tuple[str, str]]:
    """Line -> [(segment, gap whitespace after it)] at ``_gap_cuts``."""
    out: list[tuple[str, str]] = []
    start = 0
    for cut in _gap_cuts(line) + [len(line)]:
        piece = line[start:cut]
        body = piece.rstrip(" \t　")
        if body:
            out.append((body, piece[len(body):]))
        start = cut
    return out


def _is_gap(unit: "Unit") -> bool:
    """Did ``unit`` end at a ``_GAP`` (not at a sentence stop followed by spaces)?"""
    return (
        not unit.label
        and bool(_GAP_SEP.fullmatch(unit.sep))
        and bool(_HAN_KANA_1.match(unit.text.rstrip(_GAP_CLOSERS)[-1:]))
    )


def split_sentences(line: str) -> list[tuple[str, str]]:
    """One line (no newline) -> [(sentence, whitespace after it)].

    Besides sentence stops, a 2+ whitespace gap between Han / Kana text ends a piece
    (``_GAP``); its whitespace is the piece's separator, so ``join_units`` can copy it.
    """
    cuts, depth = _boundaries(line, track_brackets=True)
    if depth != 0:  # unbalanced quotes/brackets: ignore them rather than never splitting
        cuts, _ = _boundaries(line, track_brackets=False)
    cuts = sorted(set(cuts) | set(_gap_cuts(line)))
    out: list[tuple[str, str]] = []
    start = 0
    for cut in cuts + [len(line)]:
        piece = line[start:cut]
        body = piece.rstrip()
        if body.strip():
            out.append((body, piece[len(body):]))
        start = cut
    # Long quoted monologue 「…。…。…。」 kept whole by depth tracking: re-split it so
    # MADLAD does not drop sentences inside.
    final: list[tuple[str, str]] = []
    for body, sep in out:
        inner = sum(body.count(c) for c in "。！？.!?")
        if inner >= 3 and len(body) > 120:
            sub, _ = _boundaries(body, track_brackets=False)
            if sub:
                s = 0
                for cut in sub + [len(body)]:
                    p = body[s:cut]
                    b = p.rstrip()
                    if b.strip():
                        final.append((b, p[len(b):] if cut != len(body) else sep))
                    s = cut
                continue
        final.append((body, sep))
    return final


# ------------------------------------------------------------------ lines / layout

_LINE_SPLIT = re.compile(r"([ \t　]*\n\s*)")
# Other line breaks (str.splitlines() ones): MADLAD's tokenizer keeps U+2028 as its own
# piece and VT / FF / U+2029 as byte pieces, so the model would see a break inside the
# input just like "\n". They are all read (and written back) as "\n".
_LINE_BREAKS = str.maketrans({c: "\n" for c in "\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"})


def _ends_sentence(line: str) -> bool:
    core = line.rstrip()
    while core and core[-1] in CLOSERS:
        core = core[:-1]
    return bool(core) and (core[-1] in STRONG_STOPS or core[-1] in WEAK_STOPS or core[-1] in ":：")


WRAP_MIN_WIDTH = 30  # a soft wrap only happens on long lines


def _soft_wrapped(prev: str, nxt: str, width: int, wrap_width: int | None) -> bool:
    """Is the newline between ``prev`` and ``nxt`` a hard wrap inside one sentence?

    Never for short lines (titles, word lists "kite\ncape"), list items or lines that
    end a sentence. Bicameral scripts: the next line starts in lower case. Other
    scripts (CJK...): ``prev`` has the text's fixed wrap width (see ``_wrap_width``).
    """
    if width < WRAP_MIN_WIDTH or len(prev) < 0.6 * width:
        return False
    if _ends_sentence(prev) or LIST_MARKER.match(nxt + " ") or nxt[:1] in OPENERS:
        return False
    first = nxt[:1]
    if _is_lower(first) and not re.match(r"[a-z]+[A-Z]", nxt):
        return True  # "...a sentence that\nwraps here."
    if _is_upper(first) or first.isdigit():
        return False
    return wrap_width is not None and abs(len(prev) - wrap_width) <= 1


def _wrap_width(lines: list[str]) -> int | None:
    """The fixed width a CJK-like text was wrapped at, if it visibly has one.

    A tool that wraps text without spaces cuts every line at the same character count
    (+-1 for line-break rules), mid-sentence. So: the most common length of the lines
    that do not end a sentence, shared by 2+ of them. A title line, or chat messages of
    similar but not equal length, give no such width and are never merged.
    """
    counts: dict[int, int] = {}
    for line in lines:
        if len(line) >= WRAP_MIN_WIDTH and not _ends_sentence(line) and not LIST_MARKER.match(line + " "):
            counts[len(line)] = counts.get(len(line), 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]), default=None)
    return best[0] if best and best[1] >= 2 else None


def _lines(text: str) -> list[tuple[str, str]]:
    """[(line, separator after it)] with soft wraps merged."""
    parts = _LINE_SPLIT.split(text)
    lines = [(parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]
    width = max((len(l) for l, _ in lines), default=0)
    wrap_width = _wrap_width([l for l, _ in lines])
    # A block where no line ends a sentence is a list / poem / lyrics: never merge its lines.
    prose = any(_ends_sentence(l) for l, _ in lines if l)
    merged: list[tuple[str, str]] = []
    prev_raw = ""
    for line, sep in lines:
        if not line:
            if merged:
                merged[-1] = (merged[-1][0], merged[-1][1] + sep)
            continue
        prev_sep = merged[-1][1] if merged else ""
        if merged and prose and prev_sep.count("\n") == 1 and _soft_wrapped(prev_raw, line, width, wrap_width):
            prev = merged[-1][0]
            glue = "" if _CJK.match(prev[-1]) and _CJK.match(line[0]) else " "
            merged[-1] = (prev + glue + line, sep)
        else:
            merged.append((line, sep))
        prev_raw = line
    return merged


# ------------------------------------------------------------------ heading labels

LABEL_MAX_CJK_CHARS = 6
LABEL_MAX_WORDS = 2
_LABEL_PUNCT = frozenset("-'’・&/#_")


def _label_char(ch: str) -> bool:
    return ch.isspace() or ch in _LABEL_PUNCT or unicodedata.category(ch)[0] in "LMN"


def split_heading(line: str) -> tuple[str, str, str] | None:
    """'テーマ：環境問題と私たちの生活' -> ('テーマ', '：', '環境問題と私たちの生活').

    Live: MADLAD reads "テーマ：X" as the wiki namespace "Bản mẫu:X" (Template:X) and
    mistranslates the content with it, while the content alone translates fine. So a
    line made of a short label, a colon and content is translated as two units (the
    label alone also gets the glossary, e.g. ja->vi テーマ -> Chủ đề). Returns
    (label, colon with the whitespace around it, content) or None.

    A label is at most 6 characters of Han / Kana / Hangul text, or at most 2 words
    with 2+ letters otherwise ("Q: ..." stays whole), made of letters, digits, spaces
    and "-'’・&/#_" only (no sentence punctuation), starting with a letter or digit (a
    bullet before it, "・テーマ", is the line's lead). Never a heading: times and ratios
    ("10:30", "1:2", "１０：３０"), an ASCII colon glued to Latin text ("http://",
    "mailto:a", "std::x"), a longer clause before the colon ("The reasons are simple:
    cost."), empty content, or Armenian (":" is its full stop).
    """
    m = re.search(r"[:：]", line)
    if not m or _ARMENIAN.search(line):
        return None
    idx = m.start()
    label = line[:idx].rstrip()
    content = line[idx + 1 :].lstrip()
    if not label or not content:
        return None
    before, after = line[idx - 1], line[idx + 1]
    if before.isdigit() and after.isdigit():
        return None  # 10:30, 1:2
    if not all(_label_char(c) for c in label) or not _LETTER.search(label):
        return None
    if not (_LETTER.match(label[0]) or label[0].isdigit()):
        return None  # "・テーマ" / "#x": a bullet is the line's lead, not part of a label
    if _CJK_SCRIPT.search(label):
        if len(label.replace(" ", "")) > LABEL_MAX_CJK_CHARS:
            return None
    elif len(label.split()) > LABEL_MAX_WORDS or len(_LETTER.findall(label)) < 2:
        return None
    if m.group() == ":" and not after.isspace() and not _CJK_SCRIPT.match(label[-1]):
        return None  # http://, mailto:x, a:b
    return label, line[len(label) : len(line) - len(content)], content


# ------------------------------------------------------------------ long pieces


def _approx_tokens(text: str) -> int:
    return len(text) // 2 + 4


def _pack(pieces: list[str], glue_for: Callable[[str, str], str], fits: Callable[[str], bool]) -> list[str]:
    out: list[str] = []
    cur = ""
    for p in pieces:
        cand = p if not cur else cur + glue_for(cur, p) + p
        if fits(cand):
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def hard_split(text: str, token_len: Callable[[str], int], max_tokens: int) -> list[str]:
    """Over-long sentence: clause punctuation, then whitespace, then characters."""
    fits = lambda s: token_len(s) <= max_tokens
    if fits(text):
        return [text]
    glue = lambda a, b: "" if (_CJK.match(a[-1:] or " ") or not b) else " "
    out: list[str] = []
    clauses = [c for c in CLAUSE_BREAK.split(text) if c]
    for clause in _pack(clauses, glue, fits):
        if fits(clause):
            out.append(clause)
            continue
        words = clause.split()
        if len(words) > 1:
            for w in _pack(words, lambda a, b: " ", fits):
                out.extend([w] if fits(w) else _char_split(w, fits))
        else:
            out.extend(_char_split(clause, fits))
    return out


_JOINERS = "‌‍︎️"


def _graphemes(text: str) -> list[str]:
    """Approximate grapheme clusters: never separate a base from its combining marks
    (Thai / Devanagari / Khmer vowel signs, viramas), variation selectors or ZWJ."""
    out: list[str] = []
    for ch in text:
        if out and (
            unicodedata.category(ch) in ("Mn", "Mc", "Me")
            or ch in _JOINERS
            or out[-1][-1] == "‍"
        ):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _char_split(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Longest fitting runs of graphemes; binary search keeps tokenizer calls at
    O(pieces * log n) for a 20k-character run with no spaces or punctuation."""
    graphemes = _graphemes(text)
    out: list[str] = []
    i = 0
    while i < len(graphemes):
        lo, hi = 1, len(graphemes) - i  # a single grapheme is always taken
        if fits("".join(graphemes[i : i + hi])):
            lo = hi
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if fits("".join(graphemes[i : i + mid])):
                    lo = mid
                else:
                    hi = mid
        out.append("".join(graphemes[i : i + lo]))
        i += lo
    return out


# ------------------------------------------------------------------ public API

# Scripts whose sentences are never packed together: CJK sources drop sentences from
# multi-sentence input even at 2 sentences per unit (live tests), Korean is treated the
# same way to be safe, and Thai/Lao/Khmer/Myanmar/Tibetan lack reliable sentence
# punctuation to verify coverage with.
_NO_PACK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ가-힣ᄀ-ᇿ฀-໿က-႟ក-៿ༀ-࿿]")


def _pack_line(units: list[Unit], tl: Callable[[str], int], pack_tokens: int) -> list[Unit]:
    """Merge consecutive spaced-script sentences of one line up to ``pack_tokens``."""
    out: list[Unit] = []
    group: list[Unit] = []

    def flush() -> None:
        if len(group) == 1:
            out.append(group[0])
        elif group:
            out.append(
                Unit(
                    text=" ".join(u.text for u in group),
                    sep=group[-1].sep,
                    lead=group[0].lead,
                    translate=True,
                    whole_line=False,
                    line=group[0].line,
                    parts=tuple(
                        Unit(text=u.text, sep=u.sep, translate=True, whole_line=False, line=u.line, block=u.block)
                        for u in group
                    ),
                    block=group[0].block,
                )
            )
        group.clear()

    for unit in units:
        if not unit.translate or unit.label or _NO_PACK.search(unit.text):
            flush()
            out.append(unit)
            continue
        # Only sentences the source separated with plain spaces ("" = a hard-split piece).
        if group and group[-1].sep and not group[-1].sep.strip():
            candidate = " ".join(u.text for u in group) + " " + unit.text
            if tl(candidate) <= pack_tokens:
                group.append(unit)
                continue
        flush()
        group.append(unit)
    flush()
    return out


def split_text(
    text: str,
    *,
    token_len: Callable[[str], int] | None = None,
    max_tokens: int = 480,
    clause_tokens: int = 96,
    pack_tokens: int = 0,
) -> list[Unit]:
    """Whole request -> translation units in order, each with the separator after it.

    * every line is its own unit group (titles, list items, dialogue lines), except lines
      that are visibly hard-wrapped inside a sentence;
    * a heading line "LABEL：content" gives a label unit (``Unit.label``) and then the
      content's units (see ``split_heading``);
    * each line is split into sentences (see ``_decide``);
    * a sentence longer than ``max_tokens`` is split at clause punctuation / spaces /
      characters; Thai/Lao lines (no sentence punctuation) longer than ``clause_tokens``
      are split at spaces and re-packed up to ``clause_tokens``;
    * with ``pack_tokens`` > 0, consecutive spaced-script sentences of a line (en, vi,
      de, ru, ar, hi ...) are packed back into one unit of at most ``pack_tokens``
      tokens (``Unit.parts`` keeps the sentences). CJK / Hangul / Thai-like sentences
      are never packed.
    """
    tl = token_len or _approx_tokens
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").translate(_LINE_BREAKS))
    units: list[Unit] = []
    block = 0
    for line_no, (line, line_sep) in enumerate(_lines(text)):
        lead = ""
        m = LIST_MARKER.match(line) or GLUED_BULLET.match(line)
        if m and len(line) > m.end():
            rest = line[m.end():]
            lead = line[: m.end()] + rest[: len(rest) - len(rest.lstrip())]
            line = line[len(lead):]
        line_units: list[Unit] = []
        # A gap between Han / Kana text is a lost line break: each segment gets its own
        # heading check ("目次  第１章：はじめに", "大学生活  テーマ：大学").
        segments = _split_gaps(line) or [(line, "")]
        # (text, separator after it, is a heading label), in order.
        items: list[tuple[str, str, bool]] = []
        heading = False
        for seg_no, (segment, gap) in enumerate(segments):
            seg_heading = split_heading(segment)
            if seg_heading:
                heading = len(segments) == 1
                label, colon, segment = seg_heading
                items.append((label, colon, True))
            seg_sentences = split_sentences(segment)
            if seg_sentences and seg_no < len(segments) - 1:
                seg_sentences[-1] = (seg_sentences[-1][0], gap)
            items.extend((body, sep, False) for body, sep in seg_sentences)
        n_items = len(items)
        for idx, (sent, sep, is_label) in enumerate(items):
            last = idx == n_items - 1
            if is_label:
                line_units.append(
                    Unit(text=sent, sep=sep, lead=lead, whole_line=False, line=line_no, block=block, label=True)
                )
                lead = ""
                continue
            pieces = [sent]
            if _THAI_LAO.search(sent) and not any(c in sent for c in ".!?") and tl(sent) > clause_tokens and " " in sent:
                pieces = _pack(sent.split(" "), lambda a, b: " ", lambda s: tl(s) <= clause_tokens)
            pieces = [p for piece in pieces for p in hard_split(piece, tl, max_tokens)]
            for pidx, piece in enumerate(pieces):
                if pidx < len(pieces) - 1:
                    psep = " " if not _CJK.match(piece[-1]) else ""
                else:
                    psep = line_sep if last else sep
                line_units.append(
                    Unit(
                        text=piece,
                        sep=psep,
                        lead=lead if pidx == 0 else "",
                        translate=bool(_LETTER.search(piece)),
                        whole_line=False,
                        line=line_no,
                        block=block,
                    )
                )
            lead = ""
        if pack_tokens > 0 and len(line_units) > 1:
            line_units = _pack_line(line_units, tl, min(pack_tokens, max_tokens))
        if len(line_units) == 1 or (heading and len(line_units) == 2):
            line_units[-1] = replace(line_units[-1], whole_line=True)
        units.extend(line_units)
        if line_sep.count("\n") >= 2:
            block += 1  # a blank line ends the block
    return units


def line_sources(units: list[Unit]) -> dict[int, str]:
    """Source text of every line (without its list marker), for line-level glossary hits."""
    members: dict[int, list[Unit]] = {}
    for unit in units:
        members.setdefault(unit.line, []).append(unit)
    out: dict[int, str] = {}
    for line_no, line_units in members.items():
        text = ""
        for i, unit in enumerate(line_units):
            text += unit.text
            if i < len(line_units) - 1:
                text += unit.sep
        out[line_no] = text
    return out


def count_sentences(text: str) -> int:
    """Number of sentences ``split_sentences`` finds in (possibly multi-line) text."""
    return sum(
        len(split_sentences(line.strip())) for line in (text or "").splitlines() if line.strip()
    )


def target_base(target_language: str) -> str:
    """'zh_Latn' -> 'zh', 'pt-BR' -> 'pt'."""
    return (target_language or "").split("_")[0].split("-")[0].lower()


def _cjk_edge(ch: str) -> bool:
    return bool(ch) and (bool(_CJK.match(ch)) or ch in CLOSERS or ch in OPENERS)


def join_units(units: list[Unit], outputs: list[str], target_language: str) -> str:
    """Rebuild the text from per-unit outputs.

    Newline separators (blank lines, the next line's indentation such as U+3000) are
    copied from the source. Within a line, pieces are glued with "" when the target is
    written without spaces (ja/zh/yue...) and the left piece ends in a CJK character or
    bracket, otherwise with one space. A heading label is followed by "：" when the
    target is written without spaces and the label's translation has a CJK character
    ("問1：", "ステップ1："), else by ": " (the source colon is not copied: "Chủ đề: ..."
    for テーマ：...).
    The colon is only written when content follows, so a line-level glossary hit put in
    the label's slot stays as it is. Empty outputs leave no double spaces. A 2+ character
    whitespace gap of the source (``_GAP``: a title and a sentence on one line) is copied
    (U+3000 as a plain space unless the target is written without spaces).
    """
    no_space = target_base(target_language) in NO_SPACE_TARGETS
    chunks: list[str] = []
    line: list[str] = []
    after_label = False
    label_cjk = False
    gap = ""
    for i, (unit, out) in enumerate(zip(units, outputs)):
        out = (out or "").strip()
        if unit.label:
            out = out.rstrip(" :：")  # the model may echo the colon
        piece = unit.lead + out if out else unit.lead.rstrip()
        if piece:
            if line:
                cjk = no_space and _cjk_edge(line[-1][-1])
                if after_label:
                    line.append("：" if label_cjk else ": ")
                elif gap:
                    line.append(gap if no_space else gap.replace("　", " "))
                else:
                    line.append("" if cjk else " ")
            line.append(piece)
            after_label = unit.label and bool(out)
            label_cjk = no_space and bool(_CJK.search(out))
        if piece or not line:
            gap = unit.sep if _is_gap(unit) else ""
        last = i == len(units) - 1
        if last or "\n" in unit.sep:
            chunks.append("".join(line))
            line = []
            after_label = False
            if not last:
                chunks.append(unit.sep.lstrip(" \t"))
    return re.sub(r"[ \t]+\n", "\n", "".join(chunks)).strip()
