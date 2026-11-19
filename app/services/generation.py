"""Decoding policy for MADLAD-400: output budgets, batch planning and output checks.

Pure python (no torch) so it can be tested without the model. The engine uses it as:

    budgets  = [output_budget(n, cap) for n in source_lengths]
    batches  = plan_batches(source_lengths, budgets, batch_size=4, num_beams=1, max_batch_tokens=4096)
    finished, n_out = row_stats(generated_row_ids, eos_id, pad_id)
    issues   = assess(source, GenResult(...), target_language)
    kind     = retry_kind(issues, result, cap)        # "loop" | "longer" | "beam" | None
    why      = pack_incomplete(len(parts), count_sentences(out), result, issues)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.services.segmenter import NO_SPACE_TARGETS, target_base

# Source tokens that are not content: the <2xx> tag and </s>.
_TAG_AND_EOS = 2
# Target/source token ratio per sentence goes up to ~2.4 (en->th); 2c+32 left at least 31
# tokens of headroom on 292 measured sentence pairs and on whole paragraphs.
_MIN_BUDGET = 64


def content_tokens(src_len: int) -> int:
    return max(1, src_len - _TAG_AND_EOS)


def output_budget(src_len: int, cap: int) -> int:
    """max_new_tokens for one input of ``src_len`` tokens (tag and EOS included)."""
    c = content_tokens(src_len)
    return max(1, min(cap, max(_MIN_BUDGET, 2 * c + 32)))


def retry_budget(src_len: int, cap: int) -> int:
    """Bigger budget for a row that stopped without EOS."""
    c = content_tokens(src_len)
    return max(1, min(cap, max(output_budget(src_len, cap), 4 * c + 64)))


def plan_batches(
    lengths: list[int],
    budgets: list[int],
    *,
    batch_size: int,
    num_beams: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    """Group row indices into generate() calls.

    Rows are sorted longest first, so a batch pads little and the largest KV cache is
    allocated by the first call (an OOM shows up at once, not after minutes of work).
    A batch holds at most ``batch_size`` rows and at most ``max_batch_tokens`` cached
    token slots, counted as rows * beams * (longest source + largest budget). A single
    row is always allowed, whatever its size (that is what the engine did before).
    """
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    beams = max(1, num_beams)
    size = max(1, batch_size)
    batches: list[list[int]] = []
    current: list[int] = []
    longest = widest = 0
    for i in order:
        new_longest = max(longest, lengths[i])
        new_widest = max(widest, budgets[i])
        cost = (len(current) + 1) * beams * (new_longest + new_widest)
        if current and (len(current) >= size or cost > max_batch_tokens):
            batches.append(current)
            current = []
            new_longest, new_widest = lengths[i], budgets[i]
        current.append(i)
        longest, widest = new_longest, new_widest
    if current:
        batches.append(current)
    return batches


def row_stats(ids: list[int], eos_id: int, pad_id: int) -> tuple[bool, int]:
    """(finished, generated tokens before EOS) for one generate() output row.

    The row starts with the decoder start token; finished rows are right-padded after
    EOS. A row without EOS ran into max_new_tokens: its text is cut off.
    """
    body = list(ids[1:])
    if eos_id in body:
        return True, body.index(eos_id)
    n = len(body)
    while n and body[n - 1] == pad_id:
        n -= 1
    return False, n


@dataclass(frozen=True)
class GenResult:
    text: str
    finished: bool  # EOS produced
    out_tokens: int  # generated tokens before EOS
    src_len: int  # source tokens incl. tag and EOS
    limit: int  # max_new_tokens the row was generated with


# ------------------------------------------------------------------ repetition loops

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_MAX_PERIOD = 120


@lru_cache(maxsize=None)
def _loop_pattern(period: int, copies: int) -> re.Pattern[str]:
    return re.compile(r"(.{%d})\1{%d,}" % (period, copies - 1), re.DOTALL)


def _min_copies(period: int) -> int:
    # "近近近近近近近近" / "rất rất rất ..." need many copies; a phrase or a whole sentence
    # repeated three times back to back is already a decoding loop.
    return 3 if period >= 4 else 8


def find_loop(text: str) -> tuple[int, int, int] | None:
    """Longest back-to-back repetition containing a letter: (start, period, copies)."""
    best: tuple[int, int, int] | None = None
    best_span = 0
    n = len(text)
    for period in range(1, min(_MAX_PERIOD, n // 3) + 1):
        need = _min_copies(period)
        if period * need > n:
            continue
        for m in _loop_pattern(period, need).finditer(text):
            if not _LETTER.search(m.group(1)):
                continue  # "1000000", "......", "-----"
            span = len(m.group(0))
            if span > best_span:
                best, best_span = (m.start(), period, span // period), span
    return best


def max_repeats(text: str) -> int:
    """Most back-to-back copies of any span containing a letter (1 = no repetition).

    No minimum copy count: "頑張れ頑張れ頑張れ頑張れ" is 4 copies of a 3-character span
    even though ``find_loop`` needs 8 copies for such a short period.
    """
    best = 1
    for period in range(1, min(_MAX_PERIOD, len(text) // 2) + 1):
        if period * (best + 1) > len(text):
            break  # cannot beat ``best`` with a longer period
        for m in _loop_pattern(period, 2).finditer(text):
            if _LETTER.search(m.group(1)):
                best = max(best, len(m.group(0)) // period)
    return best


def is_loop(output: str, source: str) -> bool:
    """Output repeats itself back to back more often than the source does.

    The copy threshold of ``find_loop`` depends on the period, so a faithful translation
    of a repeated short word repeats a longer span ("頑張れ" x4 -> "Cố lên, cố lên, cố lên,
    cố lên!"): it is not a loop while it has no more copies than the source.
    """
    loop = find_loop(output)
    return loop is not None and loop[2] > max_repeats(source)


_STOPS = frozenset(".!?…。！？।۔؟։።။។")


def trim_loop(text: str) -> str:
    """Keep one copy of the repeated span (used only on cut-off rows).

    The text after the first copy repeats it, so the cut may slide up to one period
    further to land after a sentence stop, else before a space, instead of mid-word.
    """
    loop = find_loop(text)
    if loop is None:
        return text
    start, period, _ = loop
    end = start + period
    last = min(len(text), end + period - 1)
    cut = next((k for k in range(end, last + 1) if text[k - 1] in _STOPS), None)
    if cut is None:
        cut = next((k for k in range(end, last + 1) if k == len(text) or text[k].isspace()), end)
    return text[:cut].rstrip(" ,、，")


# ------------------------------------------------------------------ other checks

_KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_HAN_TARGETS = NO_SPACE_TARGETS  # ja, zh, yue, ...


def wrong_script(output: str, target_language: str) -> bool:
    """Output in a script the target does not use (e.g. the model wrote new Japanese
    text for a ja->vi request instead of translating)."""
    base = target_base(target_language)
    latin_variant = "_latn" in (target_language or "").lower()
    if base == "ja" and not latin_variant:
        return False
    if len(_KANA.findall(output)) >= 2:
        return True
    if base in _HAN_TARGETS and not latin_variant:
        return False
    letters = len(_LETTER.findall(output))
    return letters > 0 and len(_HAN.findall(output)) / letters > 0.3


def too_short(result: GenResult) -> bool:
    """Probably dropped content: the lowest per-sentence ratio measured is ~0.5 (ja->en)."""
    c = content_tokens(result.src_len)
    return c >= 12 and result.out_tokens < 0.3 * c


def assess(source: str, result: GenResult, target_language: str) -> frozenset[str]:
    issues: set[str] = set()
    if not result.finished:
        issues.add("truncated")
    if is_loop(result.text, source):
        issues.add("loop")
    if result.text and wrong_script(result.text, target_language):
        issues.add("script")
    if too_short(result):
        issues.add("short")
    return frozenset(issues)


# ------------------------------------------------------------------ packed units

# A pack (several spaced-script sentences of one line in one input) is re-translated
# sentence by sentence only on real evidence that a sentence was lost. Its output's
# sentence count alone is weak evidence: the target may join two sentences with a comma
# or semicolon, or contain an abbreviation the splitter does not know, and throwing the
# complete pack translation away loses the context it was packed for.
#
# PACK_MIN_RATIO: output/source content tokens below which a pack lost content whatever
#   its sentence count; 0.5 is the lowest per-sentence ratio measured (ja->en, and CJK
#   sources are never packed), so a complete spaced-script pack stays above it.
# PACK_ONE_SHORT_RATIO: one sentence of a 2-4 sentence pack is 25-50% of its tokens, so
#   an output exactly one sentence short that is also under 3/4 of the source length is
#   treated as a dropped sentence, not as two sentences merged in the target.
PACK_MIN_RATIO = 0.5
PACK_ONE_SHORT_RATIO = 0.75


def pack_incomplete(
    parts: int, out_sentences: int, result: GenResult, issues: frozenset[str]
) -> str | None:
    """Why a pack of ``parts`` sentences must be re-translated one by one, or None.

    "issues" (loop, wrong script, cut off, too short), "sentences" (2+ sentences
    missing), "length" (under PACK_MIN_RATIO of the source), "dropped" (one sentence
    missing and under PACK_ONE_SHORT_RATIO).
    """
    if issues:
        return "issues"
    if out_sentences < parts - 1:
        return "sentences"
    ratio = result.out_tokens / content_tokens(result.src_len)
    if ratio < PACK_MIN_RATIO:
        return "length"
    if out_sentences < parts and ratio < PACK_ONE_SHORT_RATIO:
        return "dropped"
    return None


_SEVERITY = {"loop": 3, "script": 3, "truncated": 2, "short": 2}


def severity(issues: frozenset[str]) -> int:
    return sum(_SEVERITY.get(i, 1) for i in issues)


def retry_kind(issues: frozenset[str], result: GenResult, cap: int) -> str | None:
    """Which retry fixes the most serious issue.

    loop   -> beam search with no_repeat_ngram_size=4 (never used globally: sizes 2-3
              corrupt correct Korean / Hindi, 4 is safe on single sentences);
    longer -> same decoding with a bigger budget (only if the cap leaves room);
    beam   -> beam search for wrong-script or suspiciously short output.
    """
    if "loop" in issues:
        return "loop"
    if "truncated" in issues and retry_budget(result.src_len, cap) > result.limit:
        return "longer"
    if issues & {"script", "short"}:
        return "beam"
    return None
