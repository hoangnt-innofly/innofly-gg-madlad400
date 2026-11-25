from __future__ import annotations

import gc
import logging
import time
from typing import Callable

from app.services.generation import (
    GenResult,
    assess,
    output_budget,
    pack_incomplete,
    plan_batches,
    retry_budget,
    retry_kind,
    row_stats,
    severity,
    trim_loop,
)
from app.services.pivot import PIVOT_LANGUAGE, route_unit
from app.services.segmenter import (
    Unit,
    count_sentences,
    join_units,
    line_sources,
    split_text,
    target_base,
)
from app.services.short_text import (
    candidate_penalty,
    clean_candidates,
    contains_source,
    english_context,
    input_variants,
    is_plain_word,
    is_short_text,
    is_word_list,
    normalize,
)

logger = logging.getLogger("madlad400-mt-api")

# Targets written without sentence punctuation in practice: a pack's sentence count
# cannot be checked in the output, so their source sentences are never packed.
_NO_SENTENCE_PUNCT_TARGETS = frozenset({"th", "lo"})
# Beam width of the "loop" / "beam" retries.
_RETRY_BEAMS = 4


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


class MTEngine:
    """Loads MADLAD-400 3B MT once and reuses it for translation jobs."""

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        free_vram: bool = True,
        max_input_tokens: int = 480,
        batch_size: int = 4,
        short_text_rerank: bool = True,
        short_text_max_words: int = 3,
        short_text_candidates: int = 8,
        short_text_backward_weight: float = 1.0,
        short_text_copy_penalty: float = 1.5,
        short_text_repeat_penalty: float = 2.0,
        short_text_max_units: int = 50,
        segment_pack_tokens: int = 256,
        segment_clause_tokens: int = 96,
        batch_max_tokens: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype_name = dtype
        self.free_vram = free_vram
        self.max_input_tokens = max_input_tokens
        self.batch_size = max(1, batch_size)
        self.short_text_rerank = short_text_rerank
        self.short_text_max_words = max(0, short_text_max_words)
        self.short_text_candidates = max(1, min(short_text_candidates, 16))
        self.short_text_backward_weight = max(0.0, short_text_backward_weight)
        self.short_text_copy_penalty = max(0.0, short_text_copy_penalty)
        self.short_text_repeat_penalty = max(0.0, short_text_repeat_penalty)
        # Reranking costs ~3 beam searches + 2 scoring passes per unit; cap it per request so
        # a long vocabulary list does not blow the 600 s request timeout.
        self.short_text_max_units = max(0, short_text_max_units)
        # Spaced-script sentences of one line are packed up to this many tokens (0 = one
        # sentence per model call everywhere). CJK-like sentences are never packed.
        self.segment_pack_tokens = max(0, segment_pack_tokens)
        # Thai/Lao lines without sentence punctuation are cut at spaces above this size.
        self.segment_clause_tokens = max(16, segment_clause_tokens)
        # KV-cache cap per generate() call, in rows * beams * (source + output) tokens.
        # ~256 KiB per slot on MADLAD-3B bf16, so 4096 is ~1 GiB: no more than one
        # 480-token input at 4-5 beams needed before batching existed.
        self.batch_max_tokens = max(1, batch_max_tokens)
        self.model = None
        self.tokenizer = None
        self.sp = None
        self._ready = False
        # Pivot language the last translate() call really used (None: every unit was
        # direct, a glossary hit or a short-text rerank). Jobs run one at a time.
        self.last_pivot: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        import torch
        import transformers
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.free_vram = self.free_vram and torch.cuda.is_available()
        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)
        self._patch_t5_tied_weights()

        load_started = time.perf_counter()
        logger.info(
            "Loading %s (transformers=%s, device=%s, dtype=%s, free_vram=%s)",
            self.model_id,
            transformers.__version__,
            device,
            dtype,
            self.free_vram,
        )
        try:
            # Do not use device_map="auto": T5 tied embeddings stay on the meta
            # device and later .cpu()/.to() raise "Cannot copy out of meta tensor".
            # 3B fp32 also does not fit the shared 12GB card.
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            self.model.to(device)
            self.model.eval()
            probe = self.tokenizer.encode("<2en> ok", add_special_tokens=True)
            probe_tokens = self.tokenizer.convert_ids_to_tokens(probe)
            if "<2en>" not in probe_tokens:
                raise RuntimeError(f"Tokenizer split <2en>: {probe_tokens}")
            self._assert_encoder_embeddings()
            self._ready = True
            logger.info(
                "MADLAD-400 3B MT ready in %.1fs (%s) pad=%s eos=%s start=%s probe=%s",
                time.perf_counter() - load_started,
                self._vram_log(),
                self.model.config.pad_token_id,
                self.model.config.eos_token_id,
                self.model.config.decoder_start_token_id,
                probe_tokens[:4],
            )
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.sp = None
            self._ready = False
            self._free_cuda()
            raise RuntimeError(
                f"Failed to load MADLAD model {self.model_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def translate(
        self,
        *,
        text: str,
        target_language: str,
        source_language: str | None = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        max_new_tokens: int = 512,
        lookup: Callable[[str], str | None] | None = None,
        pivot: bool | None = None,
        source_reliable: bool = True,
    ) -> tuple[str, int]:
        """Translate ``text``; returns (translation, number of segments).

        ``source_language`` is optional (MADLAD only needs the target tag); when known it
        enables back-translation scoring for short input. ``lookup`` is a glossary hook
        tried on every line that was split, then on every unit, before the model.
        Every unit that reaches the model is routed on its own (``pivot.route_unit``):
        copied when it is already written in the target (a Vietnamese note in a ja->vi
        sheet), translated through English (``_translate_pivot``) when neither its
        language nor the target is English, else directly. ``pivot`` is the request
        override (True / False / None = rule); ``source_reliable`` False means
        ``source_language`` is only a guess and is not used for routing. Glossary hits
        and the short-text rerank are not routed (they stay direct).

        The text is cut into lines and sentences (``segmenter.split_text``): MADLAD is a
        sentence-level model and drops sentences, or writes new source-language text,
        when a title line plus several sentences reach it as one input. Units that miss
        the glossary and the short-text path are decoded together in padded batches.

        Short-text rerank (see ``_rerank_lines``): a short whole line (or the whole
        content of a heading line, "kite" of "Word: kite") when its block is one line or
        a word list, and a short heading label ("Note" of "Note: ..."), which reaches
        the model as a lone word. A text is reranked once per request: a speaker name
        repeated on every line of a dialogue ("田中：…") reuses the first result.
        """
        if not self._ready:
            self.load()

        import torch

        started = time.perf_counter()
        units = self._segment(text, target_language)
        if not units:
            return "", 0

        outputs: list[str | None] = [None] * len(units)
        if lookup:
            self._lookup_lines(units, outputs, lookup)
        self.last_pivot = None
        rerank_lines = self._rerank_lines(units) if self.short_text_rerank else frozenset()
        route_source = source_language if source_reliable else None
        routes = {
            i: route_unit(
                unit.text,
                source_language=route_source,
                target_language=target_language,
                requested=pivot,
            )
            for i, unit in enumerate(units)
            if unit.translate
        }
        pending: list[int] = []
        reranked = 0
        copied = 0
        short_memo: dict[str, str] = {}
        try:
            with torch.no_grad():
                for i, unit in enumerate(units):
                    if outputs[i] is not None:
                        continue
                    if not unit.translate:
                        outputs[i] = unit.text  # numbers / symbols only
                        continue
                    # Glossary first: a heading label with an entry (テーマ -> Chủ đề)
                    # never reaches the rerank or the model.
                    hit = lookup(unit.text) if lookup else None
                    if hit is not None:
                        outputs[i] = hit
                    elif routes[i] == "copy":
                        outputs[i] = unit.text  # already in the target language
                        copied += 1
                    elif unit.text in short_memo:
                        outputs[i] = short_memo[unit.text]
                    elif (
                        self.short_text_rerank
                        and (unit.label or (unit.whole_line and unit.line in rerank_lines))
                        and reranked < self.short_text_max_units
                        and is_short_text(unit.text, self.short_text_max_words)
                    ):
                        reranked += 1
                        outputs[i] = short_memo[unit.text] = self._translate_short(
                            unit.text,
                            target_language=target_language,
                            source_language=source_language,
                            max_new_tokens=max_new_tokens,
                        )
                    else:
                        pending.append(i)
                decoding = dict(
                    target_language=target_language,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    max_new_tokens=max_new_tokens,
                )
                direct = [i for i in pending if routes[i] != "pivot"]
                via = [i for i in pending if routes[i] == "pivot"]
                if via:
                    decoded = self._translate_pivot([units[i] for i in via], **decoding)
                    for i, out in zip(via, decoded):
                        outputs[i] = out
                if direct:
                    decoded = self._translate_units([units[i] for i in direct], **decoding)
                    for i, out in zip(direct, decoded):
                        outputs[i] = out
            if self.short_text_rerank and reranked >= self.short_text_max_units > 0:
                logger.info(
                    "Short-text rerank capped at %d units (MT_SHORT_TEXT_MAX_UNITS); "
                    "remaining short lines used plain decoding",
                    self.short_text_max_units,
                )
            logger.info(
                "MT translate -> %s: %d units (%d glossary/rerank/symbols, %d already in "
                "target, %d direct, %d via %s) in %.1fs",
                target_language,
                len(units),
                len(units) - len(pending) - copied,
                copied,
                len(direct),
                len(via),
                PIVOT_LANGUAGE,
                time.perf_counter() - started,
            )
            return join_units(units, [o or "" for o in outputs], target_language), len(units)
        finally:
            # The job worker decides when to unload (only once its queue is empty).
            self._free_cuda()

    def _translate_short(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None,
        max_new_tokens: int,
    ) -> str:
        """Lone word / tiny phrase: generate many candidates, rerank, return the best.

        Greedy decoding on a single word is unreliable ("kite" -> "đi đi", "kit" -> "đi",
        "cape" -> "cape"). Score = forward log P(candidate | source)
        + weight * backward log P(source | candidate) - penalties, all length-normalised.
        The backward term rejects fluent-but-unrelated outputs such as "đi" (which
        translates back to "go", not "kit").
        """
        n = self.short_text_candidates
        device = self._device()
        variants = input_variants(text)
        # Inputs a candidate may be a translation of (forward score = best over these).
        anchors = [text]
        context = english_context(text) if source_language == "en" else None
        if context:
            variants.append(context)
            anchors.append(context)
        raw: list[str] = []
        for variant in variants:
            input_ids = self.tokenizer(
                f"<2{target_language}> {variant}", return_tensors="pt"
            ).input_ids.to(device)
            src_len = int(input_ids.shape[-1])
            outputs = self.model.generate(
                input_ids=input_ids,
                num_beams=n,
                num_return_sequences=n,
                do_sample=False,
                early_stopping=True,
                max_new_tokens=min(max_new_tokens, 4 * src_len + 8),
            )
            raw.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
            del outputs, input_ids

        candidates = clean_candidates(text, raw)
        if not candidates:
            return raw[0].strip() if raw else ""
        if len(candidates) == 1:
            return candidates[0]

        across = bool(source_language) and source_language != target_language
        forward = [float("-inf")] * len(candidates)
        for anchor in anchors:
            scored = self._sequence_logprobs(
                [f"<2{target_language}> {anchor}"] * len(candidates), candidates
            )
            forward = [max(a, b) for a, b in zip(forward, scored)]
        # A plain dictionary word ("cape", "first aid kit") is expected to be translated.
        # camelCase names, URLs, e-mails, numbers and code ("iPhone", "a@b.com", "2024")
        # may legitimately be copied, so they get no copy handling. Capitalised words and
        # acronyms ("Google", "NASA") get no copy penalty, only the neutral backward score.
        plain = is_plain_word(text)
        copy_like = [plain and contains_source(text, c) for c in candidates]
        backward = [0.0] * len(candidates)
        backward_scored = across and self.short_text_backward_weight > 0
        if backward_scored:
            backward = self._sequence_logprobs(
                [f"<2{source_language}> {c}" for c in candidates], [text] * len(candidates)
            )
            # Back-translating a (partial) copy of the source is trivially easy ("cape",
            # "áo cape"), so its backward score says nothing about adequacy. Give such
            # candidates the median backward score of the real translations instead.
            real = sorted(b for b, copy in zip(backward, copy_like) if not copy)
            if real:
                neutral = real[len(real) // 2]
                backward = [neutral if copy else b for b, copy in zip(backward, copy_like)]

        lowercase_word = plain and text.strip() == text.strip().lower()
        copy_penalty = self.short_text_copy_penalty if lowercase_word else 0.0
        siblings = frozenset(normalize(c) for c in candidates)
        scores: list[float] = []
        for cand, fwd, bwd in zip(candidates, forward, backward):
            penalty = candidate_penalty(
                text,
                cand,
                siblings=siblings,
                backward_scored=backward_scored,
                translating_across_languages=across,
                copy_penalty=copy_penalty,
                repeat_penalty=self.short_text_repeat_penalty,
            )
            scores.append(fwd + self.short_text_backward_weight * bwd - penalty)

        best = max(range(len(candidates)), key=lambda i: scores[i])
        logger.info(
            "Short-text rerank %r -> %r | %s",
            text,
            candidates[best],
            ", ".join(
                f"{c!r}:{sc:.2f}(f={f:.2f},b={b:.2f})"
                for c, sc, f, b in sorted(
                    zip(candidates, scores, forward, backward), key=lambda x: -x[1]
                )[:6]
            ),
        )
        return candidates[best]

    def _sequence_logprobs(self, sources: list[str], targets: list[str]) -> list[float]:
        """Length-normalised log P(target | source) for each pair (teacher forcing, incl. EOS)."""
        import torch

        device = self._device()
        scores: list[float] = []
        for start in range(0, len(sources), 16):
            src_batch = sources[start : start + 16]
            tgt_batch = targets[start : start + 16]
            enc = self.tokenizer(src_batch, return_tensors="pt", padding=True)
            lab = self.tokenizer(tgt_batch, return_tensors="pt", padding=True)
            label_ids = lab.input_ids.to(device)
            label_mask = lab.attention_mask.to(device).bool()
            logits = self.model(
                input_ids=enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
                labels=label_ids.masked_fill(~label_mask, -100),
            ).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            token_lp = logp.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
            token_lp = token_lp.masked_fill(~label_mask, 0.0)
            lengths = label_mask.sum(-1).clamp(min=1)
            scores.extend((token_lp.sum(-1) / lengths).tolist())
            del logits, logp, token_lp, enc, lab, label_ids, label_mask
        return scores

    def _translate_units(
        self,
        units: list[Unit],
        *,
        target_language: str,
        num_beams: int,
        length_penalty: float,
        max_new_tokens: int,
        issues_out: list[frozenset[str]] | None = None,
    ) -> list[str]:
        """Decode units in padded batches, check every row, retry the bad ones once.

        * a row that stopped without EOS is retried with a larger budget;
        * a repetition loop is retried with beam search + no_repeat_ngram_size=4, and cut
          back to one copy if it still runs into the budget;
        * output in the wrong script (new Japanese text for ja->vi) or far too short for
          its source is retried with 4 beams when it was decoded with fewer (with 4+
          the retry would repeat the same search); the retry wins only if it is better;
        * a packed unit (several spaced-script sentences) is re-translated sentence by
          sentence when its output has an issue, misses 2+ sentences, or is clearly too
          short for its source (``generation.pack_incomplete``).

        ``issues_out``, when given, receives the issues each output still has.
        """
        texts = [unit.text for unit in units]
        first = self._translate_batch(
            texts,
            target_language=target_language,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
        )
        results: list[GenResult] = list(first)
        issues = [assess(text, res, target_language) for text, res in zip(texts, first)]

        fallback: list[int] = []
        retries: dict[str, list[int]] = {}
        for i, unit in enumerate(units):
            if unit.parts:
                why = pack_incomplete(
                    len(unit.parts), count_sentences(results[i].text), results[i], issues[i]
                )
                if why:
                    logger.info(
                        "MT pack of %d sentences re-translated one by one (%s%s): %r",
                        len(unit.parts),
                        why,
                        f" {sorted(issues[i])}" if issues[i] else "",
                        texts[i][:60],
                    )
                    fallback.append(i)
                continue
            kind = retry_kind(issues[i], results[i], max_new_tokens)
            if kind == "beam" and num_beams >= _RETRY_BEAMS:
                kind = None  # same beams, budget and settings: same output again
            if kind:
                retries.setdefault(kind, []).append(i)

        for kind, idxs in retries.items():
            again = self._translate_batch(
                [texts[i] for i in idxs],
                target_language=target_language,
                num_beams=max(num_beams, _RETRY_BEAMS) if kind in ("loop", "beam") else num_beams,
                length_penalty=length_penalty,
                max_new_tokens=max_new_tokens,
                longer=kind == "longer",
                no_repeat_ngram_size=4 if kind == "loop" else 0,
            )
            for i, res in zip(idxs, again):
                new_issues = assess(texts[i], res, target_language)
                if kind == "longer":
                    # Still cut off, but further along: the longer text keeps more.
                    kept = severity(new_issues) > severity(issues[i])
                else:
                    kept = severity(new_issues) >= severity(issues[i])
                logger.info(
                    "MT retry (%s) %r: %s -> %s, kept %s",
                    kind,
                    texts[i][:60],
                    sorted(issues[i]),
                    sorted(new_issues),
                    "first" if kept else "retry",
                )
                if not kept:
                    results[i], issues[i] = res, new_issues

        outputs: list[str] = []
        final_issues: list[frozenset[str]] = []
        for i, res in enumerate(results):
            text = res.text
            final_issues.append(frozenset() if i in fallback else issues[i])
            if i not in fallback and issues[i]:
                if {"loop", "truncated"} <= issues[i]:
                    text = trim_loop(text)
                logger.warning(
                    "MT output still %s (src=%d tok, out=%d tok, limit=%d): %r -> %r",
                    sorted(issues[i]),
                    res.src_len,
                    res.out_tokens,
                    res.limit,
                    texts[i][:80],
                    text[:80],
                )
            outputs.append(text)

        if fallback:
            parts = [part for i in fallback for part in units[i].parts]
            logger.info(
                "MT pack fallback: %d packed units -> %d sentences", len(fallback), len(parts)
            )
            part_issues: list[frozenset[str]] = []
            decoded = self._translate_units(
                parts,
                target_language=target_language,
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_new_tokens=max_new_tokens,
                issues_out=part_issues,
            )
            pos = 0
            for i in fallback:
                n = len(units[i].parts)
                outputs[i] = join_units(list(units[i].parts), decoded[pos : pos + n], target_language)
                final_issues[i] = frozenset().union(*part_issues[pos : pos + n])
                pos += n
        if issues_out is not None:
            issues_out.extend(final_issues)
        return outputs

    def _translate_pivot(
        self,
        units: list[Unit],
        *,
        target_language: str,
        num_beams: int,
        length_penalty: float,
        max_new_tokens: int,
    ) -> list[str]:
        """source -> pivot -> target, one output per unit (see ``pivot.choose_pivot``).

        Stage 1 decodes the units into the pivot language with all of
        ``_translate_units``' checks (retries, pack fallback). Each intermediate text is
        then segmented like a request in the pivot language (sentences, packs, long
        pieces), all of them are decoded into the target in one batched pass, and each
        unit's pieces are joined back. A unit whose intermediate came back empty is
        translated directly, and so is one whose English is still flagged after the
        retries (cut off, looping, wrong script, far too short): the second stage would
        only carry the loss on, and at 4+ beams ``_translate_units`` does not retry a
        "short" / "script" row.
        """
        pivot_language = PIVOT_LANGUAGE
        started = time.perf_counter()
        decoding = dict(num_beams=num_beams, length_penalty=length_penalty, max_new_tokens=max_new_tokens)
        stage1_issues: list[frozenset[str]] = []
        middle = self._translate_units(
            units, target_language=pivot_language, issues_out=stage1_issues, **decoding
        )
        stage1_time = time.perf_counter() - started

        stage: list[Unit] = []
        spans: list[tuple[int, int] | None] = []
        direct: list[int] = []
        for i, mid in enumerate(middle):
            if stage1_issues[i]:
                logger.info(
                    "MT pivot: %s step still %s, translating directly: %r -> %r",
                    pivot_language,
                    sorted(stage1_issues[i]),
                    units[i].text[:60],
                    mid[:60],
                )
                direct.append(i)
                spans.append(None)
                continue
            sub = self._segment(mid, target_language) if mid.strip() else []
            if not sub:
                direct.append(i)
                spans.append(None)
                continue
            spans.append((len(stage), len(stage) + len(sub)))
            stage.extend(sub)

        todo = [k for k, unit in enumerate(stage) if unit.translate]
        stage_out = [unit.text for unit in stage]  # symbols / numbers only: copied
        if todo:
            for k, out in zip(
                todo,
                self._translate_units(
                    [stage[k] for k in todo], target_language=target_language, **decoding
                ),
            ):
                stage_out[k] = out

        if len(direct) < len(units):
            self.last_pivot = pivot_language
        stage2_time = time.perf_counter() - started - stage1_time
        outputs: list[str] = []
        for i, span in enumerate(spans):
            if span is None:
                outputs.append("")
                continue
            a, b = span
            outputs.append(join_units(stage[a:b], stage_out[a:b], target_language))
            logger.debug("MT pivot %r -> %r -> %r", units[i].text[:60], middle[i][:60], outputs[-1][:60])
        if direct:
            logger.warning(
                "MT pivot (%s): %d of %d units translated directly instead",
                pivot_language,
                len(direct),
                len(units),
            )
            for i, out in zip(
                direct,
                self._translate_units(
                    [units[i] for i in direct], target_language=target_language, **decoding
                ),
            ):
                outputs[i] = out
        logger.info(
            "MT pivot via %s: %d units -> %d %s pieces -> %s (stage 1 %.1fs, stage 2 %.1fs, "
            "direct fallback %.1fs)",
            pivot_language,
            len(units),
            len(stage),
            pivot_language,
            target_language,
            stage1_time,
            stage2_time,
            time.perf_counter() - started - stage1_time - stage2_time,
        )
        return outputs

    def _translate_batch(
        self,
        batch: list[str],
        *,
        target_language: str,
        num_beams: int,
        length_penalty: float,
        max_new_tokens: int,
        longer: bool = False,
        no_repeat_ngram_size: int = 0,
    ) -> list[GenResult]:
        """Generate ``batch`` in padded chunks of up to ``batch_size`` rows.

        Rows are sorted by length (little padding) and capped by ``batch_max_tokens``
        KV-cache slots. T5 uses relative positions and the attention mask hides the right
        padding, so a row decodes as it would alone (fp32: identical; bf16: rounding-level
        differences only). Each row gets max(64, 2c+32) new tokens (c = source tokens),
        or 4c+64 when ``longer``; ``max_new_tokens`` caps both.
        """
        import torch

        tag = f"<2{target_language}>"
        inputs = [f"{tag} {text}" for text in batch]
        lengths = [self._token_len(text) for text in inputs]
        budget = retry_budget if longer else output_budget
        budgets = [budget(n, max_new_tokens) for n in lengths]
        beams = max(1, min(num_beams, 5))
        results: list[GenResult | None] = [None] * len(batch)
        plan = plan_batches(
            lengths,
            budgets,
            batch_size=self.batch_size,
            num_beams=beams,
            max_batch_tokens=self.batch_max_tokens,
        )
        queue = list(reversed(plan))
        while queue:
            rows = queue.pop()
            limit = max(budgets[i] for i in rows)
            try:
                generated = self._generate(
                    [inputs[i] for i in rows],
                    max_new_tokens=limit,
                    num_beams=beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            except Exception as exc:
                if len(rows) == 1 or not _is_oom(exc):
                    raise
                # Shared 12GB card: another process grabbed memory. Halve and go on.
                logger.warning("MT batch of %d rows hit CUDA OOM; splitting it", len(rows))
                self._free_cuda()
                half = len(rows) // 2
                queue.extend([rows[half:], rows[:half]])
                continue
            for i, (text, finished, out_tokens) in zip(rows, generated):
                results[i] = GenResult(text, finished, out_tokens, lengths[i], limit)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [res for res in results if res is not None]

    def _generate(
        self,
        inputs: list[str],
        *,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int = 0,
    ) -> list[tuple[str, bool, int]]:
        """One padded generate() call -> [(text, finished with EOS, generated tokens)]."""
        device = self._device()
        enc = self.tokenizer(inputs, return_tensors="pt", padding=True)
        gen_kwargs: dict = {
            "input_ids": enc.input_ids.to(device),
            "attention_mask": enc.attention_mask.to(device),
            "max_new_tokens": max_new_tokens,
        }
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["length_penalty"] = length_penalty
            gen_kwargs["do_sample"] = False
        if no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        outputs = self.model.generate(**gen_kwargs)
        eos_id = self.model.config.eos_token_id
        pad_id = self.model.config.pad_token_id
        rows = outputs.tolist()
        decoded = self.tokenizer.batch_decode(rows, skip_special_tokens=True)
        del outputs, enc, gen_kwargs
        out: list[tuple[str, bool, int]] = []
        for row, text in zip(rows, decoded):
            finished, n_out = row_stats(row, eos_id, pad_id)
            out.append((text.strip(), finished, n_out))
        return out

    def _segment(self, text: str, target_language: str) -> list[Unit]:
        """Lines -> sentences -> units (see ``segmenter.split_text``).

        Token budgets are measured with the real target tag. Spaced-script sentences are
        packed up to ``segment_pack_tokens`` unless the target has no sentence
        punctuation to check a pack's output against (Thai, Lao).
        """
        tag = f"<2{target_language}>"
        pack = self.segment_pack_tokens
        if target_base(target_language) in _NO_SENTENCE_PUNCT_TARGETS:
            pack = 0
        return split_text(
            text,
            token_len=lambda s: self._token_len(f"{tag} {s}"),
            max_tokens=self.max_input_tokens,
            clause_tokens=self.segment_clause_tokens,
            pack_tokens=pack,
        )

    def _rerank_lines(self, units: list[Unit]) -> frozenset[int]:
        """Lines whose whole-line unit may get the short-text rerank.

        As before segmentation existed: a line that is a blank-line block of its own (the
        whole input "kite", a title paragraph), or a line of a block whose every line is a
        short term (a vocabulary list "kite\\ncape\\nkit"). Short lines of a letter or a
        paragraph ("Hi John,", "Best regards,", "Anna") keep plain decoding.
        """
        sources = line_sources(units)
        blocks: dict[int, list[int]] = {}
        for unit in units:
            lines = blocks.setdefault(unit.block, [])
            if unit.line not in lines:
                lines.append(unit.line)
        eligible: set[int] = set()
        for lines in blocks.values():
            text = "\n".join(sources[line] for line in lines)
            if len(lines) == 1 or is_word_list(text, self.short_text_max_words):
                eligible.update(lines)
        return frozenset(eligible)

    @staticmethod
    def _lookup_lines(
        units: list[Unit], outputs: list[str | None], lookup: Callable[[str], str | None]
    ) -> None:
        """Glossary entries match whole lines too, even when the line was split."""
        members: dict[int, list[int]] = {}
        for i, unit in enumerate(units):
            members.setdefault(unit.line, []).append(i)
        sources = line_sources(units)
        for line_no, idxs in members.items():
            if len(idxs) < 2:
                continue  # a one-unit line is looked up per unit
            hit = lookup(sources[line_no])
            if hit is not None:
                outputs[idxs[0]] = hit
                for i in idxs[1:]:
                    outputs[i] = ""

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def release_vram(self) -> None:
        """Drop the GPU model so Comfy/LTX/TTS can use the 12GB card (reload on next job)."""
        self.model = None
        self.tokenizer = None
        self.sp = None
        self._ready = False
        self._free_cuda()
        logger.info("Unloaded MADLAD from GPU (%s)", self._vram_log())

    def _device(self):
        import torch

        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _patch_t5_tied_weights() -> None:
        """Transformers 5.x can bind MADLAD 3B encoder embeddings to lm_head (silent garbage)."""
        import transformers
        from transformers import T5ForConditionalGeneration

        major = int(str(transformers.__version__).split(".", 1)[0])
        keys = getattr(T5ForConditionalGeneration, "_tied_weights_keys", None)
        if major < 5 or not isinstance(keys, dict):
            return
        T5ForConditionalGeneration._tied_weights_keys = {
            "encoder.embed_tokens.weight": "shared.weight",
            "decoder.embed_tokens.weight": "shared.weight",
            "lm_head.weight": "shared.weight",
        }
        logger.warning(
            "Patched T5 tied-weight order for transformers %s (MADLAD-400 3B)",
            transformers.__version__,
        )

    def _assert_encoder_embeddings(self) -> None:
        """Refuse to serve if encoder input embeddings were loaded from lm_head."""
        import torch
        import transformers

        encoder_w = self.model.get_encoder().embed_tokens.weight
        decoder_w = self.model.get_decoder().embed_tokens.weight
        lm_w = self.model.lm_head.weight
        if encoder_w.device.type == "meta" or decoder_w.device.type == "meta":
            raise RuntimeError(
                "MADLAD weights are still on the meta device. "
                "Load without device_map='auto' (this engine now uses model.to(cuda))."
            )
        encoder = encoder_w.detach().float().cpu()
        decoder = decoder_w.detach().float().cpu()
        lm_head = lm_w.detach().float().cpu()
        enc_from_lm = torch.equal(encoder, lm_head)
        enc_from_dec = torch.equal(encoder, decoder)
        logger.info(
            "Embedding check enc_absmax=%.2f lm_absmax=%.2f enc==decoder=%s enc==lm_head=%s",
            float(encoder.abs().max()),
            float(lm_head.abs().max()),
            enc_from_dec,
            enc_from_lm,
        )
        if enc_from_lm and not enc_from_dec:
            raise RuntimeError(
                "MADLAD-400 3B encoder embeddings were loaded from lm_head "
                f"(transformers {transformers.__version__} bug). "
                "Install transformers 4.57.x: pip install 'transformers>=4.44.0,<5'"
            )

    def _resolve_device(self, torch) -> str:
        requested = (self.device_map or "auto").strip()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        if requested in {"auto", "cuda"}:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return requested

    def _resolve_dtype(self, torch, device: str):
        name = (self.dtype_name or "auto").lower().strip()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if name in {"", "auto", "none", "default"}:
            # fp32 3B (~12GB weights) does not fit the shared card; bf16 is ~6GB.
            return torch.float32 if str(device) == "cpu" else torch.bfloat16
        dtype = mapping.get(name)
        if dtype is None:
            logger.warning("Unknown MT_DTYPE=%s; using bfloat16", name)
            dtype = torch.bfloat16
        if str(device) == "cpu" and dtype in {torch.bfloat16, torch.float16}:
            return torch.float32
        if dtype == torch.float16:
            logger.warning("float16 T5 is unstable; prefer bfloat16")
        return dtype

    @staticmethod
    def _vram_log() -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return f"cuda allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"
        except Exception:
            return "unknown"

    @staticmethod
    def _free_cuda() -> None:
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
