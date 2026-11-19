from __future__ import annotations

import gc
import logging
import re
from typing import Callable

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

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")
_WHITESPACE_ONLY = re.compile(r"^\s*$")


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
        self.model = None
        self.tokenizer = None
        self.sp = None
        self._ready = False

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
                "MADLAD-400 3B MT ready (%s) pad=%s eos=%s start=%s probe=%s",
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
    ) -> tuple[str, int]:
        """Translate ``text``; returns (translation, number of segments).

        ``source_language`` is optional (MADLAD only needs the target tag); when known it
        enables back-translation scoring for short input. ``lookup`` is a glossary hook
        tried on every segment before the model.
        """
        if not self._ready:
            self.load()

        import torch

        paragraphs = self._segment(text)
        if not paragraphs:
            return "", 0

        out_paragraphs: list[str] = []
        segments = 0
        reranked = 0
        try:
            with torch.no_grad():
                for units, joiner in paragraphs:
                    out_units: list[str] = []
                    for unit in units:
                        segments += 1
                        hit = lookup(unit) if lookup else None
                        if hit is not None:
                            out_units.append(hit)
                        elif (
                            self.short_text_rerank
                            and reranked < self.short_text_max_units
                            and is_short_text(unit, self.short_text_max_words)
                        ):
                            reranked += 1
                            out_units.append(
                                self._translate_short(
                                    unit,
                                    target_language=target_language,
                                    source_language=source_language,
                                    max_new_tokens=max_new_tokens,
                                )
                            )
                        else:
                            out_units.extend(
                                self._translate_batch(
                                    [unit],
                                    target_language=target_language,
                                    num_beams=num_beams,
                                    length_penalty=length_penalty,
                                    max_new_tokens=max_new_tokens,
                                )
                            )
                    out_paragraphs.append(joiner.join(out_units))
            if self.short_text_rerank and reranked >= self.short_text_max_units > 0:
                logger.info(
                    "Short-text rerank capped at %d units (MT_SHORT_TEXT_MAX_UNITS); "
                    "remaining short lines used plain decoding",
                    self.short_text_max_units,
                )
            return "\n\n".join(out_paragraphs), segments
        finally:
            if self.free_vram:
                self.release_vram()
            else:
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

    def _translate_batch(
        self,
        batch: list[str],
        *,
        target_language: str,
        num_beams: int,
        length_penalty: float,
        max_new_tokens: int,
    ) -> list[str]:
        import torch

        decoded: list[str] = []
        for chunk in batch:
            text = f"<2{target_language}> {chunk}"
            # Same as the Hub card: tokenizer(text).input_ids.to(model.device)
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self._device())
            src_len = int(input_ids.shape[-1])
            gen_kwargs: dict = {
                "input_ids": input_ids,
                "max_new_tokens": min(max_new_tokens, max(48, src_len + 16)),
            }
            if num_beams > 1:
                gen_kwargs["num_beams"] = max(1, min(num_beams, 5))
                gen_kwargs["length_penalty"] = length_penalty
                gen_kwargs["do_sample"] = False
            outputs = self.model.generate(**gen_kwargs)
            decoded.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
            del outputs, input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return decoded

    def _segment(self, text: str) -> list[tuple[list[str], str]]:
        """Split into paragraphs of translation units, each with the joiner to rebuild it.

        * a paragraph that is a list of short terms (one per line) is translated line by
          line, so "kite\\nkit\\ncape" is not fed to the model as one run-on phrase;
        * otherwise a paragraph is one unit if it fits the encoder, else packed sentences.
        Paragraphs are rejoined with a blank line.
        """
        paragraphs: list[tuple[list[str], str]] = []
        for block in re.split(r"\n\s*\n", text):
            if _WHITESPACE_ONLY.match(block):
                continue
            stripped = block.strip()
            if self.short_text_rerank and is_word_list(stripped, self.short_text_max_words):
                lines = [line.strip() for line in stripped.splitlines() if line.strip()]
                paragraphs.append((lines, "\n"))
                continue
            if self._token_len(f"<2en> {stripped}") <= self.max_input_tokens:
                paragraphs.append(([stripped], " "))
                continue
            joiner = "\n" if "\n" in stripped else " "
            paragraphs.append((self._pack_sentences(stripped), joiner))
        return paragraphs

    def _pack_sentences(self, paragraph: str) -> list[str]:
        sentences = [part.strip() for part in _SENTENCE_SPLIT.split(paragraph) if part.strip()]
        if not sentences:
            return self._hard_split(paragraph)

        packed: list[str] = []
        current = ""
        prefix_budget = self.max_input_tokens
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if self._token_len(f"<2en> {candidate}") <= prefix_budget:
                current = candidate
                continue
            if current:
                packed.append(current)
            if self._token_len(f"<2en> {sentence}") <= prefix_budget:
                current = sentence
            else:
                packed.extend(self._hard_split(sentence))
                current = ""
        if current:
            packed.append(current)
        return packed

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return [text]
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and self._token_len(f"<2en> {candidate}") > self.max_input_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

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
