from __future__ import annotations

import gc
import logging
import re

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
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype_name = dtype
        self.free_vram = free_vram
        self.max_input_tokens = max_input_tokens
        self.batch_size = max(1, batch_size)
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
        device_map = self._resolve_device_map(torch)
        dtype = self._resolve_dtype(torch, device_map)
        self._patch_t5_tied_weights()

        logger.info(
            "Loading %s (transformers=%s, device_map=%s, dtype=%s, free_vram=%s)",
            self.model_id,
            transformers.__version__,
            device_map,
            dtype if dtype is not None else "auto/fp32",
            self.free_vram,
        )
        try:
            # Same classes/call pattern as the google/madlad400-3b-mt card.
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
            load_kwargs: dict = {"device_map": device_map}
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
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
        num_beams: int = 4,
        length_penalty: float = 1.0,
        max_new_tokens: int = 512,
    ) -> tuple[str, int]:
        if not self._ready:
            self.load()

        import torch

        chunks = self._chunk_text(text)
        if not chunks:
            return "", 0

        translations: list[str] = []
        try:
            with torch.no_grad():
                for start in range(0, len(chunks), self.batch_size):
                    batch = chunks[start : start + self.batch_size]
                    translations.extend(
                        self._translate_batch(
                            batch,
                            target_language=target_language,
                            num_beams=num_beams,
                            length_penalty=length_penalty,
                            max_new_tokens=max_new_tokens,
                        )
                    )
            return self._rejoin(text, translations), len(chunks)
        finally:
            if self.free_vram:
                self.release_vram()
            else:
                self._free_cuda()

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
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
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

    def _chunk_text(self, text: str) -> list[str]:
        """Pack paragraphs / sentences under the encoder limit to keep document context."""
        paragraphs = re.split(r"(\n\s*\n)", text)
        chunks: list[str] = []
        for block in paragraphs:
            if _WHITESPACE_ONLY.match(block):
                continue
            stripped = block.strip()
            if not stripped:
                continue
            if self._token_len(f"<2en> {stripped}") <= self.max_input_tokens:
                chunks.append(stripped)
                continue
            packed = self._pack_sentences(stripped)
            chunks.extend(packed)
        return chunks

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

    @staticmethod
    def _rejoin(original: str, translations: list[str]) -> str:
        if not translations:
            return ""
        if len(translations) == 1:
            return translations[0]
        if re.search(r"\n\s*\n", original):
            return "\n\n".join(translations)
        if "\n" in original:
            return "\n".join(translations)
        return " ".join(translations)

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

        encoder = self.model.get_encoder().embed_tokens.weight.detach().float().cpu()
        decoder = self.model.get_decoder().embed_tokens.weight.detach().float().cpu()
        lm_head = self.model.lm_head.weight.detach().float().cpu()
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

    def _resolve_device_map(self, torch) -> str:
        requested = (self.device_map or "auto").strip()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        # Keep "auto": the Hub card uses device_map="auto" (not a forced cuda:0).
        if requested in {"auto", "cuda"}:
            return "auto" if torch.cuda.is_available() else "cpu"
        return requested

    def _resolve_dtype(self, torch, device_map: str):
        name = (self.dtype_name or "auto").lower().strip()
        if name in {"", "auto", "none", "default"}:
            return None
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = mapping.get(name)
        if dtype is None:
            logger.warning("Unknown MT_DTYPE=%s; using Hub default (fp32)", name)
            return None
        if str(device_map) == "cpu" and dtype in {torch.bfloat16, torch.float16}:
            return torch.float32
        if dtype == torch.float16:
            logger.warning("float16 T5 often emits garbage; Hub card uses default fp32")
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
