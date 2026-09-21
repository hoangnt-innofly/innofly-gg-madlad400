from __future__ import annotations

import gc
import logging
import re
from typing import Any

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
        dtype: str = "bfloat16",
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
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._ready:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.free_vram = self.free_vram and torch.cuda.is_available()
        device_map = self._resolve_device_map(torch)
        dtype = self._resolve_dtype(torch, device_map)

        logger.info(
            "Loading %s (device_map=%s, dtype=%s, free_vram=%s)",
            self.model_id,
            device_map,
            dtype,
            self.free_vram,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self._ready = True
            logger.info("MADLAD-400 3B MT ready (%s)", self._vram_log())
        except Exception as exc:
            self.model = None
            self.tokenizer = None
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
            with torch.inference_mode():
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

        prefixed = [f"<2{target_language}> {chunk}" for chunk in batch]
        enc = self.tokenizer(
            prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self._device())
        src_len = int(enc["input_ids"].shape[-1])
        gen_tokens = min(max_new_tokens, max(48, int(src_len * 2.0) + 16))

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_tokens,
            "num_beams": max(1, num_beams),
            "length_penalty": length_penalty,
            "early_stopping": True,
            "do_sample": False,
            "use_cache": True,
        }
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is not None:
            generate_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            generate_kwargs["eos_token_id"] = eos_id

        outputs = self.model.generate(**enc, **generate_kwargs)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        del outputs, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [item.strip() for item in decoded]

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
            if self._token_len(f"<2xx> {stripped}") <= self.max_input_tokens:
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
            if self._token_len(f"<2xx> {candidate}") <= prefix_budget:
                current = candidate
                continue
            if current:
                packed.append(current)
            if self._token_len(f"<2xx> {sentence}") <= prefix_budget:
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
            if current and self._token_len(f"<2xx> {candidate}") > self.max_input_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

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

    def _resolve_device_map(self, torch) -> str:
        requested = (self.device_map or "auto").strip()
        if requested in {"auto", "cuda", "cuda:0"} and torch.cuda.is_available():
            return "cuda:0"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return requested

    def _resolve_dtype(self, torch, device_map: str):
        name = (self.dtype_name or "bfloat16").lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = mapping.get(name, torch.bfloat16)
        if str(device_map) == "cpu" and dtype in {torch.bfloat16, torch.float16}:
            return torch.float32
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
