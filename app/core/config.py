from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.languages import DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8002
    public_base_url: str = "http://127.0.0.1:8002"
    secret_api_key: str = ""

    mt_mock: bool = False
    mt_model_id: str = "google/madlad400-3b-mt"
    mt_device_map: str = "cuda:0"
    mt_dtype: str = "bfloat16"
    # Unload the model from the GPU once no job is waiting, so the 12GB card is free for
    # LTX/TTS/Comfy (the next request reloads it). Set 0 only if MT owns the GPU.
    mt_free_vram: bool = True

    mt_default_source_language: str = DEFAULT_SOURCE_LANGUAGE
    mt_default_target_language: str = DEFAULT_TARGET_LANGUAGE
    # Beam width when a request does not send num_beams. Read from MT_NUM_BEAMS, not
    # MT_DEFAULT_NUM_BEAMS: deployed .env files copied from the old example pin
    # MT_DEFAULT_NUM_BEAMS=1 (greedy), which mistranslates short Japanese sentences more
    # often, and 4 should apply without editing the server. The old name is only read to
    # warn that it is ignored.
    mt_default_num_beams: int = Field(default=4, ge=1, le=8, validation_alias="MT_NUM_BEAMS")
    legacy_default_num_beams: int | None = Field(
        default=None, validation_alias="MT_DEFAULT_NUM_BEAMS"
    )
    mt_default_length_penalty: float = 1.0
    mt_default_max_new_tokens: int = 512
    mt_default_max_input_tokens: int = 480
    # Sentences decoded together per generate() call (padded, sorted by length).
    mt_default_batch_size: int = 4
    # Cap per generate() call in KV-cache token slots: rows * beams * (source + output).
    # ~256 KiB per slot for MADLAD-3B in bf16, so 4096 is ~1 GiB.
    mt_batch_max_tokens: int = 4096

    # Segmentation. Every line and every sentence is its own model input (MADLAD drops
    # sentences from multi-sentence input). Consecutive sentences of a spaced script
    # (en, vi, de, ...) on one line are packed up to this many tokens because MADLAD keeps
    # them and uses the context; a pack whose output looks incomplete is re-translated
    # sentence by sentence. 0 = one sentence per input everywhere. CJK is never packed.
    mt_segment_pack_tokens: int = 256
    # Thai/Lao lines (no sentence punctuation) longer than this are cut at spaces.
    mt_segment_clause_tokens: int = 96

    # Lone words / tiny phrases ("kite", "kit", "cape"): MADLAD is sentence-level and greedy
    # decoding gives garbage ("đi đi") or copies the word. For input of at most
    # MT_SHORT_TEXT_MAX_WORDS words, generate several candidates and rerank them with a
    # forward + back-translation score. 0 / false disables it.
    mt_short_text_rerank: bool = True
    mt_short_text_max_words: int = 3
    mt_short_text_candidates: int = 8
    mt_short_text_backward_weight: float = 1.0
    mt_short_text_copy_penalty: float = 1.5
    mt_short_text_repeat_penalty: float = 2.0
    # Per request; further short lines (long vocabulary lists) use plain decoding.
    mt_short_text_max_units: int = 50

    # Fixed translations that override the model (re-read on change). Relative to project root.
    mt_glossary_path: str = "data/glossary.json"

    max_text_chars: int = 20000
    job_ttl_seconds: int = 86400


@lru_cache
def get_settings() -> Settings:
    return Settings()
