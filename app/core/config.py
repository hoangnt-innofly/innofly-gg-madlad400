from functools import lru_cache
from pathlib import Path

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
    # After each job, park weights in RAM so the 12GB card is free for LTX/TTS/Comfy.
    # Translate still runs on GPU. Set 0 only if MT owns the GPU.
    mt_free_vram: bool = True

    mt_default_source_language: str = DEFAULT_SOURCE_LANGUAGE
    mt_default_target_language: str = DEFAULT_TARGET_LANGUAGE
    mt_default_num_beams: int = 1
    mt_default_length_penalty: float = 1.0
    mt_default_max_new_tokens: int = 512
    mt_default_max_input_tokens: int = 480
    mt_default_batch_size: int = 4

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
