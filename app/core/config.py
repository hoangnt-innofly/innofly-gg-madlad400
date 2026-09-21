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
    mt_device_map: str = "auto"
    mt_dtype: str = "auto"
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

    max_text_chars: int = 20000
    job_ttl_seconds: int = 86400


@lru_cache
def get_settings() -> Settings:
    return Settings()
