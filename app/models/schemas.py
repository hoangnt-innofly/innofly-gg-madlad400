from typing import Any

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    job_id: str
    status: str
    text: str
    translation: str | None = None
    source_language: str
    target_language: str
    detected_language: str | None = None
    model: str = "google/madlad400-3b-mt"
    mode: str = "mt"
    num_beams: int | None = None
    chunks: int | None = None
    error: str | None = None
    progress: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    mock: bool
    cuda_available: bool
    device_name: str | None = None
    pipeline_ready: bool
    defaults: dict[str, Any] = Field(default_factory=dict)


class LanguagesResponse(BaseModel):
    model: str
    mode: str = "mt"
    prefix: str = "<2xx>"
    languages: list[dict[str, str]]


class DetectResponse(BaseModel):
    language: str
    name_vi: str | None = None
    name_en: str | None = None
