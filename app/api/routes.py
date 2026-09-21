from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from app.core.auth import require_api_key
from app.core.config import get_settings
from app.core.languages import (
    LANGUAGES,
    canonicalize_language,
    detect_language,
    languages_payload,
)
from app.models.schemas import DetectResponse, HealthResponse, JobResponse, LanguagesResponse
from app.services.jobs import JobService

router = APIRouter()
api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
settings = get_settings()
jobs = JobService(settings)

_LANG_BY_CODE = {lang.code: lang for lang in LANGUAGES}


def public_base(request: Request) -> str:
    configured = settings.public_base_url.rstrip("/")
    if configured:
        return configured
    return str(request.base_url).rstrip("/")


def to_response(request: Request, job) -> JobResponse:
    return JobResponse.model_validate(job.to_public(public_base(request)))


async def _await_job(job, timeout: float = 600):
    try:
        job = await jobs.wait(job.id, timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise HTTPException(status_code=504, detail="Translation timed out") from exc
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Translation failed")
    return job


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cuda_available, device_name = jobs.cuda_info()
    return HealthResponse(
        mock=settings.mt_mock,
        cuda_available=cuda_available,
        device_name=device_name,
        pipeline_ready=jobs.pipeline_ready(),
        defaults={
            "model": settings.mt_model_id,
            "mode": "mt",
            "source_language": settings.mt_default_source_language,
            "target_language": settings.mt_default_target_language,
            "num_beams": settings.mt_default_num_beams,
            "length_penalty": settings.mt_default_length_penalty,
            "max_new_tokens": settings.mt_default_max_new_tokens,
            "max_input_tokens": settings.mt_default_max_input_tokens,
            "batch_size": settings.mt_default_batch_size,
            "device_map": settings.mt_device_map,
            "dtype": settings.mt_dtype,
            "free_vram": settings.mt_free_vram,
        },
    )


@api.get("/languages", response_model=LanguagesResponse)
def list_languages() -> LanguagesResponse:
    return LanguagesResponse.model_validate(languages_payload())


@api.post("/detect", response_model=DetectResponse)
def detect_source_language(text: Annotated[str, Form(min_length=1)]) -> DetectResponse:
    cleaned = (text or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="text is required")
    code = detect_language(cleaned)
    lang = _LANG_BY_CODE.get(code)
    return DetectResponse(
        language=code,
        name_vi=lang.name_vi if lang else None,
        name_en=lang.name_en if lang else None,
    )


@api.post("/free-memory")
def free_memory() -> dict:
    """Release MADLAD VRAM after use so other 12GB-card apps (Comfy, LTX, TTS) can run."""
    return jobs.free_vram()


@api.post("/translate", response_model=JobResponse)
async def translate(
    request: Request,
    text: Annotated[str, Form(min_length=1)],
    source_language: Annotated[Optional[str], Form()] = None,
    target_language: Annotated[Optional[str], Form()] = None,
    num_beams: Annotated[Optional[int], Form()] = None,
    length_penalty: Annotated[Optional[float], Form()] = None,
    max_new_tokens: Annotated[Optional[int], Form()] = None,
    wait: Annotated[bool, Form()] = True,
) -> JobResponse:
    """Machine translation with MADLAD-400 3B. Waits and returns translation."""
    job = await _enqueue(text, source_language, target_language, num_beams, length_penalty, max_new_tokens)
    if wait:
        job = await _await_job(job)
    return to_response(request, job)


@api.post("/jobs", response_model=JobResponse)
async def create_job(
    request: Request,
    text: Annotated[str, Form(min_length=1)],
    source_language: Annotated[Optional[str], Form()] = None,
    target_language: Annotated[Optional[str], Form()] = None,
    num_beams: Annotated[Optional[int], Form()] = None,
    length_penalty: Annotated[Optional[float], Form()] = None,
    max_new_tokens: Annotated[Optional[int], Form()] = None,
) -> JobResponse:
    """Same as /translate: wait until done, then return translation."""
    job = await _enqueue(text, source_language, target_language, num_beams, length_penalty, max_new_tokens)
    job = await _await_job(job)
    return to_response(request, job)


@api.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return to_response(request, job)


async def _enqueue(
    text: str,
    source_language: str | None,
    target_language: str | None,
    num_beams: int | None,
    length_penalty: float | None,
    max_new_tokens: int | None,
):
    cleaned = (text or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="text is required")
    if len(cleaned) > settings.max_text_chars:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds {settings.max_text_chars} characters",
        )

    try:
        if source_language is not None:
            source_language = canonicalize_language(source_language, allow_auto=True)
        if target_language is not None:
            target_language = canonicalize_language(target_language, allow_auto=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if num_beams is not None and not (1 <= num_beams <= 8):
        raise HTTPException(status_code=400, detail="num_beams must be between 1 and 8")
    if length_penalty is not None and not (0.5 <= length_penalty <= 2.0):
        raise HTTPException(status_code=400, detail="length_penalty must be between 0.5 and 2")
    if max_new_tokens is not None and not (16 <= max_new_tokens <= 1024):
        raise HTTPException(status_code=400, detail="max_new_tokens must be between 16 and 1024")

    return jobs.create_job(
        text=cleaned,
        source_language=source_language,
        target_language=target_language,
        num_beams=num_beams,
        length_penalty=length_penalty,
        max_new_tokens=max_new_tokens,
    )
