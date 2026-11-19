from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.core.config import ROOT_DIR, Settings
from app.core.languages import canonicalize_language, detect_language_info
from app.services.glossary import Glossary
from app.services.mock_engine import mock_translate

logger = logging.getLogger("madlad400-mt-api")

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    text: str
    source_language: str
    target_language: str
    num_beams: int
    length_penalty: float
    max_new_tokens: int
    status: JobStatus = "queued"
    translation: str | None = None
    detected_language: str | None = None
    model_id: str = "google/madlad400-3b-mt"
    chunks: int | None = None
    error: str | None = None
    progress: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_public(self, base_url: str) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "text": self.text,
            "translation": self.translation,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "detected_language": self.detected_language,
            "model": self.model_id,
            "mode": "mt",
            "num_beams": self.num_beams,
            "chunks": self.chunks,
            "error": self.error,
        }


class JobService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._engine = None
        self._engine_lock = threading.Lock()
        self._worker_task: asyncio.Task | None = None
        glossary_path = (settings.mt_glossary_path or "").strip()
        if glossary_path and not Path(glossary_path).is_absolute():
            glossary_path = str(ROOT_DIR / glossary_path)
        self.glossary = Glossary(glossary_path or None)

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop(), name="madlad400-mt-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def create_job(
        self,
        *,
        text: str,
        source_language: str | None,
        target_language: str | None,
        num_beams: int | None,
        length_penalty: float | None,
        max_new_tokens: int | None,
    ) -> Job:
        source = canonicalize_language(
            source_language or self.settings.mt_default_source_language,
            allow_auto=True,
        )
        target = canonicalize_language(
            target_language or self.settings.mt_default_target_language,
            allow_auto=False,
        )
        job_id = uuid.uuid4().hex
        job = Job(
            id=job_id,
            text=text.strip(),
            source_language=source,
            target_language=target,
            num_beams=self.settings.mt_default_num_beams if num_beams is None else num_beams,
            length_penalty=(
                self.settings.mt_default_length_penalty if length_penalty is None else length_penalty
            ),
            max_new_tokens=(
                self.settings.mt_default_max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
            model_id=self.settings.mt_model_id,
        )
        self.jobs[job_id] = job
        self._queue.put_nowait(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def wait(self, job_id: str, timeout: float = 600) -> Job:
        job = self.jobs[job_id]
        await asyncio.wait_for(job._done.wait(), timeout=timeout)
        return job

    def pipeline_ready(self) -> bool:
        if self.settings.mt_mock:
            return True
        return bool(self._engine and getattr(self._engine, "ready", False))

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def free_vram(self) -> dict:
        """Park MT weights off GPU so Comfy/LTX/TTS can use the 12GB card."""
        with self._engine_lock:
            if self._engine is not None:
                self._engine.release_vram()
            else:
                self._free_cuda()
        return {"freed": True}

    def cuda_info(self) -> tuple[bool, str | None]:
        try:
            import torch

            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False, None

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs[job_id]
            job.status = "running"
            job.progress = "starting"
            try:
                await asyncio.to_thread(self._run_job, job)
                job.status = "succeeded"
                job.progress = "done"
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                job.status = "failed"
                job.progress = "failed"
                job.error = str(exc)
                if exc.__cause__ and str(exc.__cause__) not in job.error:
                    job.error = f"{exc} | {exc.__cause__}"
                self._free_cuda()
            finally:
                if self.settings.mt_free_vram:
                    self.free_vram()
                job.finished_at = datetime.now(timezone.utc)
                job._done.set()
                self._queue.task_done()

    def _run_job(self, job: Job) -> None:
        job.progress = "detecting_language"
        detected, reliable = detect_language_info(job.text)
        job.detected_language = detected
        explicit_source = job.source_language != "auto"
        if not explicit_source:
            job.source_language = detected

        # Only skip on an explicit source or a trustworthy detection: langdetect guesses
        # on one or two words are noise, and skipping would echo the input untranslated.
        if job.source_language == job.target_language and (explicit_source or reliable):
            job.translation = job.text
            job.chunks = 1
            job.progress = "skipped_same_language"
            return

        # Unreliable guess: still usable as a back-translation hint, unless it equals the
        # target (then we really do not know the source language).
        source_hint = job.source_language
        if not explicit_source and not reliable and source_hint == job.target_language:
            source_hint = None

        def lookup(segment: str) -> str | None:
            return self.glossary.lookup(
                segment, source_language=source_hint, target_language=job.target_language
            )

        whole = lookup(job.text)
        if whole is not None:
            job.translation = whole
            job.chunks = 1
            job.progress = "glossary"
            return

        if self.settings.mt_mock:
            job.translation = mock_translate(job.text, target_language=job.target_language)
            job.chunks = 1
            return

        job.progress = "loading_model"
        engine = self._get_engine()
        job.progress = "translating"
        translation, chunks = engine.translate(
            text=job.text,
            target_language=job.target_language,
            source_language=source_hint,
            lookup=lookup,
            num_beams=job.num_beams,
            length_penalty=job.length_penalty,
            max_new_tokens=job.max_new_tokens,
        )
        job.translation = translation
        job.chunks = chunks

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                from app.services.mt_engine import MTEngine

                self._engine = MTEngine(
                    self.settings.mt_model_id,
                    device_map=self.settings.mt_device_map,
                    dtype=self.settings.mt_dtype,
                    free_vram=self.settings.mt_free_vram,
                    max_input_tokens=self.settings.mt_default_max_input_tokens,
                    batch_size=self.settings.mt_default_batch_size,
                    short_text_rerank=self.settings.mt_short_text_rerank,
                    short_text_max_words=self.settings.mt_short_text_max_words,
                    short_text_candidates=self.settings.mt_short_text_candidates,
                    short_text_backward_weight=self.settings.mt_short_text_backward_weight,
                    short_text_copy_penalty=self.settings.mt_short_text_copy_penalty,
                    short_text_repeat_penalty=self.settings.mt_short_text_repeat_penalty,
                    short_text_max_units=self.settings.mt_short_text_max_units,
                    segment_pack_tokens=self.settings.mt_segment_pack_tokens,
                    segment_clause_tokens=self.settings.mt_segment_clause_tokens,
                    batch_max_tokens=self.settings.mt_batch_max_tokens,
                )
            if not getattr(self._engine, "ready", False):
                try:
                    self._engine.load()
                except Exception:
                    self._engine = None
                    raise
            return self._engine
