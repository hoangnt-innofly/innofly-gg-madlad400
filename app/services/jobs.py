from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.core.config import Settings
from app.core.languages import canonicalize_language, detect_language
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
    chunks: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_public(self, base_url: str) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "text": self.text,
            "translation": self.translation,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "detected_language": self.detected_language,
            "model": "google/madlad400-3b-mt",
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
            try:
                await asyncio.to_thread(self._run_job, job)
                job.status = "succeeded"
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                job.status = "failed"
                job.error = str(exc)
                if exc.__cause__ and str(exc.__cause__) not in job.error:
                    job.error = f"{exc} | {exc.__cause__}"
                self._free_cuda()
            finally:
                job.finished_at = datetime.now(timezone.utc)
                job._done.set()
                self._queue.task_done()

    def _run_job(self, job: Job) -> None:
        detected = detect_language(job.text)
        job.detected_language = detected
        if job.source_language == "auto":
            job.source_language = detected

        if job.source_language == job.target_language:
            job.translation = job.text
            job.chunks = 1
            return

        if self.settings.mt_mock:
            job.translation = mock_translate(job.text, target_language=job.target_language)
            job.chunks = 1
            return

        engine = self._get_engine()
        translation, chunks = engine.translate(
            text=job.text,
            target_language=job.target_language,
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
                )
                try:
                    self._engine.load()
                except Exception:
                    self._engine = None
                    raise
            return self._engine
