"""JobService: unload policy, /free-memory during a job, TTL pruning. No model, no GPU."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.core.config import Settings
from app.services.jobs import JobService


class FakeEngine:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.ready = True
        self.last_pivot = None
        self.releases = 0
        self.gate = gate
        self.calls: list[dict] = []

    def load(self) -> None:
        self.ready = True

    def release_vram(self) -> None:
        self.releases += 1
        self.ready = False

    def translate(self, **kwargs):
        self.calls.append(kwargs)
        if self.gate is not None:
            assert self.gate.wait(5), "test gate never opened"
        return f"[{kwargs['text']}]", 1


def make_service(engine: FakeEngine, **overrides) -> JobService:
    settings = Settings(_env_file=None, **{"mt_glossary_path": "", "mt_free_vram": True, **overrides})
    service = JobService(settings)
    service._engine = engine
    return service


def submit(service: JobService, text: str):
    return service.create_job(
        text=text,
        source_language="ja",
        target_language="vi",
        num_beams=None,
        length_penalty=None,
        max_new_tokens=None,
    )


def test_unload_only_when_the_queue_is_empty():
    engine = FakeEngine()
    service = make_service(engine)

    async def run():
        service.start()
        jobs = [submit(service, f"文{i}です。") for i in range(3)]
        for job in jobs:
            await service.wait(job.id, timeout=5)
        await service.stop()
        return jobs

    jobs = asyncio.run(run())
    assert [j.status for j in jobs] == ["succeeded"] * 3
    assert engine.releases == 1  # once, after the last queued job, not after each
    assert all(set(j.timings) >= {"queue", "load", "translate"} for j in jobs)
    # Routing gets the request source only when it is trustworthy (explicit here).
    assert all(c["source_reliable"] and c["pivot"] is None for c in engine.calls)


def test_free_memory_during_a_job_is_deferred_until_it_ends():
    gate = threading.Event()
    engine = FakeEngine(gate)
    service = make_service(engine, mt_free_vram=False)

    async def run():
        service.start()
        job = submit(service, "文です。")
        while not engine.calls:  # the job is inside translate()
            await asyncio.sleep(0.01)
        answer = service.free_vram()
        assert engine.releases == 0  # weights not pulled out from under generate()
        gate.set()
        await service.wait(job.id, timeout=5)
        await service.stop()
        return answer, job

    answer, job = asyncio.run(run())
    assert answer == {"freed": False, "deferred": True}
    assert job.status == "succeeded"
    assert engine.releases == 1  # applied after the job
    assert service.free_vram() == {"freed": True}


def test_finished_jobs_expire_after_the_ttl():
    service = make_service(FakeEngine(), job_ttl_seconds=60)
    old = submit(service, "古い")
    old.finished_at = datetime.now(timezone.utc) - timedelta(seconds=61)
    running = submit(service, "実行中")  # not finished: never pruned
    submit(service, "新しい")
    assert old.id not in service.jobs
    assert running.id in service.jobs


def test_timeout_answers_202_with_the_job_instead_of_504(monkeypatch):
    from app.api import routes

    job = SimpleNamespace(id="abc", status="running")

    async def slow_wait(job_id, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr(routes.jobs, "wait", slow_wait)
    response = SimpleNamespace(status_code=200)
    got = asyncio.run(routes._await_job(job, response, timeout=0.01))
    assert got is job and response.status_code == 202
