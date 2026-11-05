"""Local dashboard + background scheduler. `gpuo serve` -> http://127.0.0.1:8787"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .app import job_view, load, logs, metrics, targets_view

STATIC = Path(__file__).parent / "static"


def create_app(interval: float = 30, home: Path | None = None) -> FastAPI:
    app_state = load(home)
    lock = threading.Lock()
    stop = threading.Event()
    last_tick: dict[str, Any] = {"t": None, "result": None, "error": None}

    def loop() -> None:
        while not stop.is_set():
            with lock:
                try:
                    last_tick["result"] = app_state.scheduler.tick()
                    last_tick["error"] = None
                except Exception as e:  # keep the loop alive; surface the error in the UI
                    last_tick["error"] = repr(e)
                last_tick["t"] = time.time()
            stop.wait(interval)

    api = FastAPI(title="gpuo", version="0.1.0")

    @api.on_event("startup")
    def _start() -> None:
        if interval > 0:
            threading.Thread(target=loop, daemon=True, name="gpuo-scheduler").start()

    @api.on_event("shutdown")
    def _stop() -> None:
        stop.set()

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @api.get("/api/overview")
    def overview() -> dict[str, Any]:
        with lock:
            return {
                "owner": app_state.config["owner"],
                "storage": app_state.config["storage"],
                "scheduler": last_tick,
                "targets": targets_view(app_state),
                "jobs": app_state.db.jobs(),
            }

    @api.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        with lock:
            try:
                return job_view(app_state, job_id)
            except KeyError:
                raise HTTPException(404, "no such job") from None

    @api.get("/api/jobs/{job_id}/metrics")
    def job_metrics(job_id: str) -> list[dict[str, Any]]:
        with lock:
            return metrics(app_state, job_id)

    @api.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: str) -> dict[str, str]:
        with lock:
            return {"text": logs(app_state, job_id)}

    @api.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, str]:
        with lock:
            app_state.scheduler.cancel(job_id)
        return {"status": "cancelled"}

    @api.post("/api/tick")
    def tick() -> dict[str, Any]:
        with lock:
            return app_state.scheduler.tick()

    return api
