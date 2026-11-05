"""Control-plane wiring: config, DB, storage, scheduler and job submission."""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import layout
from .db import DB, new_id
from .scheduler import Policy, Scheduler
from .storage import Storage, get_json, iter_lines, open_storage, put_json

EXCLUDE_DIRS = {
    ".git",
    ".gpuo",
    ".gpuo-standalone",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "wandb",
    "mlruns",
}
MAX_BUNDLE_FILE_MB = 50


def default_home() -> Path:
    return Path(os.environ.get("GPUO_HOME", Path.home() / ".gpuo")).expanduser()


@dataclass
class App:
    home: Path
    config: dict[str, Any]
    db: DB
    store: Storage
    scheduler: Scheduler

    @property
    def owner_id(self) -> str:
        return self.db.ensure_owner(self.config["owner"])


def init_home(home: Path | None = None, storage: str | None = None, owner: str | None = None) -> Path:
    home = home or default_home()
    home.mkdir(parents=True, exist_ok=True)
    cfg_path = home / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.setdefault("storage", f"file://{home / 'storage'}")
    cfg.setdefault("owner", os.environ.get("USER", "me"))
    if storage:
        cfg["storage"] = storage
    if owner:
        cfg["owner"] = owner
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return home


def load(home: Path | None = None, policy: Policy | None = None) -> App:
    home = home or default_home()
    cfg_path = home / "config.json"
    if not cfg_path.exists():
        init_home(home)
    cfg = json.loads(cfg_path.read_text())
    db = DB(home / "gpuo.db")
    store = open_storage(cfg["storage"])
    db.ensure_owner(cfg["owner"])
    return App(home, cfg, db, store, Scheduler(db, store, home, policy))


# ---- submission ------------------------------------------------------------------


def make_bundle(src: Path) -> bytes:
    src = src.resolve()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".gpuo")]
            for f in files:
                p = Path(root) / f
                if p.is_symlink():
                    continue
                if p.stat().st_size > MAX_BUNDLE_FILE_MB * 1024 * 1024:
                    raise ValueError(
                        f"{p.relative_to(src)} is over {MAX_BUNDLE_FILE_MB} MB. Put datasets in "
                        "storage or a Kaggle Dataset, not in the code bundle."
                    )
                tar.add(p, arcname=str(p.relative_to(src)))
    return buf.getvalue()


def submit(
    app: App,
    entrypoint: str,
    src: Path,
    name: str | None = None,
    min_vram_gb: float = 0,
    est_gpu_hours: float | None = None,
    priority: int = 0,
    config: dict[str, Any] | None = None,
) -> str:
    src = Path(src)
    if not (src / entrypoint.split()[0]).exists():
        raise FileNotFoundError(f"{entrypoint.split()[0]} not found in {src}")
    job_id = new_id("job")
    data = make_bundle(src)
    app.store.put_bytes(layout.bundle(job_id), data)
    cfg = dict(config or {})
    put_json(
        app.store,
        layout.job_json(job_id),
        {
            "id": job_id,
            "entrypoint": entrypoint,
            "config": cfg,
            "submitted_at": time.time(),
        },
    )
    app.db.add_job(
        app.owner_id,
        name or Path(entrypoint.split()[0]).stem,
        entrypoint,
        layout.bundle(job_id),
        cfg,
        min_vram_gb,
        est_gpu_hours,
        priority,
        job_id=job_id,
    )
    return job_id


# ---- read models for CLI / dashboard -------------------------------------------------


def job_view(app: App, job_id: str) -> dict[str, Any]:
    job = app.db.job(job_id)
    segs = app.db.segments(job_id=job_id)
    targets = {t["id"]: t for t in app.db.targets()}
    man = get_json(app.store, layout.manifest(job_id), {}) or {}
    for s in segs:
        s["target_name"] = targets.get(s["target_id"], {}).get("name")
        hb = get_json(app.store, layout.heartbeat(job_id, s["id"]))
        s["heartbeat"] = hb
    return {
        "job": job,
        "segments": segs,
        "checkpoints": man.get("history", []),
        "latest_checkpoint": man.get("latest"),
    }


def metrics(app: App, job_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in app.db.segments(job_id=job_id):
        for rec in iter_lines(app.store, layout.metrics_dir(job_id, seg["id"])):
            rec["segment"] = seg["id"]
            out.append(rec)
    out.sort(key=lambda r: r.get("t", 0))
    return out[-limit:]


def logs(app: App, job_id: str, segment_id: str | None = None, max_chars: int = 20000) -> str:
    segs = app.db.segments(job_id=job_id)
    if segment_id:
        segs = [s for s in segs if s["id"] == segment_id]
    parts = []
    for s in segs:
        prefix = "/".join(["jobs", job_id, "segments", s["id"], "logs"])
        body = "".join(app.store.get_bytes(k).decode(errors="replace") for k in app.store.list(prefix))
        if body:
            parts.append(f"===== {s['id']} on {s['target_id']} ({s['status']}) =====\n{body}")
    text = "\n".join(parts)
    return text[-max_chars:]


def targets_view(app: App) -> list[dict[str, Any]]:
    out = []
    for t in app.db.targets():
        rem = app.scheduler.remaining_hours(t)
        out.append(
            {
                **t,
                "remaining_h": None if rem == float("inf") else round(rem, 2),
                "usable_h": round(max(0.0, app.scheduler.usable_hours(t)), 2),
                "busy": app.scheduler.busy(t),
            }
        )
    return out
