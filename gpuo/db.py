"""SQLite state for the control plane. Stdlib only."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS owner (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS target (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL REFERENCES owner(id),
  kind TEXT NOT NULL,                 -- kaggle | local | fake | runpod ...
  name TEXT NOT NULL,
  gpu TEXT NOT NULL,
  vram_gb REAL NOT NULL,
  session_cap_h REAL NOT NULL,
  weekly_quota_h REAL,                -- NULL = unlimited (local)
  reset_weekday INTEGER DEFAULT 5,    -- 0=Mon ... 5=Sat (UTC) for weekly quota reset
  reset_hour_utc INTEGER DEFAULT 0,
  manual_offset_h REAL DEFAULT 0,     -- user correction from `gpuo target sync`
  cost_per_hour REAL DEFAULT 0,
  enabled INTEGER DEFAULT 1,
  health REAL DEFAULT 1.0,
  settings TEXT DEFAULT '{}',         -- JSON, provider-specific (no secrets here)
  created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_kaggle_per_owner
  ON target(owner_id) WHERE kind = 'kaggle';
CREATE TABLE IF NOT EXISTS job (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL REFERENCES owner(id),
  name TEXT NOT NULL,
  entrypoint TEXT NOT NULL,
  bundle_key TEXT NOT NULL,
  config TEXT NOT NULL DEFAULT '{}',
  min_vram_gb REAL DEFAULT 0,
  est_gpu_hours REAL,
  priority INTEGER DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued', -- queued | running | waiting | completed | failed | cancelled
  last_target_id TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS segment (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL REFERENCES job(id),
  target_id TEXT NOT NULL REFERENCES target(id),
  lease_token TEXT NOT NULL,
  started_at REAL NOT NULL,
  deadline_at REAL NOT NULL,
  ended_at REAL,
  status TEXT NOT NULL DEFAULT 'running', -- running | completed | deadline | lost | failed | cancelled
  exit_reason TEXT,
  provider_handle TEXT,
  start_step INTEGER,
  end_step INTEGER
);
CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_id TEXT NOT NULL REFERENCES target(id),
  segment_id TEXT NOT NULL REFERENCES segment(id),
  started_at REAL NOT NULL,
  ended_at REAL NOT NULL,
  gpu_seconds REAL NOT NULL
);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def one(self, sql: str, *args: Any) -> dict[str, Any] | None:
        r = self.conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    def all(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def exec(self, sql: str, *args: Any) -> None:
        self.conn.execute(sql, args)

    # ---- owners --------------------------------------------------------------
    def ensure_owner(self, name: str) -> str:
        row = self.one("SELECT id FROM owner WHERE name = ?", name)
        if row:
            return row["id"]
        oid = new_id("own")
        self.exec("INSERT INTO owner VALUES (?, ?, ?)", oid, name, time.time())
        return oid

    # ---- targets -------------------------------------------------------------
    def add_target(
        self,
        owner_id: str,
        kind: str,
        name: str,
        gpu: str,
        vram_gb: float,
        session_cap_h: float,
        weekly_quota_h: float | None = None,
        cost_per_hour: float = 0.0,
        settings: dict | None = None,
        reset_weekday: int = 5,
        reset_hour_utc: int = 0,
    ) -> str:
        if kind == "kaggle" and self.one(
            "SELECT id FROM target WHERE owner_id = ? AND kind = 'kaggle'", owner_id
        ):
            raise ValueError(
                "This owner already has a Kaggle target. Kaggle allows one account per person; "
                "gpuo will not register a second one."
            )
        tid = new_id("tgt")
        self.exec(
            """INSERT INTO target (id, owner_id, kind, name, gpu, vram_gb, session_cap_h,
               weekly_quota_h, reset_weekday, reset_hour_utc, cost_per_hour, settings, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tid,
            owner_id,
            kind,
            name,
            gpu,
            vram_gb,
            session_cap_h,
            weekly_quota_h,
            reset_weekday,
            reset_hour_utc,
            cost_per_hour,
            json.dumps(settings or {}),
            time.time(),
        )
        return tid

    def target(self, tid: str) -> dict[str, Any]:
        t = self.one("SELECT * FROM target WHERE id = ?", tid)
        if not t:
            raise KeyError(tid)
        t["settings"] = json.loads(t["settings"] or "{}")
        return t

    def targets(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        rows = (
            self.all("SELECT * FROM target WHERE owner_id = ? ORDER BY created_at", owner_id)
            if owner_id
            else self.all("SELECT * FROM target ORDER BY created_at")
        )
        for t in rows:
            t["settings"] = json.loads(t["settings"] or "{}")
        return rows

    # ---- jobs ----------------------------------------------------------------
    def add_job(
        self,
        owner_id: str,
        name: str,
        entrypoint: str,
        bundle_key: str,
        config: dict | None = None,
        min_vram_gb: float = 0,
        est_gpu_hours: float | None = None,
        priority: int = 0,
        job_id: str | None = None,
    ) -> str:
        jid = job_id or new_id("job")
        now = time.time()
        self.exec(
            """INSERT INTO job (id, owner_id, name, entrypoint, bundle_key, config, min_vram_gb,
               est_gpu_hours, priority, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            jid,
            owner_id,
            name,
            entrypoint,
            bundle_key,
            json.dumps(config or {}),
            min_vram_gb,
            est_gpu_hours,
            priority,
            now,
            now,
        )
        return jid

    def job(self, jid: str) -> dict[str, Any]:
        j = self.one("SELECT * FROM job WHERE id = ?", jid)
        if not j:
            raise KeyError(jid)
        j["config"] = json.loads(j["config"] or "{}")
        return j

    def jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = (
            self.all("SELECT * FROM job WHERE status = ? ORDER BY priority DESC, created_at", status)
            if status
            else self.all("SELECT * FROM job ORDER BY created_at DESC")
        )
        for j in rows:
            j["config"] = json.loads(j["config"] or "{}")
        return rows

    def set_job(self, jid: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.exec(f"UPDATE job SET {cols} WHERE id = ?", *fields.values(), jid)

    # ---- segments ------------------------------------------------------------
    def segments(self, job_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM segment WHERE 1=1", []
        if job_id:
            sql += " AND job_id = ?"
            args.append(job_id)
        if status:
            sql += " AND status = ?"
            args.append(status)
        return self.all(sql + " ORDER BY started_at", *args)

    def set_segment(self, sid: str, **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.exec(f"UPDATE segment SET {cols} WHERE id = ?", *fields.values(), sid)
