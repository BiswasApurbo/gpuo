"""Scheduler: places each job segment on the owner's target with the most usable GPU time.

usable_hours(t) = min(remaining_quota(t), session_cap(t)) - safety margin
score(t)        = usable_hours(t) * SPEED[gpu] (+10% if the job last ran there) - cost weight

Only targets owned by the job's owner are ever considered (the one-owner rule).
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import layout
from .db import DB, new_id
from .storage import Storage, get_json, put_json
from .targets import ProviderStatus, get_adapter

# Relative training throughput. Rough starting values; refine by measuring steps/s per job.
SPEED: dict[str, float] = {
    "cpu": 0.05,
    "P100": 1.0,
    "T4": 0.8,
    "T4x2": 1.3,
    "RTX3060": 1.1,
    "RTX4090": 3.5,
    "A10G": 1.8,
    "L4": 1.7,
    "A100": 4.0,
    "H100": 7.0,
}


@dataclass
class Policy:
    safety_h: float = 0.25  # stop this long before the session cap / quota runs out
    min_segment_h: float = 1.0  # don't start a segment shorter than this
    heartbeat_timeout_s: float = 300  # no heartbeat for this long (and provider not running) = lost
    startup_grace_s: float = 900  # time allowed for pip install / data download
    same_target_bonus: float = 0.10
    cost_weight: float = 1.0
    max_no_progress_segments: int = 3
    checkpoint_every_s: float = 600


def week_start(t: dict[str, Any], now: float) -> float:
    """Timestamp of the most recent weekly quota reset for this target."""
    n = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc)
    reset_today = n.replace(hour=int(t["reset_hour_utc"]), minute=0, second=0, microsecond=0)
    days_back = (n.weekday() - int(t["reset_weekday"])) % 7
    start = reset_today - dt.timedelta(days=days_back)
    if start > n:
        start -= dt.timedelta(days=7)
    return start.timestamp()


class Scheduler:
    def __init__(self, db: DB, store: Storage, home: str | Path, policy: Policy | None = None):
        self.db = db
        self.store = store
        self.home = str(home)
        self.policy = policy or Policy()

    # ---- capacity ------------------------------------------------------------
    def used_hours(self, t: dict[str, Any], now: float | None = None) -> float:
        now = now or time.time()
        since = week_start(t, now)
        row = self.db.one(
            "SELECT COALESCE(SUM(MIN(ended_at, ?) - MAX(started_at, ?)), 0) AS s FROM ledger "
            "WHERE target_id = ? AND ended_at > ?",
            now,
            since,
            t["id"],
            since,
        )
        used = float(row["s"]) if row else 0.0
        for seg in self.db.segments(status="running"):
            if seg["target_id"] == t["id"]:
                used += max(0.0, now - max(seg["started_at"], since))
        return used / 3600.0

    def remaining_hours(self, t: dict[str, Any], now: float | None = None) -> float:
        if t["weekly_quota_h"] is None:
            return float("inf")
        return max(0.0, t["weekly_quota_h"] - self.used_hours(t, now) - (t["manual_offset_h"] or 0))

    def usable_hours(self, t: dict[str, Any], now: float | None = None) -> float:
        return min(self.remaining_hours(t, now), t["session_cap_h"]) - self.policy.safety_h

    def busy(self, t: dict[str, Any]) -> bool:
        return any(s["target_id"] == t["id"] for s in self.db.segments(status="running"))

    def sync_remaining(self, target_id: str, remaining_h: float) -> None:
        """User tells us what the provider shows; store the difference as an offset."""
        t = self.db.target(target_id)
        if t["weekly_quota_h"] is None:
            return
        est = t["weekly_quota_h"] - self.used_hours(t)
        self.db.exec("UPDATE target SET manual_offset_h = ? WHERE id = ?", est - remaining_h, target_id)

    # ---- placement -----------------------------------------------------------
    def score(self, job: dict[str, Any], t: dict[str, Any]) -> float:
        s = self.usable_hours(t) * SPEED.get(t["gpu"], 1.0)
        if job.get("last_target_id") == t["id"]:
            s *= 1 + self.policy.same_target_bonus
        return s - self.policy.cost_weight * (t["cost_per_hour"] or 0)

    def candidates(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for t in self.db.targets(owner_id=job["owner_id"]):  # one-owner rule
            if not t["enabled"] or t["health"] < 0.5 or self.busy(t):
                continue
            if t["vram_gb"] < (job["min_vram_gb"] or 0):
                continue
            if self.usable_hours(t) < self.policy.min_segment_h:
                continue
            allowed = job["config"].get("targets")
            if allowed and t["id"] not in allowed and t["name"] not in allowed:
                continue
            out.append(t)
        return sorted(out, key=lambda t: self.score(job, t), reverse=True)

    def place(self, job: dict[str, Any]) -> str | None:
        cands = self.candidates(job)
        if not cands:
            if job["status"] != "waiting":
                self.db.set_job(job["id"], status="waiting")
            return None
        t = cands[0]
        now = time.time()
        hours = self.usable_hours(t, now)
        seg_id = new_id("seg")
        lease = uuid.uuid4().hex
        deadline = now + hours * 3600
        man = get_json(self.store, layout.manifest(job["id"]), {}) or {}
        start_step = (man.get("latest") or {}).get("global_step", 0)

        # 1. take the lease in storage first: any older segment's commits are now refused
        put_json(self.store, layout.lease(job["id"]), {"token": lease, "segment": seg_id, "t": now})
        self.db.exec(
            """INSERT INTO segment (id, job_id, target_id, lease_token, started_at, deadline_at,
               status, start_step) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)""",
            seg_id,
            job["id"],
            t["id"],
            lease,
            now,
            deadline,
            start_step,
        )
        env = {
            "GPUO_STORAGE": self.store.uri,
            "GPUO_JOB_ID": job["id"],
            "GPUO_SEGMENT_ID": seg_id,
            "GPUO_LEASE": lease,
            "GPUO_DEADLINE": f"{deadline:.0f}",
            "GPUO_SAFETY_S": str(int(job["config"].get("safety_s", self.policy.safety_h * 3600 / 2))),
            "GPUO_CKPT_EVERY_S": str(
                int(job["config"].get("checkpoint_every_s", self.policy.checkpoint_every_s))
            ),
            "GPUO_HOME": self.home,
        }
        try:
            handle = get_adapter(t["kind"]).launch(t, job, env)
        except Exception as e:  # launch failed: release target, keep job queued
            self.db.set_segment(seg_id, status="failed", exit_reason=f"launch: {e}", ended_at=time.time())
            self.db.exec("UPDATE target SET health = MAX(0, health - 0.3) WHERE id = ?", t["id"])
            self.db.set_job(job["id"], status="queued", error=f"launch on {t['name']} failed: {e}")
            return None
        self.db.set_segment(seg_id, provider_handle=handle)
        self.db.set_job(job["id"], status="running", last_target_id=t["id"], error=None)
        return seg_id

    # ---- monitoring ----------------------------------------------------------
    def _end(self, seg: dict[str, Any], status: str, reason: str | None) -> None:
        now = time.time()
        man = get_json(self.store, layout.manifest(seg["job_id"]), {}) or {}
        end_step = (man.get("latest") or {}).get("global_step", seg["start_step"] or 0)
        self.db.set_segment(seg["id"], status=status, exit_reason=reason, ended_at=now, end_step=end_step)
        self.db.exec(
            "INSERT INTO ledger (target_id, segment_id, started_at, ended_at, gpu_seconds) "
            "VALUES (?, ?, ?, ?, ?)",
            seg["target_id"],
            seg["id"],
            seg["started_at"],
            now,
            now - seg["started_at"],
        )
        job = self.db.job(seg["job_id"])
        if status == "completed":
            self.db.set_job(job["id"], status="completed")
        elif status == "failed":
            self.db.set_job(job["id"], status="failed", error=reason)
        elif job["status"] != "cancelled":
            if self._stuck(job["id"]):
                self.db.set_job(
                    job["id"], status="failed", error="no progress across several segments; check the logs"
                )
            else:
                self.db.set_job(job["id"], status="queued")
        if status in ("completed", "deadline"):
            self.db.exec("UPDATE target SET health = MIN(1, health + 0.1) WHERE id = ?", seg["target_id"])

    def _stuck(self, job_id: str) -> bool:
        segs = [s for s in self.db.segments(job_id=job_id) if s["status"] != "running"]
        recent = segs[-self.policy.max_no_progress_segments :]
        return len(recent) == self.policy.max_no_progress_segments and all(
            (s["end_step"] or 0) <= (s["start_step"] or 0) for s in recent
        )

    def check(self, seg: dict[str, Any]) -> str | None:
        """Look at one running segment; end it if it is over. Returns the new status or None."""
        st = get_json(self.store, layout.status(seg["job_id"], seg["id"]))
        if st:
            status = st.get("status", "lost")
            reason = st.get("error") or st.get("tail") or st.get("note")
            if status not in ("completed", "deadline", "failed", "lost"):
                status = "lost"
            self._end(seg, status, reason)
            return status

        t = self.db.target(seg["target_id"])
        pst = (
            get_adapter(t["kind"]).status(t, seg["provider_handle"])
            if seg["provider_handle"]
            else ProviderStatus.UNKNOWN
        )
        now = time.time()
        hb = get_json(self.store, layout.heartbeat(seg["job_id"], seg["id"]))
        last_seen = hb["t"] if hb else seg["started_at"] + self.policy.startup_grace_s
        stale = now - last_seen > self.policy.heartbeat_timeout_s

        if pst in (ProviderStatus.EXITED, ProviderStatus.ERROR):
            # give the agent a moment to write status.json after the process exits
            time.sleep(0.5)
            if get_json(self.store, layout.status(seg["job_id"], seg["id"])):
                return self.check(seg)
            self._end(seg, "lost", f"provider reported {pst.value} without a segment status")
            return "lost"
        if now > seg["deadline_at"] + self.policy.heartbeat_timeout_s:
            self._end(seg, "lost", "past deadline with no status")
            return "lost"
        if stale and pst != ProviderStatus.RUNNING and pst != ProviderStatus.QUEUED:
            self._end(seg, "lost", "heartbeats stopped")
            return "lost"
        return None

    def tick(self) -> dict[str, int]:
        ended = 0
        for seg in self.db.segments(status="running"):
            if self.check(seg):
                ended += 1
        placed = 0
        for job in self.db.jobs(status="queued") + self.db.jobs(status="waiting"):
            if any(s["job_id"] == job["id"] for s in self.db.segments(status="running")):
                continue
            if self.place(job):
                placed += 1
        return {"ended": ended, "placed": placed}

    def cancel(self, job_id: str) -> None:
        self.db.set_job(job_id, status="cancelled")
        # revoke the lease so the agent stops at its next checkpoint
        put_json(self.store, layout.lease(job_id), {"token": None, "segment": None, "t": time.time()})
        for seg in self.db.segments(job_id=job_id, status="running"):
            t = self.db.target(seg["target_id"])
            if seg["provider_handle"]:
                get_adapter(t["kind"]).cancel(t, seg["provider_handle"])
            self._end(seg, "cancelled", "cancelled by user")
