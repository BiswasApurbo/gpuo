"""The SDK imported by user training code.

Resume is only correct if *everything* that shapes training is saved: model, optimizer,
LR scheduler, AMP scaler, RNG states and the position in the data. `checkpoint()` saves
all of those; `restore()` puts them back.

Commit protocol (see docs/design.md):
  1. serialize -> bytes, sha256
  2. upload to jobs/<job>/ckpt/<id>.bin
  3. re-read lease.json; if our lease is no longer current, abort (a newer segment owns the job)
  4. write manifest.json pointing at the new checkpoint
A checkpoint that is not in the manifest does not exist.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import layout
from .storage import Storage, get_json, open_storage, put_json

KEEP_LAST = 3


class LeaseLost(RuntimeError):
    """Raised when another segment has taken over this job. Stop immediately."""


@dataclass
class RunState:
    epoch: int = 0
    step_in_epoch: int = 0
    global_step: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    resumed: bool = False


# --- serialization -------------------------------------------------------------


def _has_state_dict(obj: Any) -> bool:
    return hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict")


def _rng_state() -> dict[str, Any]:
    st: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        st["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        st["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return st


def _set_rng_state(st: dict[str, Any]) -> None:
    if "python" in st:
        random.setstate(st["python"])
    if "numpy" in st:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(st["numpy"])
        except ImportError:
            pass
    if "torch" in st:
        try:
            import torch  # type: ignore

            torch.set_rng_state(st["torch"])
            if "cuda" in st and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["cuda"])
        except ImportError:
            pass


def _dumps(payload: dict[str, Any]) -> bytes:
    try:
        import torch  # type: ignore

        buf = io.BytesIO()
        torch.save(payload, buf)
        return b"TORCH" + buf.getvalue()
    except ImportError:
        return b"PICKL" + pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _loads(data: bytes) -> dict[str, Any]:
    tag, body = data[:5], data[5:]
    if tag == b"TORCH":
        import torch  # type: ignore

        return torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
    if tag == b"PICKL":
        return pickle.loads(body)
    raise ValueError("unknown checkpoint format")


# --- run -----------------------------------------------------------------------


class Run:
    def __init__(
        self,
        store: Storage,
        job_id: str,
        segment_id: str,
        lease_token: str,
        deadline: float | None,
        checkpoint_every_s: float = 600.0,
        safety_s: float = 900.0,
        config: dict[str, Any] | None = None,
    ):
        self.store = store
        self.job_id = job_id
        self.segment_id = segment_id
        self.lease_token = lease_token
        self.deadline = deadline
        self.checkpoint_every_s = checkpoint_every_s
        self.safety_s = safety_s
        self.config = config or {}
        self.state = RunState()
        self._last_ckpt = time.time()
        self._metrics: list[dict[str, Any]] = []
        self._chunk = 0
        self._last_flush = time.time()
        self._finished = False

    # ---- lease ---------------------------------------------------------------
    def _check_lease(self) -> None:
        cur = get_json(self.store, layout.lease(self.job_id), {}) or {}
        if cur.get("token") != self.lease_token:
            raise LeaseLost(f"lease for job {self.job_id} now belongs to segment {cur.get('segment')}")

    # ---- restore -------------------------------------------------------------
    def restore(self, **objects: Any) -> RunState:
        """Load the latest committed checkpoint into the given objects (model, optimizer, ...)."""
        man = get_json(self.store, layout.manifest(self.job_id), {}) or {}
        latest = man.get("latest")
        if not latest:
            self.state = RunState()
            return self.state
        data = self.store.get_bytes(latest["key"])
        if hashlib.sha256(data).hexdigest() != latest["sha256"]:
            raise RuntimeError(f"checkpoint {latest['id']} failed checksum")
        payload = _loads(data)
        for name, obj in objects.items():
            if obj is None:
                continue
            if name not in payload["objects"]:
                raise KeyError(f"checkpoint has no saved object named {name!r}")
            saved = payload["objects"][name]
            if _has_state_dict(obj):
                obj.load_state_dict(saved)
            elif isinstance(obj, dict):
                obj.clear()
                obj.update(saved)
            else:
                raise TypeError(f"{name}: pass objects with load_state_dict() or dicts")
        _set_rng_state(payload.get("rng", {}))
        s = payload["state"]
        self.state = RunState(
            epoch=s["epoch"],
            step_in_epoch=s["step_in_epoch"],
            global_step=s["global_step"],
            extra=s.get("extra", {}),
            resumed=True,
        )
        return self.state

    # ---- logging -------------------------------------------------------------
    def log(self, **metrics: Any) -> None:
        self.state.global_step += 1
        rec = {"t": time.time(), "step": self.state.global_step, **metrics}
        self._metrics.append(rec)
        if time.time() - self._last_flush > 30 or len(self._metrics) >= 500:
            self.flush()

    def flush(self) -> None:
        if self._metrics:
            body = "\n".join(json.dumps(m, default=float) for m in self._metrics) + "\n"
            self.store.put_bytes(
                layout.metrics_chunk(self.job_id, self.segment_id, self._chunk), body.encode()
            )
            self._chunk += 1
            self._metrics = []
        self.heartbeat()
        self._last_flush = time.time()

    def heartbeat(self, status: str = "running") -> None:
        put_json(
            self.store,
            layout.heartbeat(self.job_id, self.segment_id),
            {
                "t": time.time(),
                "status": status,
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "deadline": self.deadline,
            },
        )

    # ---- timing --------------------------------------------------------------
    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def should_stop(self) -> bool:
        """True when the segment must checkpoint and exit before the provider kills it."""
        return self.time_left() <= self.safety_s

    def should_checkpoint(self) -> bool:
        return time.time() - self._last_ckpt >= self.checkpoint_every_s or self.should_stop()

    # ---- checkpoint ----------------------------------------------------------
    def checkpoint(
        self,
        epoch: int,
        step_in_epoch: int,
        extra: dict[str, Any] | None = None,
        **objects: Any,
    ) -> str:
        """Save and commit a checkpoint. `step_in_epoch` = the next step to run on resume."""
        saved = {}
        for name, obj in objects.items():
            if obj is None:
                continue
            if _has_state_dict(obj):
                saved[name] = obj.state_dict()
            elif isinstance(obj, dict):
                saved[name] = dict(obj)
            else:
                raise TypeError(f"{name}: pass objects with state_dict() or dicts")
        self.state.epoch = epoch
        self.state.step_in_epoch = step_in_epoch
        if extra is not None:
            self.state.extra = extra
        payload = {
            "objects": saved,
            "rng": _rng_state(),
            "state": {
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": self.state.global_step,
                "extra": self.state.extra,
            },
            "code_ref": self.config.get("code_ref"),
        }
        data = _dumps(payload)
        digest = hashlib.sha256(data).hexdigest()
        ckpt_id = f"{self.state.global_step:010d}-{uuid.uuid4().hex[:8]}"
        key = layout.ckpt(self.job_id, ckpt_id)

        self._check_lease()
        self.store.put_bytes(key, data)
        if hashlib.sha256(self.store.get_bytes(key)).hexdigest() != digest:
            raise RuntimeError("checkpoint upload verification failed")
        self._check_lease()  # re-check right before commit

        man = get_json(self.store, layout.manifest(self.job_id), {}) or {}
        entry = {
            "id": ckpt_id,
            "key": key,
            "sha256": digest,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": self.state.global_step,
            "segment": self.segment_id,
            "lease": self.lease_token,
            "t": time.time(),
            "bytes": len(data),
        }
        history = [entry] + list(man.get("history", []))
        for old in history[KEEP_LAST:]:
            self.store.delete(old["key"])
        put_json(self.store, layout.manifest(self.job_id), {"latest": entry, "history": history[:KEEP_LAST]})
        self._last_ckpt = time.time()
        self.flush()
        return ckpt_id

    # ---- exit ----------------------------------------------------------------
    def _write_status(self, status: str, **info: Any) -> None:
        put_json(
            self.store,
            layout.status(self.job_id, self.segment_id),
            {"status": status, "t": time.time(), "global_step": self.state.global_step, **info},
        )

    def exit_for_resume(self) -> None:
        """Call after the final checkpoint when should_stop() is True."""
        self.flush()
        self.heartbeat("deadline")
        self._write_status("deadline")
        self._finished = True
        sys.exit(0)

    def finish(self, **info: Any) -> None:
        """Call once when training is complete."""
        self.flush()
        self.heartbeat("completed")
        self._write_status("completed", **info)
        self._finished = True


def init(**overrides: Any) -> Run:
    """Create a Run from the environment the agent sets up.

    Run outside gpuo (plain `python train.py`) it still works: state goes to ./.gpuo-standalone
    with no deadline, so the same script runs locally without the platform.
    """
    env = os.environ
    store_uri = overrides.pop("storage", None) or env.get("GPUO_STORAGE") or "file://./.gpuo-standalone"
    store = open_storage(store_uri)
    job_id = overrides.pop("job_id", None) or env.get("GPUO_JOB_ID") or "standalone"
    segment_id = overrides.pop("segment_id", None) or env.get("GPUO_SEGMENT_ID") or "seg-0"
    lease_token = overrides.pop("lease_token", None) or env.get("GPUO_LEASE")
    if lease_token is None:
        # standalone: create our own lease
        lease_token = uuid.uuid4().hex
        put_json(store, layout.lease(job_id), {"token": lease_token, "segment": segment_id})
    deadline_s = overrides.pop("deadline", None) or env.get("GPUO_DEADLINE")
    deadline = float(deadline_s) if deadline_s else None
    config = get_json(store, layout.job_json(job_id), {}) or {}
    run = Run(
        store,
        job_id,
        segment_id,
        lease_token,
        deadline,
        checkpoint_every_s=float(env.get("GPUO_CKPT_EVERY_S", overrides.pop("checkpoint_every_s", 600))),
        safety_s=float(env.get("GPUO_SAFETY_S", overrides.pop("safety_s", 900))),
        config=config,
    )
    run.heartbeat()
    return run
