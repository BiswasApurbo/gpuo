"""Local GPU (or CPU) target: runs the agent as a subprocess on this machine."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import ProviderStatus, register

_PROCS: dict[str, subprocess.Popen] = {}


class LocalTarget:
    kind = "local"

    def launch(self, target: dict[str, Any], job: dict[str, Any], env: dict[str, str]) -> str:
        workdir = Path(env["GPUO_HOME"]) / "work" / env["GPUO_SEGMENT_ID"]
        workdir.mkdir(parents=True, exist_ok=True)
        full_env = {**os.environ, **env}
        cuda = target["settings"].get("cuda_visible_devices")
        if cuda is not None:
            full_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
        log = open(workdir / "agent.log", "ab")
        p = subprocess.Popen(
            [sys.executable, "-m", "gpuo.agent", "--workdir", str(workdir)],
            env=full_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle = str(p.pid)
        _PROCS[handle] = p
        return handle

    def status(self, target: dict[str, Any], handle: str) -> ProviderStatus:
        p = _PROCS.get(handle)
        if p is not None:
            return ProviderStatus.RUNNING if p.poll() is None else ProviderStatus.EXITED
        # control plane restarted: fall back to the pid
        try:
            os.kill(int(handle), 0)
            return ProviderStatus.RUNNING
        except (ProcessLookupError, ValueError):
            return ProviderStatus.EXITED
        except PermissionError:
            return ProviderStatus.UNKNOWN

    def cancel(self, target: dict[str, Any], handle: str) -> None:
        try:
            os.killpg(int(handle), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass

    def kill_hard(self, handle: str) -> None:
        """Simulate a provider killing the session (used by tests and the fake target)."""
        try:
            os.killpg(int(handle), signal.SIGKILL)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        p = _PROCS.get(handle)
        if p is not None:
            p.wait()


register(LocalTarget())
