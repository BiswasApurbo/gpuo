"""The agent: runs inside a target session (Kaggle kernel, local subprocess, VM).

    python -m gpuo.agent --workdir DIR

Reads its instructions from GPUO_* environment variables, pulls the code bundle from
storage, runs the user's entrypoint, ships logs, and makes sure a status.json exists when
the segment ends. Outbound connections only.
"""

from __future__ import annotations

import argparse
import io
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

from . import layout
from .storage import get_json, open_storage, put_json

LOG_FLUSH_S = 20


def _safe_extract(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        root = dest.resolve()
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe path in bundle: {m.name}")
            if m.issym() or m.islnk():
                raise RuntimeError(f"links are not allowed in bundles: {m.name}")
        tar.extractall(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpuo.agent")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args(argv)

    env = os.environ
    store = open_storage(env["GPUO_STORAGE"])
    job_id, seg_id = env["GPUO_JOB_ID"], env["GPUO_SEGMENT_ID"]
    deadline = float(env["GPUO_DEADLINE"]) if env.get("GPUO_DEADLINE") else None
    job = get_json(store, layout.job_json(job_id), {}) or {}
    entry = job.get("entrypoint") or env.get("GPUO_ENTRYPOINT")
    if not entry:
        put_json(store, layout.status(job_id, seg_id), {"status": "failed", "error": "no entrypoint"})
        return 2

    work = Path(args.workdir)
    code = work / "code"
    _safe_extract(store.get_bytes(layout.bundle(job_id)), code)

    child_env = {**env, "PYTHONUNBUFFERED": "1"}
    parts = shlex.split(entry)
    cmd = [sys.executable, *parts] if parts[0].endswith(".py") else parts
    proc = subprocess.Popen(
        cmd, cwd=code, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    buf: list[str] = []
    tail: list[str] = []
    lock = threading.Lock()
    chunk = [0]

    def flush() -> None:
        with lock:
            if not buf:
                return
            body = "".join(buf)
            buf.clear()
            n = chunk[0]
            chunk[0] += 1
        store.put_bytes(layout.log_chunk(job_id, seg_id, n), body.encode())

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            with lock:
                buf.append(line)
                tail.append(line)
                del tail[:-50]

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    killed = False
    last = time.time()
    while proc.poll() is None:
        time.sleep(1)
        if time.time() - last >= LOG_FLUSH_S:
            flush()
            last = time.time()
        if deadline is not None and time.time() >= deadline:
            proc.kill()  # script ignored should_stop(); provider would kill us now anyway
            killed = True
            break
    proc.wait()
    t.join(timeout=5)
    flush()

    if not store.exists(layout.status(job_id, seg_id)):
        if killed:
            status = {"status": "lost", "error": "killed at deadline (script did not stop in time)"}
        elif proc.returncode == 0:
            status = {"status": "completed", "note": "script exited 0 without run.finish()"}
        else:
            status = {"status": "failed", "returncode": proc.returncode, "tail": "".join(tail)}
        put_json(store, layout.status(job_id, seg_id), {**status, "t": time.time()})
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
