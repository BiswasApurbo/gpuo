import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "gpuo-home"
    monkeypatch.setenv("GPUO_HOME", str(h))
    monkeypatch.setenv("PYTHONPATH", str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return h


def reference_result(tmp_path: Path, env: dict) -> dict:
    """Run the toy script once, uninterrupted, outside the platform."""
    import json

    out = tmp_path / "ref.json"
    e = {
        **os.environ,
        **env,
        "TOY_RESULT": str(out),
        "PYTHONPATH": str(ROOT),
        "GPUO_STORAGE": f"file://{tmp_path / 'ref-store'}",
    }
    for k in ("GPUO_JOB_ID", "GPUO_LEASE", "GPUO_DEADLINE", "GPUO_SEGMENT_ID"):
        e.pop(k, None)
    subprocess.run([sys.executable, "train.py"], cwd=TOY, env=e, check=True, capture_output=True)
    return json.loads(out.read_text())


def drive(app, job_id: str, timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.scheduler.tick()
        job = app.db.job(job_id)
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} still {app.db.job(job_id)['status']} after {timeout}s")
