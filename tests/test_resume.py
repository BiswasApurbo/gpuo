"""End-to-end: a job split across many short, sometimes-killed sessions must finish with
exactly the same result as an uninterrupted run."""

from conftest import TOY, drive, reference_result

from gpuo import layout
from gpuo.app import load, submit
from gpuo.scheduler import Policy
from gpuo.storage import get_json

TOY_ENV = {"TOY_EPOCHS": "4", "TOY_STEPS": "150", "TOY_SLEEP": "0.004"}  # ~2.4 s of work

FAST = Policy(
    safety_h=0, min_segment_h=1e-5, heartbeat_timeout_s=10, startup_grace_s=10, checkpoint_every_s=0.3
)


def _setup(home, monkeypatch, kill_probability=0.0):
    for k, v in TOY_ENV.items():
        monkeypatch.setenv(k, v)
    app = load(home, FAST)
    # 1-second sessions: forces several deadline hops
    app.db.add_target(
        app.owner_id,
        "fake",
        "fake-1",
        "cpu",
        0,
        session_cap_h=1.0 / 3600,
        settings={"kill_probability": kill_probability},
    )
    return app


def _final(app, job_id):
    segs = app.db.segments(job_id=job_id)
    last = [s for s in segs if s["status"] == "completed"][-1]
    st = get_json(app.store, layout.status(job_id, last["id"]))
    return {"w": st["w"], "b": st["b"]}, segs


def test_deadline_hops_match_uninterrupted(home, monkeypatch, tmp_path):
    ref = reference_result(tmp_path, TOY_ENV)
    app = _setup(home, monkeypatch)
    jid = submit(app, "train.py", TOY, config={"safety_s": 0.4})
    job = drive(app, jid)
    assert job["status"] == "completed", job
    got, segs = _final(app, jid)
    assert got == ref
    assert len(segs) >= 2, "expected the job to span several sessions"
    assert {s["status"] for s in segs} <= {"deadline", "completed"}


def test_random_kills_still_match(home, monkeypatch, tmp_path):
    ref = reference_result(tmp_path, TOY_ENV)
    app = _setup(home, monkeypatch, kill_probability=0.08)
    app.scheduler.policy.max_no_progress_segments = 50
    jid = submit(app, "train.py", TOY, config={"safety_s": 0.4})
    job = drive(app, jid, timeout=240)
    assert job["status"] == "completed", job
    got, segs = _final(app, jid)
    assert got == ref
