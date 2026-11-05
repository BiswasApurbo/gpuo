import datetime as dt
import sqlite3

import pytest

from gpuo import layout
from gpuo.app import load
from gpuo.db import DB
from gpuo.scheduler import Scheduler, week_start
from gpuo.sdk import LeaseLost, Run
from gpuo.storage import LocalStorage, StorageError, get_json, put_json


def test_old_lease_cannot_commit(tmp_path):
    store = LocalStorage(tmp_path)
    put_json(store, layout.lease("j"), {"token": "A", "segment": "s1"})
    old = Run(store, "j", "s1", "A", None)
    old.checkpoint(epoch=0, step_in_epoch=1, model={"w": 1})
    put_json(store, layout.lease("j"), {"token": "B", "segment": "s2"})  # new segment takes over
    with pytest.raises(LeaseLost):
        old.checkpoint(epoch=0, step_in_epoch=2, model={"w": 2})
    man = get_json(store, layout.manifest("j"))
    assert man["latest"]["lease"] == "A" and man["latest"]["step_in_epoch"] == 1


def test_restore_roundtrip_and_retention(tmp_path):
    store = LocalStorage(tmp_path)
    put_json(store, layout.lease("j"), {"token": "A", "segment": "s1"})
    r = Run(store, "j", "s1", "A", None)
    for i in range(6):
        r.log(loss=1.0)
        r.checkpoint(epoch=i, step_in_epoch=3, model={"w": i})
    man = get_json(store, layout.manifest("j"))
    assert len(man["history"]) == 3
    assert len(store.list("jobs/j/ckpt")) == 3
    m = {}
    st = Run(store, "j", "s2", "A", None).restore(model=m)
    assert m == {"w": 5} and st.epoch == 5 and st.global_step == 6 and st.resumed


def test_one_kaggle_target_per_owner(tmp_path):
    db = DB(tmp_path / "x.db")
    o = db.ensure_owner("me")
    db.add_target(o, "kaggle", "k1", "T4x2", 15, 12, 30, settings={"username": "me"})
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        db.add_target(o, "kaggle", "k2", "P100", 16, 12, 30, settings={"username": "other"})


def test_scheduler_picks_most_usable_time_and_respects_owner(home):
    app = load(home)
    me = app.owner_id
    other = app.db.ensure_owner("someone-else")
    small = app.db.add_target(me, "local", "small", "T4", 16, session_cap_h=12, weekly_quota_h=2)
    big = app.db.add_target(me, "local", "big", "T4", 16, session_cap_h=12, weekly_quota_h=30)
    app.db.add_target(other, "local", "not-mine", "H100", 80, session_cap_h=24)
    jid = app.db.add_job(me, "j", "train.py", "b", {}, min_vram_gb=8)
    names = [t["name"] for t in app.scheduler.candidates(app.db.job(jid))]
    assert names[0] == "big"
    assert "not-mine" not in names
    assert app.scheduler.usable_hours(app.db.target(small)) == pytest.approx(2 - 0.25)
    assert app.scheduler.usable_hours(app.db.target(big)) == pytest.approx(12 - 0.25)


def test_vram_filter(home):
    app = load(home)
    app.db.add_target(app.owner_id, "local", "tiny", "cpu", 4, 12)
    jid = app.db.add_job(app.owner_id, "j", "train.py", "b", {}, min_vram_gb=16)
    assert app.scheduler.candidates(app.db.job(jid)) == []


def test_week_start():
    t = {"reset_weekday": 5, "reset_hour_utc": 0}  # Saturday 00:00 UTC
    wed = dt.datetime(2026, 11, 4, 12, tzinfo=dt.timezone.utc).timestamp()
    assert dt.datetime.fromtimestamp(week_start(t, wed), dt.timezone.utc) == dt.datetime(
        2026, 10, 31, tzinfo=dt.timezone.utc
    )
    sat = dt.datetime(2026, 11, 7, 1, tzinfo=dt.timezone.utc).timestamp()
    assert dt.datetime.fromtimestamp(week_start(t, sat), dt.timezone.utc) == dt.datetime(
        2026, 11, 7, tzinfo=dt.timezone.utc
    )


def test_sync_remaining(home):
    app = load(home)
    tid = app.db.add_target(app.owner_id, "local", "q", "T4", 16, 12, weekly_quota_h=30)
    app.scheduler.sync_remaining(tid, 22.5)
    assert app.scheduler.remaining_hours(app.db.target(tid)) == pytest.approx(22.5)


def test_storage_rejects_path_escape(tmp_path):
    s = LocalStorage(tmp_path / "root")
    with pytest.raises(StorageError):
        s.put_bytes("../evil", b"x")


def test_scheduler_unused_import_guard():
    assert Scheduler  # keeps the public import path stable
