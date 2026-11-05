"""gpuo command line."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .app import init_home, job_view, load, logs, submit, targets_view


def _print_table(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.upper().ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "") if r.get(c) is not None else "-").ljust(widths[c]) for c in cols))


def cmd_init(a: argparse.Namespace) -> None:
    home = init_home(Path(a.home).expanduser() if a.home else None, a.storage, a.owner)
    app = load(home)
    print(f"gpuo home: {home}")
    print(f"storage:   {app.config['storage']}")
    print(f"owner:     {app.config['owner']}")
    print("Next: gpuo target add local   (or: gpuo target add kaggle --username YOU)")


def cmd_target_add(a: argparse.Namespace) -> None:
    app = load()
    settings: dict = {}
    if a.kind == "kaggle":
        if not a.username:
            sys.exit("--username is required for Kaggle targets")
        if not shutil.which("kaggle"):
            sys.exit("Kaggle CLI not found. Run: pip install 'gpuo[kaggle]' and set up your token.")
        check = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "1"], capture_output=True, text=True
        )
        if check.returncode != 0:
            sys.exit(f"Kaggle CLI is not authenticated:\n{check.stderr or check.stdout}")
        settings = {"username": a.username, "pip_spec": a.pip_spec}
        gpu = a.gpu or "T4x2"
        vram = a.vram or 15
        cap = a.session_cap or 12
        quota = a.weekly_quota if a.weekly_quota is not None else 30
    elif a.kind in ("local", "fake"):
        gpu = a.gpu or "cpu"
        vram = a.vram or 0
        cap = a.session_cap or (0.05 if a.kind == "fake" else 24)
        quota = a.weekly_quota
        if a.kind == "fake":
            settings["kill_probability"] = a.kill_probability
        if a.cuda is not None:
            settings["cuda_visible_devices"] = a.cuda
    else:
        sys.exit(f"unknown target kind {a.kind}")
    try:
        tid = app.db.add_target(
            app.owner_id,
            a.kind,
            a.name or a.kind,
            gpu,
            vram,
            cap,
            quota,
            settings=settings,
            reset_weekday=a.reset_weekday,
            reset_hour_utc=a.reset_hour,
        )
    except ValueError as e:
        sys.exit(str(e))
    except sqlite3.IntegrityError:  # unique index: one Kaggle target per owner
        sys.exit("This owner already has a Kaggle target (Kaggle allows one account per person).")
    print(f"added target {tid}")


def cmd_target_list(a: argparse.Namespace) -> None:
    app = load()
    _print_table(
        targets_view(app),
        [
            "id",
            "name",
            "kind",
            "gpu",
            "vram_gb",
            "session_cap_h",
            "remaining_h",
            "usable_h",
            "busy",
            "health",
        ],
    )


def cmd_target_sync(a: argparse.Namespace) -> None:
    app = load()
    t = next((t for t in app.db.targets() if a.target in (t["id"], t["name"])), None)
    if not t:
        sys.exit(f"no target {a.target}")
    app.scheduler.sync_remaining(t["id"], a.remaining)
    print(f"{t['name']}: remaining set to {a.remaining} h (estimate will follow your usage from now)")


def cmd_target_disable(a: argparse.Namespace) -> None:
    app = load()
    for t in app.db.targets():
        if a.target in (t["id"], t["name"]):
            app.db.exec("UPDATE target SET enabled = ? WHERE id = ?", 0 if a.off else 1, t["id"])
            print(f"{t['name']}: {'disabled' if a.off else 'enabled'}")
            return
    sys.exit(f"no target {a.target}")


def cmd_submit(a: argparse.Namespace) -> None:
    app = load()
    cfg = json.loads(a.config) if a.config else {}
    if a.kaggle_dataset:
        cfg["kaggle_datasets"] = a.kaggle_dataset
    if a.target:
        cfg["targets"] = a.target
    jid = submit(app, a.entrypoint, Path(a.dir), a.name, a.min_vram, a.gpu_hours, a.priority, cfg)
    print(f"submitted {jid}")
    if not a.no_schedule:
        r = app.scheduler.tick()
        print(f"scheduler: placed {r['placed']} segment(s). Run `gpuo run` or `gpuo serve` to keep it going.")


def cmd_status(a: argparse.Namespace) -> None:
    app = load()
    if a.job:
        v = job_view(app, a.job)
        j = v["job"]
        print(f"{j['id']}  {j['name']}  status={j['status']}  error={j['error'] or '-'}")
        _print_table(v["segments"], ["id", "target_name", "status", "start_step", "end_step", "exit_reason"])
        if v["latest_checkpoint"]:
            c = v["latest_checkpoint"]
            print(f"latest checkpoint: {c['id']} epoch={c['epoch']} step={c['global_step']}")
        return
    _print_table(app.db.jobs(), ["id", "name", "status", "last_target_id", "error"])


def cmd_logs(a: argparse.Namespace) -> None:
    print(logs(load(), a.job, a.segment))


def cmd_cancel(a: argparse.Namespace) -> None:
    load().scheduler.cancel(a.job)
    print(f"cancelled {a.job}")


def cmd_run(a: argparse.Namespace) -> None:
    app = load()
    print(f"gpuo scheduler running every {a.interval}s (Ctrl+C to stop)")
    try:
        while True:
            r = app.scheduler.tick()
            if r["ended"] or r["placed"]:
                print(time.strftime("%H:%M:%S"), r, flush=True)
            if a.until_idle and not app.db.segments(status="running") and not (app.db.jobs(status="queued")):
                return
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


def cmd_serve(a: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit("The dashboard needs: pip install 'gpuo[server]'")
    from .server import create_app

    uvicorn.run(create_app(interval=a.interval), host=a.host, port=a.port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuo", description="Run training across your own GPU targets.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create ~/.gpuo with config, DB and storage")
    s.add_argument("--home")
    s.add_argument("--storage", help="file:///path or s3://bucket/prefix (R2: set GPUO_S3_ENDPOINT)")
    s.add_argument("--owner", help="your name; jobs only run on targets you own")
    s.set_defaults(fn=cmd_init)

    t = sub.add_parser("target", help="manage GPU targets").add_subparsers(dest="tcmd", required=True)
    ta = t.add_parser("add")
    ta.add_argument("kind", choices=["local", "kaggle", "fake"])
    ta.add_argument("--name")
    ta.add_argument("--username", help="Kaggle username")
    ta.add_argument("--gpu", help="e.g. T4x2, P100, RTX3060, cpu")
    ta.add_argument("--vram", type=float)
    ta.add_argument("--session-cap", type=float, help="hours per session")
    ta.add_argument("--weekly-quota", type=float, help="GPU hours per week (omit = unlimited)")
    ta.add_argument("--reset-weekday", type=int, default=5, help="0=Mon..6=Sun (UTC)")
    ta.add_argument("--reset-hour", type=int, default=0)
    ta.add_argument("--cuda", help="CUDA_VISIBLE_DEVICES for local targets")
    ta.add_argument("--kill-probability", type=float, default=0.0, help="fake target only")
    ta.add_argument(
        "--pip-spec",
        default="gpuo",
        help="what Kaggle kernels pip-install, e.g. git+https://github.com/YOU/gpuo",
    )
    ta.set_defaults(fn=cmd_target_add)
    tl = t.add_parser("list")
    tl.set_defaults(fn=cmd_target_list)
    ts = t.add_parser("sync", help="correct remaining hours from what the provider shows")
    ts.add_argument("target")
    ts.add_argument("--remaining", type=float, required=True)
    ts.set_defaults(fn=cmd_target_sync)
    td = t.add_parser("disable")
    td.add_argument("target")
    td.add_argument("--on", dest="off", action="store_false", default=True)
    td.set_defaults(fn=cmd_target_disable)

    s = sub.add_parser("submit", help="submit a training job")
    s.add_argument("entrypoint", help="script (and args) relative to --dir, e.g. 'train.py --lr 3e-4'")
    s.add_argument("--dir", default=".")
    s.add_argument("--name")
    s.add_argument("--min-vram", type=float, default=0)
    s.add_argument("--gpu-hours", type=float)
    s.add_argument("--priority", type=int, default=0)
    s.add_argument("--kaggle-dataset", action="append", help="owner/slug, mounted under /kaggle/input")
    s.add_argument("--target", action="append", help="restrict to these target names/ids")
    s.add_argument("--config", help="extra JSON config")
    s.add_argument("--no-schedule", action="store_true")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("status")
    s.add_argument("job", nargs="?")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("logs")
    s.add_argument("job")
    s.add_argument("--segment")
    s.set_defaults(fn=cmd_logs)
    s = sub.add_parser("cancel")
    s.add_argument("job")
    s.set_defaults(fn=cmd_cancel)
    s = sub.add_parser("run", help="run the scheduler loop in the terminal")
    s.add_argument("--interval", type=float, default=30)
    s.add_argument("--until-idle", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("serve", help="dashboard + scheduler at http://localhost:8787")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--interval", type=float, default=30)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
