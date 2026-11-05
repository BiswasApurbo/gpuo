"""Key layout inside the storage mailbox. One place, so the agent and control plane agree."""

from .storage import keys


def job_json(job: str) -> str:
    return keys("jobs", job, "job.json")


def bundle(job: str) -> str:
    return keys("jobs", job, "bundle.tar.gz")


def lease(job: str) -> str:
    return keys("jobs", job, "lease.json")


def manifest(job: str) -> str:
    return keys("jobs", job, "manifest.json")


def ckpt(job: str, ckpt_id: str) -> str:
    return keys("jobs", job, "ckpt", f"{ckpt_id}.bin")


def heartbeat(job: str, seg: str) -> str:
    return keys("jobs", job, "segments", seg, "heartbeat.json")


def status(job: str, seg: str) -> str:
    return keys("jobs", job, "segments", seg, "status.json")


def metrics_dir(job: str, seg: str) -> str:
    return keys("jobs", job, "segments", seg, "metrics")


def metrics_chunk(job: str, seg: str, n: int) -> str:
    return keys("jobs", job, "segments", seg, "metrics", f"{n:06d}.jsonl")


def log_chunk(job: str, seg: str, n: int) -> str:
    return keys("jobs", job, "segments", seg, "logs", f"{n:06d}.log")
