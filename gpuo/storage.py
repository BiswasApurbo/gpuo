"""Object storage backends.

The storage bucket is the "mailbox" between the control plane (on the user's machine)
and the agent (inside a Kaggle kernel or other target). Neither side needs an inbound
connection: both read and write keys here.

URIs:
    file:///absolute/path        local directory (tests, local GPU)
    s3://bucket/prefix           AWS S3, or Cloudflare R2 with GPUO_S3_ENDPOINT set
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class Storage(Protocol):
    uri: str

    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete(self, key: str) -> None: ...


class StorageError(RuntimeError):
    pass


class LocalStorage:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = f"file://{self.root}"

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise StorageError(f"key escapes storage root: {key}")
        return p

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)  # atomic on POSIX

    def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.exists():
            raise KeyError(key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            return [prefix]
        if not base.exists():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                out.append(str(p.relative_to(self.root)))
        return sorted(out)

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


class S3Storage:
    """S3 or Cloudflare R2. Needs `pip install gpuo[s3]`.

    Credentials come from the standard AWS environment variables
    (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY). For R2 set GPUO_S3_ENDPOINT to
    https://<account-id>.r2.cloudflarestorage.com
    """

    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None):
        try:
            import boto3  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise StorageError("S3 storage needs: pip install 'gpuo[s3]'") from e
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.uri = f"s3://{bucket}/{self.prefix}".rstrip("/")
        self._s3 = boto3.client("s3", endpoint_url=endpoint_url or os.environ.get("GPUO_S3_ENDPOINT"))

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)

    def get_bytes(self, key: str) -> bytes:
        try:
            r = self._s3.get_object(Bucket=self.bucket, Key=self._k(key))
        except self._s3.exceptions.NoSuchKey as e:
            raise KeyError(key) from e
        return r["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> list[str]:
        out: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._k(prefix)):
            for obj in page.get("Contents", []):
                out.append(obj["Key"][strip:])
        return sorted(out)

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._k(key))


def open_storage(uri: str) -> Storage:
    u = urlparse(uri)
    if u.scheme in ("", "file"):
        return LocalStorage(u.path if u.scheme else uri)
    if u.scheme == "s3":
        return S3Storage(u.netloc, u.path)
    raise StorageError(f"unsupported storage URI: {uri}")


# --- small JSON helpers -------------------------------------------------------


def put_json(store: Storage, key: str, obj: object) -> None:
    store.put_bytes(key, json.dumps(obj, indent=2, sort_keys=True).encode())


def get_json(store: Storage, key: str, default: object = None) -> object:
    try:
        return json.loads(store.get_bytes(key))
    except KeyError:
        return default


def keys(*parts: str | int) -> str:
    return "/".join(str(p).strip("/") for p in parts)


def iter_lines(store: Storage, prefix: str) -> Iterable[dict]:
    for k in store.list(prefix):
        for line in store.get_bytes(k).decode().splitlines():
            if line.strip():
                yield json.loads(line)
