"""Target adapter interface. Every provider (Kaggle, local GPU, paid clouds) implements this."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol


class ProviderStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    EXITED = "exited"  # process/kernel ended; read status.json to learn how
    ERROR = "error"  # provider reports failure (kernel error, VM died)
    UNKNOWN = "unknown"  # provider can't tell us (transient API error, lost handle)


class Target(Protocol):
    kind: str

    def launch(self, target: dict[str, Any], job: dict[str, Any], env: dict[str, str]) -> str:
        """Start the gpuo agent on this target with `env`. Return a provider handle."""

    def status(self, target: dict[str, Any], handle: str) -> ProviderStatus: ...

    def cancel(self, target: dict[str, Any], handle: str) -> None: ...


_REGISTRY: dict[str, Target] = {}


def register(adapter: Target) -> Target:
    _REGISTRY[adapter.kind] = adapter
    return adapter


def get_adapter(kind: str) -> Target:
    if not _REGISTRY:
        from . import fake, kaggle, local  # noqa: F401  (registers adapters)
    if kind not in _REGISTRY:
        raise KeyError(f"no adapter for target kind {kind!r}")
    return _REGISTRY[kind]
