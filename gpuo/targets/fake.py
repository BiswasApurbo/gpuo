"""Fake target for testing resume: a local subprocess with a tiny session cap that the
provider may also kill at random (like a preempted or disconnected notebook)."""

from __future__ import annotations

import random
from typing import Any

from .base import ProviderStatus, register
from .local import LocalTarget


class FakeTarget(LocalTarget):
    kind = "fake"

    def status(self, target: dict[str, Any], handle: str) -> ProviderStatus:
        st = super().status(target, handle)
        p_kill = float(target["settings"].get("kill_probability", 0.0))
        if st == ProviderStatus.RUNNING and p_kill and random.random() < p_kill:
            self.kill_hard(handle)
            return ProviderStatus.ERROR
        return st


register(FakeTarget())
