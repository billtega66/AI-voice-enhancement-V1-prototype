from __future__ import annotations

from abc import ABC, abstractmethod

from ..dsp.engine import Pipeline


class ProcessingBackend(ABC):
    """Contract shared by all backends (mirrors the ProcessingBackend interface in docs/ARCHITECTURE.md)."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str]:
        """(usable on this machine?, human-readable reason)."""

    @abstractmethod
    def create_pipeline(self, fs: float, params: dict) -> Pipeline:
        """A stateful pipeline for one stream; process(block, bypass) works on float32 mono blocks."""

    def describe(self) -> dict:
        ok, why = self.available()
        return {"name": self.name, "available": ok, "detail": why}
