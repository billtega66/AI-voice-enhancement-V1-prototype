"""Processing backends. Every backend builds a Pipeline with the same Voice Profile contract,
so the frontend, API and evaluation code never change when the compute device does."""
from .base import ProcessingBackend
from .cpu import CpuBackend
from .rocm import RocmBackend
from .registry import available_backends, get_backend

__all__ = ["ProcessingBackend", "CpuBackend", "RocmBackend", "available_backends", "get_backend"]
