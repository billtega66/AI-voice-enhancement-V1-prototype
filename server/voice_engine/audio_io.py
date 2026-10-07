"""Audio file helpers (soundfile): read anything libsndfile supports, write 32-bit float WAV."""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def read_audio(src, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    data, fs = sf.read(src if not isinstance(src, (bytes, bytearray)) else io.BytesIO(src), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if max_seconds:
        mono = mono[: int(fs * max_seconds)]
    return mono, int(fs)


def wav_bytes(data: np.ndarray, fs: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(data, np.float32), fs, format="WAV", subtype="FLOAT")
    return buf.getvalue()
