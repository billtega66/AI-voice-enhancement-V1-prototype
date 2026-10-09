"""Voice Profile: loaded from shared/voice_profile.schema.json, the single source of truth
that the browser (web/src/schema.generated.js) and the server both use."""
from __future__ import annotations

import json
import math
from pathlib import Path

import os


def _shared_dir() -> Path:
    """shared/ lives at the repo root. Found via VOICE_SHARED_DIR, the source tree, or the working directory."""
    for cand in (os.environ.get("VOICE_SHARED_DIR"), Path(__file__).resolve().parents[2] / "shared", Path.cwd() / "shared", Path.cwd().parent / "shared"):
        if cand and (Path(cand) / "voice_profile.schema.json").exists():
            return Path(cand)
    raise FileNotFoundError("shared/voice_profile.schema.json not found: run from the repo, use `pip install -e server`, or set VOICE_SHARED_DIR")


SHARED = _shared_dir()
SCHEMA_PATH = SHARED / "voice_profile.schema.json"
REFERENCE_PATH = SHARED / "reference_profile.json"

_SCHEMA_DOC = json.loads(SCHEMA_PATH.read_text())
SCHEMA: dict = _SCHEMA_DOC["parameters"]
GROUPS: list = _SCHEMA_DOC["groups"]
MODULES: list = _SCHEMA_DOC.get('modules', [])
DEFAULTS: dict = {k: s["def"] for k, s in SCHEMA.items()}
DEFAULT_REFERENCE: dict = json.loads(REFERENCE_PATH.read_text())


def clamp_value(key: str, v):
    """Return the validated value, or None if the key is unknown or the value unusable (mirrors profile.js)."""
    s = SCHEMA.get(key)
    if s is None:
        return None
    if s["type"] == "bool":
        if isinstance(v, bool):
            return v
        if v == "true" or (type(v) in (int, float) and v == 1):
            return True
        if v == "false" or (type(v) in (int, float) and v == 0):
            return False
        return None
    if isinstance(v, bool):
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n):
        return None
    n = min(s["max"], max(s["min"], n))
    q = math.floor(n / s["step"] + 0.5) * s["step"]  # JS Math.round semantics
    return round(q, 4)


def validate_changes(changes) -> tuple[dict, list]:
    ok, rejected = {}, []
    if not isinstance(changes, dict):
        return ok, ["changes was not an object"]
    for k, v in changes.items():
        c = clamp_value(k, v)
        if c is None:
            rejected.append(k)
        else:
            ok[k] = c
    return ok, rejected


def full_profile(params: dict | None) -> dict:
    """Defaults overlaid with validated params (invalid entries dropped)."""
    ok, _ = validate_changes(params or {})
    return {**DEFAULTS, **ok}
