"""AI Voice Profile Generator (server side). Natural language -> validated change set.

Two interpreters behind one interface, like web/src/ai.js:
  ClaudeInterpreter  - Anthropic Messages API, used when ANTHROPIC_API_KEY is set
  OfflineInterpreter - keyword rules, a line-for-line port of localInterpret() in ai.js
Both start from the CURRENT profile and return only changes; the caller validates them.
"""
from __future__ import annotations

import json
import os
import re

from ..analysis import match_to_reference
from ..profile import DEFAULTS, SCHEMA, validate_changes

A = re.ASCII  # JavaScript \w and \b are ASCII-only; match that behaviour

RULES = [
    (r"\b(s sounds?|s's|sibilan\w*|ess(es)?|hiss\w*|lisp)\b", "problem", lambda k, p, s: {"deEssEnabled": True, "deEssThresholdDb": p["deEssThresholdDb"] - 5 * k, "deEssMaxDb": p["deEssMaxDb"] + 3 * k}),
    (r"\b(thin|weak|tinny)\b", "problem", lambda k, p, s: {"warmthDb": p["warmthDb"] + 2 * k, "mudDb": min(0, p["mudDb"] + 1 * k)}),
    (r"\b(muffled|muddy|boomy|boxy|dull|unclear|mumbl\w*)\b", "problem", lambda k, p, s: {"mudDb": p["mudDb"] - 2 * k, "presenceDb": p["presenceDb"] + 1.5 * k}),
    (r"\b(harsh|shrill|piercing|edgy|nasal)\b|\bsharp\b(?! s\b| s sound| sibil| ess)", "problem", lambda k, p, s: {"presenceDb": p["presenceDb"] - 1.5 * k, "airDb": p["airDb"] - 1.5 * k}),
    (r"\b(noise|noisy|background|keyboard|typing|keys|fan|hum|buzz|paper|room|traffic|clicks?|air ?con\w*)\b", "problem", lambda k, p, s: {"nsEnabled": True, "nsAmount": p["nsAmount"] + 0.2 * k, "vadEnabled": True, "vadAttenuationDb": p["vadAttenuationDb"] - 8 * k}),
    (r"\b(inconsistent|uneven|jumps?|varies|fluctuat\w*|up and down)\b", "problem", lambda k, p, s: {"compEnabled": True, "compRatio": p["compRatio"] + 1 * k, "compThresholdDb": p["compThresholdDb"] - 4 * k, "makeupDb": p["makeupDb"] + 1 * k}),
    (r"\b(cut off|cuts off|chopp\w*|clipped words|missing words)\b", "problem", lambda k, p, s: {"vadHoldMs": p["vadHoldMs"] + 150 * k, "vadSensitivity": p["vadSensitivity"] + 0.15 * k}),
    (r"\b(too quiet|quiet|soft spoken|can't hear)\b", "problem", lambda k, p, s: {"targetLufs": p["targetLufs"] + 2 * k}),
    (r"\b(warm\w*|body|fuller|full|rich\w*)\b", "quality", lambda k, p, s: {"warmthDb": p["warmthDb"] + 2 * k * s}),
    (r"\b(deep\w*|bass\w*|lower (voice|pitch|tone))\b", "quality", lambda k, p, s: {"warmthDb": p["warmthDb"] + 1.5 * k * s, "pitchSemitones": p["pitchSemitones"] - 0.5 * k * s}),
    (r"\b(higher|lighter)\b", "quality", lambda k, p, s: {"pitchSemitones": p["pitchSemitones"] + 0.5 * k * s, "warmthDb": p["warmthDb"] - 1 * k * s}),
    (r"\b(clear\w*|crisp\w*|articulat\w*|intelligib\w*|understandable|present)\b", "quality", lambda k, p, s: {"presenceDb": p["presenceDb"] + 2 * k * s, "mudDb": min(0, p["mudDb"] - 1 * k * s)}),
    (r"\b(bright\w*|airy|sparkl\w*)\b|\bair\b(?! ?con)", "quality", lambda k, p, s: {"airDb": p["airDb"] + 2 * k * s}),
    (r"\b(consistent|steady|steadier|stable|controlled|compress\w*|more even|evenly|evener)\b", "quality", lambda k, p, s: {"compEnabled": True, "compRatio": p["compRatio"] + 1 * k * s, "compThresholdDb": p["compThresholdDb"] - 3 * k * s}),
    (r"\b(loud\w*)\b", "quality", lambda k, p, s: {"targetLufs": p["targetLufs"] + 2 * k * s}),
    (r"\b(quieter)\b", "quality", lambda k, p, s: {"targetLufs": p["targetLufs"] - 2 * k * s}),
]
RULES = [(re.compile(r, A), kind, fn) for r, kind, fn in RULES]


def _intensity(c: str) -> float:
    if re.search(r"\b(slightly|slight|a bit|bit|a little|little|touch|tad|somewhat|subtly)\b", c, A):
        return 0.5
    if re.search(r"\b(much|a lot|lots|very|really|way|significantly|strongly|heavily)\b", c, A):
        return 1.6
    return 1.0


def _podcast_changes(p, analysis, ref):
    if analysis and analysis.get("ok"):
        base = match_to_reference(analysis, ref, p)
    else:
        base = {"warmthDb": 2, "presenceDb": 2, "mudDb": -2, "deEssEnabled": True, "deEssMaxDb": 6, "compEnabled": True, "compRatio": 3,
                "compThresholdDb": -24, "makeupDb": 3, "nsEnabled": True, "nsAmount": 0.6, "loudEnabled": True,
                "targetLufs": ref["integratedLufs"], "limiterEnabled": True, "limiterCeilingDb": ref["truePeakDb"]}
    out = dict(base)
    for k in ("warmthDb", "presenceDb"):
        if p[k] > out.get(k, -99):
            out[k] = p[k]
    return out


def local_interpret(text: str, profile: dict, analysis: dict | None, ref: dict) -> dict:
    t = (text or "").lower().replace("\u2019", "'")
    p = dict(profile)
    touched: list[str] = []
    touch = lambda keys: [touched.append(k) for k in keys if k not in touched]
    matched, special = False, ""
    if re.search(r"\b(reset|start over|from scratch|default settings)\b", t, A):
        return {"reply": "I reset the voice to the neutral starting point.", "changes": dict(DEFAULTS), "matched": True}
    if re.search(r"\b(natural|less processed|subtle|untouched|original sound)\b", t, A) and not re.search(r"\b(more processed)\b", t, A):
        for k in ("warmthDb", "presenceDb", "airDb", "mudDb", "pitchSemitones", "makeupDb"):
            p[k] = p[k] / 2
        p["compRatio"] = 1 + (p["compRatio"] - 1) / 2
        touch(["warmthDb", "presenceDb", "airDb", "mudDb", "pitchSemitones", "makeupDb", "compRatio"])
        matched, special = True, "natural"
    if re.search(r"\b(podcast|professional|broadcast|radio|studio|announcer|presenter)\b", t, A):
        ch = _podcast_changes(p, analysis, ref)
        p.update(ch)
        touch(_podcast_changes(p, analysis, ref).keys())
        matched, special = True, "podcast"
    clauses = [c.strip() for c in re.split(r"[,.;!?]|\bbut\b|\band\b|\bthen\b|\balso\b", t, flags=A) if c and c.strip()]
    for c in clauses:
        k = _intensity(c)
        neg = bool(re.search(r"\b(less|not so|not as|too|without|reduce|cut|tone down|remove|decrease|turn down|lower the)\b", c, A))
        for rx, kind, fn in RULES:
            if not rx.search(c):
                continue
            if kind == "quality" and re.search(r"\btoo quiet\b", c, A):
                continue
            ch = fn(k, p, 1 if kind == "problem" else (-1 if neg else 1))
            p.update(ch)
            touch(ch.keys())
            matched = True
    return {"changes": {k: p[k] for k in touched}, "matched": matched, "special": special}


PLAIN = {
    "warmthDb": ("more warmth", "less warmth"), "mudDb": ("more low-mid body", "less boominess"), "presenceDb": ("more clarity", "less edge"),
    "airDb": ("more brightness", "less brightness"), "pitchSemitones": ("a slightly higher pitch", "a slightly deeper pitch"),
    "deEssThresholdDb": ("gentler S control", "softer S sounds"), "deEssMaxDb": ("softer S sounds", "gentler S control"),
    "compRatio": ("steadier volume", "more natural dynamics"), "compThresholdDb": ("more natural dynamics", "steadier volume"),
    "targetLufs": ("louder output", "quieter output"), "nsAmount": ("more background cleanup", "less background cleanup"),
    "vadAttenuationDb": ("less muting between words", "quieter pauses"),
}


def describe_diff(before: dict, after: dict) -> list[str]:
    out = []
    for k, (up, down) in PLAIN.items():
        if k in after and after[k] != before.get(k):
            ph = up if after[k] > before.get(k, 0) else down
            if ph not in out:
                out.append(ph)
    return out


class OfflineInterpreter:
    name = "offline"

    def interpret(self, text: str, ctx: dict) -> dict:
        r = local_interpret(text, ctx["profile"], ctx.get("analysis"), ctx["reference"])
        if not r["matched"]:
            return {"reply": "I could not map that to a sound change yet. Try words like warmer, clearer, deeper, steadier volume, less background noise, or softer S sounds.", "changes": {}, "rejected": []}
        ok, rej = validate_changes(r["changes"])
        changed = {k: v for k, v in ok.items() if ctx["profile"].get(k) != v}
        plain = describe_diff(ctx["profile"], changed)
        reply = r.get("reply") or ("Done: " + ", ".join(plain) + "." if plain else "Done.")
        return {"reply": reply, "changes": ok, "rejected": rej}


# ----------------------------------------------------------------------------- Claude


def _schema_text(profile: dict) -> str:
    lines = []
    for k, s in SCHEMA.items():
        if s["type"] == "bool":
            lines.append(f"{k} ({s['label']}): boolean, now {json.dumps(profile[k])}")
        else:
            unit = f" {s['unit']}" if s["unit"] else ""
            lines.append(f"{k} ({s['label']}): {s['min']}..{s['max']}{unit}, step {s['step']}, now {profile[k]}")
    return "\n".join(lines)


def build_messages(history: list, text: str, ctx: dict) -> tuple[str, list]:
    """Return (system prompt, messages) — same content the browser sends through Claude."""
    a = ctx.get("analysis") if ctx.get("analysis") and ctx["analysis"].get("ok") else None
    rounded = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in (a or {}).items()}
    system = "\n".join([
        "You are the voice-profile generator inside a real-time voice enhancement app. You never edit audio. You edit a parameter set that a DSP chain applies:",
        "input gain -> noise suppression -> speech detection (mutes non-speech) -> high-pass -> EQ (mud 350 Hz, warmth 180 Hz, presence 4 kHz, air 10 kHz shelf) -> pitch -> de-esser -> compressor + makeup -> loudness normalisation -> output gain -> limiter.",
        "", "PARAMETERS (current values are the starting point):", _schema_text(ctx["profile"]), "",
        "MEASURED USER VOICE: " + (json.dumps(rounded) if a else "not analysed yet"),
        "PRODUCTION REFERENCE (podcast target): " + json.dumps(ctx["reference"]),
        "PERSONALISED MATCH TO REFERENCE (deterministic suggestion from the measurements): " + json.dumps(match_to_reference(a, ctx["reference"], ctx["profile"]) if a else {}), "",
        "RULES:",
        "- Start from the current values and change only what the request needs. Never reset settings the user did not mention, including ones they set by hand in the Mixer.",
        "- \"slightly\"/\"a bit\" means small steps (EQ about 1 dB, ratio about 0.5); no qualifier means moderate (EQ 2-3 dB); \"much\" means larger.",
        "- Keep the speaker recognisable. Prefer warmth EQ for \"deeper\". Use pitchSemitones only when the user clearly asks for a deeper/higher voice, and stay within +/-1.5 unless asked for more.",
        "- For \"podcast\", \"professional\" or \"broadcast\", move toward the personalised match above rather than generic numbers.",
        "- Sharp S sounds: de-esser (lower threshold and/or more max reduction). Background, keyboard, paper or fan noise: noise suppression and speech detection. Uneven volume: compressor.",
        "- reply: one or two short sentences in the same language the user wrote in, plain words, no units or jargon (no dB, Hz, LUFS, ratio, compressor, EQ).",
        "Reply with only a JSON object: {\"reply\": string, \"changes\": {parameterKey: value}}. Include only keys you change. Use numbers for numeric keys and true/false for booleans.",
    ])
    msgs = []
    for h in (history or [])[-12:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            if msgs and msgs[-1]["role"] == h["role"]:
                msgs[-1]["content"] += "\n" + h["content"]
            else:
                msgs.append({"role": h["role"], "content": h["content"]})
    if msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n" + text
    else:
        msgs.append({"role": "user", "content": text})
    return system, msgs


def parse_ai_response(text_or_obj) -> dict:
    obj = text_or_obj
    if isinstance(text_or_obj, str):
        s = text_or_obj.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if m:
            s = m.group(1)
        i, j = s.find("{"), s.rfind("}")
        try:
            obj = json.loads(s[i:j + 1]) if i >= 0 and j > i else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return {"reply": None, "changes": {}, "rejected": ["response was not a JSON object"]}
    ok, rej = validate_changes(obj.get("changes") or {})
    reply = obj.get("reply") if isinstance(obj.get("reply"), str) else None
    return {"reply": reply[:600] if reply else None, "changes": ok, "rejected": rej}


class ClaudeInterpreter:
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        self.model = model or os.environ.get("VOICE_LLM_MODEL", "claude-haiku-4-5-20251001")
        if client is None:
            import anthropic  # optional dependency
            client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.client = client

    def interpret(self, text: str, ctx: dict) -> dict:
        system, msgs = build_messages(ctx.get("history", []), text, ctx)
        resp = self.client.messages.create(model=self.model, max_tokens=600, system=system, messages=msgs)
        out = "".join(getattr(b, "text", "") for b in resp.content)
        return parse_ai_response(out)


def get_interpreter():
    """Selection order: OpenAI-compatible gateway (LLM_BASE_URL/LLM_MODEL/LLM_API_KEY: Bailian DeepSeek,
    vLLM on ROCm, ...), then Claude (ANTHROPIC_API_KEY), then the offline rules."""
    from .gateway import GatewayInterpreter, gateway_configured
    if gateway_configured():
        return GatewayInterpreter()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeInterpreter()
        except Exception:  # noqa: BLE001 - missing SDK or bad config: fall back, never crash the server
            pass
    return OfflineInterpreter()
