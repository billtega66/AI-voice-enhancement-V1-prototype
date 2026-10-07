/* ==========================================================================
   Voice Profile: the single parameter set that BOTH the AI and the Mixer edit.
   The Mixer UI is generated from SCHEMA, and the Pipeline reads every key, so
   a control cannot exist without a processing path behind it.
   ========================================================================== */
(function (root) {
  'use strict';

  // Loaded from shared/voice_profile.schema.json (browser: schema.generated.js; Node: require).
  const DOC = root.VOICE_SCHEMA_DOC || (typeof require === 'function' ? require('../../shared/voice_profile.schema.json') : null);
  if (!DOC) throw new Error('Voice Profile schema missing: load src/schema.generated.js first');
  const GROUPS = DOC.groups;
  const SCHEMA = DOC.parameters;

  const DEFAULTS = Object.fromEntries(Object.entries(SCHEMA).map(([k, s]) => [k, s.def]));

  function clampValue(key, v) {
    const s = SCHEMA[key];
    if (!s) return undefined;
    if (s.type === 'bool') {
      if (typeof v === 'boolean') return v;
      if (v === 'true' || v === 1) return true;
      if (v === 'false' || v === 0) return false;
      return undefined;
    }
    const n = typeof v === 'string' ? parseFloat(v) : v;
    if (typeof n !== 'number' || !Number.isFinite(n)) return undefined;
    const q = Math.round(Math.min(s.max, Math.max(s.min, n)) / s.step) * s.step;
    return +q.toFixed(4);
  }

  /** Validate a {key: value} change set. Unknown keys and bad values are reported, not applied. */
  function validateChanges(changes) {
    const ok = {}, rejected = [];
    if (!changes || typeof changes !== 'object') return { ok, rejected: ['changes was not an object'] };
    for (const [k, v] of Object.entries(changes)) {
      const c = clampValue(k, v);
      if (c === undefined) rejected.push(k); else ok[k] = c;
    }
    return { ok, rejected };
  }

  function formatValue(key, v) {
    const s = SCHEMA[key];
    if (!s) return String(v);
    if (s.type === 'bool') return v ? 'on' : 'off';
    if (s.pct) return Math.round(v * 100) + '%';
    const d = s.step < 0.1 ? 2 : s.step < 1 ? 1 : 0;
    const sign = (s.unit === 'dB' || s.unit === 'st') && v > 0 ? '+' : '';
    return sign + v.toFixed(d) + (s.unit ? (s.unit === ':1' ? ':1' : ' ' + s.unit) : '');
  }

  /** Observable store. Every edit (AI, Mixer, reset, load) goes through set(). */
  class ProfileStore {
    constructor(params) { this.params = { ...DEFAULTS, ...(params || {}) }; this.listeners = []; this.log = []; this.version = 0; }
    get() { return { ...this.params }; }
    on(fn) { this.listeners.push(fn); }
    set(changes, source) {
      const { ok, rejected } = validateChanges(changes);
      const diff = [];
      for (const [k, v] of Object.entries(ok)) {
        if (this.params[k] !== v) { diff.push({ key: k, from: this.params[k], to: v, source }); this.params[k] = v; }
      }
      if (diff.length) {
        this.version++;
        this.log.push(...diff.map(d => ({ ...d, t: Date.now(), v: this.version })));
        if (this.log.length > 300) this.log.splice(0, this.log.length - 300);
        this.listeners.forEach(fn => fn(diff, source));
      }
      return { diff, rejected };
    }
    replace(params, source) {
      const full = { ...DEFAULTS, ...(params || {}) };
      return this.set(full, source);
    }
  }

  /* Plain-language descriptors for the simple view (0..1 bars), derived from the same params. */
  function describe(p) {
    const n = (v, lo, hi) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    return [
      { id: 'warmth', label: 'Warmth', v: n(p.warmthDb - p.mudDb * 0.15 - p.pitchSemitones * 0.8, -4, 8) },
      { id: 'clarity', label: 'Clarity', v: n(p.presenceDb + p.airDb * 0.5 - p.mudDb * 0.4, -4, 9) },
      { id: 'smooth', label: 'Softer S sounds', v: p.deEssEnabled ? n(p.deEssMaxDb + (-p.deEssThresholdDb - 30) * 0.25, 0, 16) : 0 },
      { id: 'steady', label: 'Steady volume', v: p.compEnabled ? n((1 - 1 / p.compRatio) * 10 + (-p.compThresholdDb - 15) * 0.1, 0, 11) : 0 },
      { id: 'clean', label: 'Background cleanup', v: (p.nsEnabled ? p.nsAmount * 0.6 : 0) + (p.vadEnabled ? 0.4 * n(-p.vadAttenuationDb, 0, 40) : 0) },
      { id: 'loud', label: 'Loudness', v: p.loudEnabled ? n(p.targetLufs, -30, -10) : n(p.outputGainDb, -24, 6) },
    ];
  }

  root.VoiceProfile = { GROUPS, SCHEMA, DEFAULTS, clampValue, validateChanges, formatValue, ProfileStore, describe };
})(typeof window !== 'undefined' ? window : globalThis);
