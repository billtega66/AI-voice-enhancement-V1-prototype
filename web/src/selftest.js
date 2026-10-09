/* ==========================================================================
   Self-test: offline checks of every processing path at 48 kHz.
   Each test maps to a numbered item in requirements section 14 where relevant.
   ========================================================================== */
(function (root) {
  'use strict';
  const D = root.DSP, VP = root.VoiceProfile, VA = root.VoiceAnalysis, AI = root.VoiceAI;
  const FS = 48000;
  const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : String(v));
  const sgn = (v, d = 1) => (v > 0 ? '+' : '') + fmt(v, d);
  function sine(f, sec, db = -12) { const n = Math.round(sec * FS), a = D.dbToLin(db), x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = a * Math.sin(2 * Math.PI * f * i / FS); return x; }
  const lvl = (x, s, e) => D.linToDb(D.rms(x, Math.round(s * FS), Math.round(e * FS)));
  const ONLY = (extra) => ({ ...VP.DEFAULTS, nsEnabled: false, vadEnabled: false, hpfEnabled: false, deEssEnabled: false, compEnabled: false, loudEnabled: false, limiterEnabled: false, makeupDb: 0, ...extra });
  const render = (x, params, chunk = 1024) => D.renderOffline(x, FS, params, chunk);

  const TESTS = [
    { id: 'sample', req: '–', name: 'Sample voice is finite and contains speech, keyboard and paper events', run() {
      const s = D.makeSampleVoice(FS); let fin = true; for (const v of s.data) if (!Number.isFinite(v)) { fin = false; break; }
      const types = [...new Set(s.events.map(e => e.type))];
      return { pass: fin && types.length === 3, detail: `${s.events.length} events (${types.join(', ')}), ${fmt(s.seconds, 0)} s` }; } },

    { id: 'ns', req: '8', name: 'Noise suppression lowers steady noise and keeps speech', run() {
      const s = D.makeSampleVoice(FS, { noiseDb: -40, events: [{ type: 'speech', s: 1.2, e: 3.2, db: -22 }], seconds: 4 });
      const o = render(s.data, ONLY({ nsEnabled: true, nsAmount: 0.6 })).data;
      const noise = lvl(o, 3.5, 3.95) - lvl(s.data, 3.5, 3.95), sp = lvl(o, 1.6, 3.0) - lvl(s.data, 1.6, 3.0);
      return { pass: noise < -9 && sp > -3, detail: `noise ${sgn(noise)} dB, speech ${sgn(sp)} dB` }; } },

    { id: 'nslat', req: '–', name: 'Noise suppressor latency is reported exactly (for Original/Enhanced alignment)', run() {
      const ns = new D.NoiseSuppressor(FS); ns.amount = 0;
      const x = new Float32Array(4096); x[100] = 1; const y = x.slice(); ns.process(y);
      let at = -1, m = 0; for (let i = 0; i < y.length; i++) if (Math.abs(y[i]) > m) { m = Math.abs(y[i]); at = i; }
      return { pass: at - 100 === ns.latency && Math.abs(m - 1) < 1e-3, detail: `impulse moved ${at - 100} samples (reported ${ns.latency}), gain ${fmt(m, 4)}` }; } },

    { id: 'vad', req: '9', name: 'Speech detection opens for speech, stays shut for keyboard and paper', run() {
      const s = D.makeSampleVoice(FS), g = new D.SpeechGate(FS), hop = g.hop, b = new Float32Array(hop);
      const tally = { speech: [0, 0], keys: [0, 0], paper: [0, 0], none: [0, 0] };
      for (let i = 0; i + hop <= s.data.length; i += hop) {
        b.set(s.data.subarray(i, i + hop)); g.process(b, false);
        const t = i / FS, ty = s.at(t), ev = s.events.find(e => t >= e.s && t < e.e);
        if (ty === 'speech' && (t - ev.s < 0.15 || ev.e - t < 0.1)) continue;
        if (ty === 'none' && s.events.some(e => e.type === 'speech' && t >= e.e && t < e.e + 0.45)) continue; // hold time after speech
        tally[ty][0]++; if (g.speech) tally[ty][1]++;
      }
      const hit = tally.speech[1] / tally.speech[0], fk = tally.keys[1] / tally.keys[0], fp = tally.paper[1] / tally.paper[0];
      const eo = D.energyOnlyDecisions(s.data, FS), eoKeys = eo.filter(d => s.at(d.t) === 'keys'), eoRate = eoKeys.filter(d => d.on).length / eoKeys.length;
      return { pass: hit > 0.9 && fk < 0.05 && fp < 0.05, detail: `speech ${fmt(hit * 100, 0)}%, keyboard ${fmt(fk * 100, 0)}%, paper ${fmt(fp * 100, 0)}% (an energy-only gate opens on ${fmt(eoRate * 100, 0)}% of keyboard frames)` }; } },

    { id: 'vadmute', req: '9', name: 'Non-speech muting lowers keyboard noise in the output', run() {
      const s = D.makeSampleVoice(FS);
      const on = render(s.data, ONLY({ vadEnabled: true, vadAttenuationDb: -30 })).data, off = render(s.data, ONLY({})).data;
      const k = s.events.find(e => e.type === 'keys'), d = lvl(on, k.s + 0.2, k.e) - lvl(off, k.s + 0.2, k.e);
      const sp = s.events.find(e => e.type === 'speech'), ds = lvl(on, sp.s + 0.3, sp.e - 0.1) - lvl(off, sp.s + 0.3, sp.e - 0.1);
      return { pass: d < -20 && Math.abs(ds) < 1, detail: `keyboard ${sgn(d)} dB, speech ${sgn(ds, 2)} dB` }; } },

    { id: 'eq', req: '10', name: 'EQ controls change their own frequency bands', run() {
      const cases = [['warmthDb', 180, 6], ['presenceDb', 4000, 6], ['airDb', 14000, 6], ['mudDb', 350, -6]], res = [];
      for (const [k, f, g] of cases) { const x = sine(f, 0.6), o = render(x, ONLY({ [k]: g })).data; res.push([k, lvl(o, 0.3, 0.6) - lvl(x, 0.3, 0.6), g]); }
      const x = sine(1000, 0.6), o = render(x, ONLY({ hpfEnabled: true, hpfHz: 200 })).data, x2 = sine(60, 0.6), o2 = render(x2, ONLY({ hpfEnabled: true, hpfHz: 200 })).data;
      const hp1k = lvl(o, 0.3, 0.6) - lvl(x, 0.3, 0.6), hp60 = lvl(o2, 0.3, 0.6) - lvl(x2, 0.3, 0.6);
      const ok = res.every(([, m, g]) => Math.abs(m - g) < 1.2) && hp60 < -15 && Math.abs(hp1k) < 0.5;
      return { pass: ok, detail: res.map(([k, m]) => `${k.replace('Db', '')} ${sgn(m)}`).join(', ') + `, high-pass 60 Hz ${sgn(hp60)} / 1 kHz ${sgn(hp1k, 2)} dB` }; } },

    { id: 'deess', req: '12', name: 'De-esser reduces the sibilant band only', run() {
      const P = ONLY({ deEssEnabled: true, deEssFreqHz: 6500, deEssThresholdDb: -30, deEssMaxDb: 10 });
      const hi = sine(6500, 0.6, -10), lo = sine(500, 0.6, -10);
      const dh = lvl(render(hi, P).data, 0.3, 0.6) + 10, dl = lvl(render(lo, P).data, 0.3, 0.6) - lvl(lo, 0.3, 0.6);
      return { pass: dh < -5 && Math.abs(dl) < 0.5, detail: `6.5 kHz ${sgn(dh)} dB, 500 Hz ${sgn(dl, 2)} dB` }; } },

    { id: 'comp', req: '11', name: 'Compressor narrows the gap between loud and quiet passages', run() {
      const a = sine(300, 1, -36), b = sine(300, 1, -12), x = new Float32Array(a.length * 2); x.set(a); x.set(b, a.length);
      const o = render(x, ONLY({ compEnabled: true, compThresholdDb: -36, compRatio: 4, compAttackMs: 5, compReleaseMs: 100, makeupDb: 0 })).data;
      const inGap = 24, outGap = lvl(o, 1.5, 2) - lvl(o, 0.5, 1);
      return { pass: outGap < 9, detail: `level gap ${inGap} dB in, ${fmt(outGap)} dB out` }; } },

    { id: 'loud', req: '–', name: 'Loudness normalisation brings quiet speech to the target', run() {
      const s = D.makeSampleVoice(FS, { events: [{ type: 'speech', s: 0.3, e: 9.7, db: -40 }], seconds: 10 });
      const o = render(s.data, ONLY({ loudEnabled: true, targetLufs: -16 })).data;
      const L = D.integratedLufs(o.subarray(6 * FS), FS).integrated;
      return { pass: Math.abs(L + 16) < 1.5, detail: `input about ${fmt(D.integratedLufs(s.data, FS).integrated)} LUFS, output ${fmt(L)} LUFS (target -16)` }; } },

    { id: 'limit', req: '13', name: 'Limiter holds the ceiling and output gain scales the level', run() {
      const x = sine(440, 1, 0); for (let i = 0; i < x.length; i += 997) x[i] = 2.5;
      const pk = D.linToDb(D.peak(render(x, ONLY({ limiterEnabled: true, limiterCeilingDb: -1, outputGainDb: 6 })).data));
      const y = sine(440, 1, -30), g = lvl(render(y, ONLY({ outputGainDb: -6 })).data, 0.2, 1) - lvl(y, 0.2, 1);
      return { pass: pk <= -1 + 1e-4 && Math.abs(g + 6) < 0.05, detail: `peak ${fmt(pk, 3)} dBFS with +6 dB into it; output gain -6 gives ${sgn(g, 2)} dB` }; } },

    { id: 'bypass', req: '16', name: 'Bypass returns the original signal bit for bit', run() {
      const s = D.makeSampleVoice(FS), pl = new D.Pipeline(FS, { ...VP.DEFAULTS, warmthDb: 4, pitchSemitones: 1 });
      let same = true;
      for (let i = 0; i + 1024 <= s.data.length; i += 1024) { const b = s.data.slice(i, i + 1024); pl.process(b, true); for (let j = 0; j < 1024; j++) if (b[j] !== s.data[i + j]) { same = false; break; } }
      return { pass: same, detail: same ? 'identical' : 'differs' }; } },

    { id: 'align', req: '4', name: 'Enhanced render is time-aligned with the original', run() {
      const x = sine(220, 1, -20), o = render(x, ONLY({ nsEnabled: true, nsAmount: 0, vadEnabled: false })).data;
      let best = 0, at = 0;
      for (let lag = -200; lag <= 200; lag++) { let s = 0; for (let i = 2000; i < 30000; i++) s += x[i] * o[i + lag]; if (s > best) { best = s; at = lag; } }
      return { pass: at === 0 || Math.abs(at) === 218 /* one period of 220 Hz */, detail: `best alignment offset ${at} samples` }; } },

    { id: 'every', req: '5, 6, 10-13', name: 'Every Mixer parameter changes the processed audio', run() {
      const s = D.makeSampleVoice(FS, { seconds: 6, events: [{ type: 'speech', s: 0.5, e: 2.5, db: -22 }, { type: 'keys', s: 2.8, e: 3.4, db: -24 }, { type: 'speech', s: 3.8, e: 5.6, db: -46 }] });
      const base = { ...VP.DEFAULTS, outputGainDb: 4, limiterCeilingDb: -3, compThresholdDb: -40, metallicMix: 0.2, distortionDrive: 0.2, echoMix: 0.2 };
      const ref = render(s.data, base, 512).data, dead = [];
      for (const [k, sc] of Object.entries(VP.SCHEMA)) {
        const v = sc.type === 'bool' ? !base[k] : (sc.max - base[k] > base[k] - sc.min ? sc.max : sc.min);
        const o = render(s.data, { ...base, [k]: v }, 512).data;
        let d = 0; for (let i = 0; i < o.length; i++) d += Math.abs(o[i] - ref[i]);
        if (!(d / o.length > 1e-6)) dead.push(k);
      }
      const n = Object.keys(VP.SCHEMA).length;
      return { pass: !dead.length, detail: dead.length ? `no effect: ${dead.join(', ')}` : `${n} of ${n} parameters change the output` }; } },

    { id: 'chunks', req: '–', name: 'Output does not depend on chunk size (live and preview match)', run() {
      const s = D.makeSampleVoice(FS, { seconds: 4 }), a = render(s.data, VP.DEFAULTS, 256).data, b = render(s.data, VP.DEFAULTS, 2048).data;
      let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
      return { pass: m < 1e-3, detail: `max difference ${m.toExponential(1)}` }; } },

    { id: 'stable', req: '–', name: 'Stable on silence, clipping and extreme settings', run() {
      const P = { ...VP.DEFAULTS }; for (const [k, s] of Object.entries(VP.SCHEMA)) if (s.type === 'num') P[k] = s.max;
      const pl = new D.Pipeline(FS, P); let seed = 9, bad = 0;
      const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296 * 2 - 1; };
      for (let k = 0; k < 400; k++) { const b = new Float32Array(512); for (let i = 0; i < 512; i++) b[i] = k % 3 === 0 ? 0 : k % 3 === 1 ? rnd() * 5 : 1e-30; pl.process(b); for (const v of b) if (!Number.isFinite(v) || Math.abs(v) > 1) { bad++; break; } }
      return { pass: bad === 0, detail: `400 blocks at every slider maximum, ${bad} bad` }; } },

    { id: 'speed', req: '–', name: 'Full chain runs faster than real time', run() {
      const s = D.makeSampleVoice(FS, { seconds: 6 }), t0 = D.now(); render(s.data, VP.DEFAULTS, 1024); const ms = D.now() - t0;
      return { pass: ms / 6000 < 0.5, detail: `6 s of audio in ${fmt(ms, 0)} ms (real-time factor ${fmt(ms / 6000, 3)})` }; } },

    { id: 'analysis', req: '–', name: 'Voice analysis measures pitch, loudness, noise and pauses', run() {
      const s = D.makeSampleVoice(FS), a = VA.analyzeVoice(s.data, FS);
      const ok = a.ok && a.f0MedianHz > 110 && a.f0MedianHz < 175 && Number.isFinite(a.integratedLufs) && a.noiseFloorDb < -45 && a.pausesPerMin > 5 && a.syllablesPerSec > 2 && a.syllablesPerSec < 6;
      return { pass: ok, detail: a.ok ? `F0 ${fmt(a.f0MedianHz, 0)} Hz, ${fmt(a.integratedLufs)} LUFS, floor ${fmt(a.noiseFloorDb, 0)} dB, ${fmt(a.syllablesPerSec)} syll/s, ${fmt(a.pausesPerMin, 0)} pauses/min` : a.reason }; } },

    { id: 'personal', req: '–', name: 'Matching to the reference adapts to the voice and never changes pitch', run() {
      const ref = VA.DEFAULT_REFERENCE, base = { ...VP.DEFAULTS };
      const thin = VA.matchToReference({ ok: true, lowMidRelDb: -14, presenceRelDb: -14, sibilanceRelDb: -24, sibilanceMedianDb: -45, sibilancePeakDb: -35, loudnessRangeLu: 6, rmsDb: -25, noiseFloorDb: -65 }, ref, base);
      const boomy = VA.matchToReference({ ok: true, lowMidRelDb: -1, presenceRelDb: -20, sibilanceRelDb: -16, sibilanceMedianDb: -40, sibilancePeakDb: -30, loudnessRangeLu: 12, rmsDb: -25, noiseFloorDb: -45 }, ref, base);
      const ok = thin.warmthDb > 0 && boomy.mudDb < 0 && boomy.presenceDb > thin.presenceDb && boomy.deEssMaxDb > thin.deEssMaxDb && boomy.compRatio > thin.compRatio && !('pitchSemitones' in thin) && !('pitchSemitones' in boomy);
      return { pass: ok, detail: `thin voice: warmth ${sgn(thin.warmthDb)}; boomy voice: mud ${sgn(boomy.mudDb)}, ratio ${boomy.compRatio} vs ${thin.compRatio}` }; } },

    { id: 'store', req: '7', name: 'AI and Mixer edit the same profile without resetting each other', run() {
      const st = new VP.ProfileStore();
      const r1 = AI.localInterpret('Make my voice warmer and more professional', st.get(), null, VA.DEFAULT_REFERENCE); st.set(r1.changes, 'ai');
      st.set({ warmthDb: st.get().warmthDb + 1.5 }, 'mixer'); const afterMixer = st.get();
      const r2 = AI.localInterpret('Keep everything else but make it slightly clearer', st.get(), null, VA.DEFAULT_REFERENCE); st.set(r2.changes, 'ai');
      const fin = st.get();
      const ok = fin.warmthDb === afterMixer.warmthDb && fin.presenceDb > afterMixer.presenceDb && fin.compRatio === afterMixer.compRatio && Object.keys(r2.changes).every(k => ['presenceDb', 'mudDb'].includes(k));
      return { pass: ok, detail: `warmth kept at ${sgn(fin.warmthDb)} dB after the Mixer edit; clarity ${sgn(afterMixer.presenceDb)} → ${sgn(fin.presenceDb)} dB; second prompt touched ${Object.keys(r2.changes).join(', ')}` }; } },

    { id: 'interp', req: '5', name: 'Offline interpreter maps everyday words to the right controls', run() {
      const p = { ...VP.DEFAULTS }, I = (t) => AI.localInterpret(t, p, null, VA.DEFAULT_REFERENCE).changes;
      const s1 = I('make it slightly warmer').warmthDb, s2 = I('make it much warmer').warmthDb;
      const ds = I('reduce the sharp S sounds'), deep = I('slightly deeper'), noise = I('there is keyboard noise in the background');
      const less = I('a bit less bright').airDb, lvl2 = I('my voice sounds thin and inconsistent, keep the volume consistent');
      const ok = s1 > 0 && s2 > s1 && ds.deEssThresholdDb < p.deEssThresholdDb && !('presenceDb' in ds) && deep.warmthDb > 0 && deep.pitchSemitones < 0 && deep.pitchSemitones >= -0.5
        && noise.nsAmount > p.nsAmount && less < 0 && lvl2.warmthDb > 0 && lvl2.compRatio > p.compRatio;
      return { pass: ok, detail: `warmer: slightly ${sgn(s1)} vs much ${sgn(s2)}; S sounds threshold ${ds.deEssThresholdDb}; deeper pitch ${deep.pitchSemitones} st; less bright air ${less}` }; } },

    { id: 'validate', req: '5', name: 'AI responses are validated and clamped before they reach the engine', run() {
      const r = AI.parseAIResponse({ reply: 'ok', changes: { warmthDb: 50, presenceDb: '2.3', compEnabled: 'false', hackerKey: 1, airDb: 'loud' } });
      const ok = r.ok.warmthDb === 9 && r.ok.presenceDb === 2.5 && r.ok.compEnabled === false && r.rejected.includes('hackerKey') && r.rejected.includes('airDb');
      return { pass: ok, detail: `warmth 50 → ${r.ok.warmthDb}, presence "2.3" → ${r.ok.presenceDb}, rejected ${r.rejected.join(', ')}` }; } },
  ];

  root.SelfTest = { TESTS, runAll() { return TESTS.map(t => { const t0 = D.now(); let r; try { r = t.run(); } catch (e) { r = { pass: false, detail: 'threw: ' + e.message }; } return { id: t.id, req: t.req, name: t.name, ...r, ms: D.now() - t0 }; }); } };
})(typeof window !== 'undefined' ? window : globalThis);
