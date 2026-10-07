/* ==========================================================================
   Voice analysis: measures the user's own voice so the profile is adapted to
   them instead of pushing every speaker to the same settings.
   ========================================================================== */
(function (root) {
  'use strict';
  const D = root.DSP;

  /* Production reference for a podcast/broadcast sound, from shared/reference_profile.json.
     Editable in the Mixer; paste the team's measured values there or in the JSON file. */
  const DEFAULT_REFERENCE = root.VOICE_REFERENCE || (typeof require === 'function' ? require('../../shared/reference_profile.json') : null);

  function bandEnergy(spec, fs, N, lo, hi) {
    let s = 0;
    const a = Math.max(1, Math.floor(lo * N / fs)), b = Math.min(N / 2, Math.ceil(hi * N / fs));
    for (let j = a; j <= b; j++) s += spec[j];
    return s;
  }

  /** Measure a recording. Returns plain numbers; null for anything that cannot be measured. */
  function analyzeVoice(data, fs) {
    const gate = new D.SpeechGate(fs);
    const hop = gate.hop, frames = [];
    const blk = new Float32Array(hop);
    for (let i = 0; i + hop <= data.length; i += hop) {
      blk.set(data.subarray(i, i + hop));
      gate.process(blk, false);
      frames.push({ t: i / fs, speech: gate.speech, active: gate.active, voiced: gate.voiced, f0: gate.voiced ? gate.f0 : 0, r: gate.periodicity, db: D.linToDb(D.rms(data, i, i + hop)) });
    }
    const speechFrames = frames.filter(f => f.speech), voiced = frames.filter(f => f.voiced && f.f0 > 0);
    const speechS = speechFrames.length * hop / fs, durationS = data.length / fs;
    if (speechS < 0.8) return { ok: false, reason: 'Not enough speech was detected. Record at least a few seconds of talking.', durationS, speechS };

    // F0
    const f0s = voiced.map(f => f.f0).sort((a, b) => a - b);
    const q = (arr, p) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(p * (arr.length - 1))))];
    const f0Median = f0s.length ? q(f0s, 0.5) : null;
    const f0Mean = f0s.length ? f0s.reduce((a, b) => a + b, 0) / f0s.length : null;
    const f0RangeSt = f0s.length > 5 ? 12 * Math.log2(q(f0s, 0.9) / q(f0s, 0.1)) : null;
    const rs = voiced.map(f => Math.min(0.995, f.r));
    const rMean = rs.length ? rs.reduce((a, b) => a + b, 0) / rs.length : null;
    const hnrDb = rMean ? 10 * Math.log10(rMean / (1 - rMean)) : null;

    // Long-term spectrum of speech frames
    const N = 2048, fft = new D.FFT(N), re = new Float64Array(N), im = new Float64Array(N), spec = new Float64Array(N / 2 + 1);
    let nSpec = 0;
    for (let k = 0; k < speechFrames.length; k += 2) {
      const i0 = Math.round(speechFrames[k].t * fs);
      if (i0 + N > data.length) break;
      for (let i = 0; i < N; i++) { const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N); re[i] = data[i0 + i] * w; im[i] = 0; }
      fft.transform(re, im);
      for (let j = 0; j <= N / 2; j++) spec[j] += re[j] * re[j] + im[j] * im[j];
      nSpec++;
    }
    const total = bandEnergy(spec, fs, N, 100, 8000) || 1e-20;
    const rel = (lo, hi) => D.powToDb(bandEnergy(spec, fs, N, lo, hi) / total);
    let cNum = 0, cDen = 0;
    for (let j = 1; j <= N / 2; j++) { const f = j * fs / N; if (f < 100 || f > 8000) continue; cNum += f * spec[j]; cDen += spec[j]; }
    // spectral slope: regression of band level (dB) on octave, 1/3-octave-ish bands 125 Hz - 8 kHz
    const pts = [];
    for (let f = 125; f <= 8000; f *= Math.SQRT2) { const e = bandEnergy(spec, fs, N, f / 1.19, f * 1.19); if (e > 0) pts.push([Math.log2(f), D.powToDb(e)]); }
    let slope = null;
    if (pts.length > 3) {
      const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length, my = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      let num = 0, den = 0; for (const [x, y] of pts) { num += (x - mx) * (y - my); den += (x - mx) ** 2; }
      slope = num / den;
    }

    // Level and loudness
    let se = 0, sn = 0, pk = 0;
    for (const f of speechFrames) { const i0 = Math.round(f.t * fs); for (let i = i0; i < i0 + hop && i < data.length; i++) { se += data[i] * data[i]; sn++; } }
    for (let i = 0; i < data.length; i++) { const a = Math.abs(data[i]); if (a > pk) pk = a; }
    const rmsDb = D.powToDb(se / Math.max(1, sn));
    const lu = D.integratedLufs(data, fs);
    const speechBlocks = lu.blocks.filter((l, i) => { const t = i * 0.1 + 0.2; const fr = frames[Math.min(frames.length - 1, Math.round(t * fs / hop))]; return fr && fr.speech && l > -70; }).sort((a, b) => a - b);
    const lra = speechBlocks.length > 4 ? q(speechBlocks, 0.95) - q(speechBlocks, 0.10) : null;
    const quiet = frames.filter(f => !f.speech).map(f => f.db).sort((a, b) => a - b);
    const noiseFloorDb = quiet.length ? q(quiet, 0.5) : null;

    // Sibilance peaks: fraction of speech frames whose 5-9 kHz band is strong
    const sibBand = new D.Biquad().set('bandpass', fs, 6500, 1.2), sibLv = [];
    for (let i = 0, fi = 0; i + hop <= data.length; i += hop, fi++) {
      let s = 0; for (let j = i; j < i + hop; j++) { const y = sibBand.tick(data[j]); s += y * y; }
      if (frames[fi] && frames[fi].speech) sibLv.push(D.powToDb(s / hop));
    }
    sibLv.sort((a, b) => a - b);

    // Speaking rate: syllable nuclei = peaks of the smoothed speech envelope
    const env = frames.map(f => (f.speech ? f.db : -120));
    let syll = 0, lastPk = -1;
    for (let i = 2; i < env.length - 2; i++) {
      const e = env[i];
      if (e > -60 && e >= env[i - 1] && e >= env[i + 1] && e >= env[i - 2] && e >= env[i + 2] && i - lastPk >= 10) {
        const lo = Math.min(...env.slice(Math.max(0, i - 12), i)), lo2 = Math.min(...env.slice(i + 1, i + 13));
        if (e - Math.max(lo, lo2) > 3) { syll++; lastPk = i; }
      }
    }
    // Pauses: silences >= 250 ms between speech
    const pauses = []; let run = 0, seen = false;
    for (const f of frames) {
      if (f.speech) { if (seen && run * hop / fs >= 0.25) pauses.push(run * hop / fs); run = 0; seen = true; }
      else if (seen) run++;
    }

    return {
      ok: true, durationS, speechS,
      f0MedianHz: f0Median, f0MeanHz: f0Mean, f0RangeSt, hnrDb,
      formants: null, // not measured in the browser build (needs LPC analysis; planned for the ROCm build)
      spectralCentroidHz: cDen ? cNum / cDen : null, spectralSlopeDbOct: slope,
      lowMidRelDb: rel(150, 500), presenceRelDb: rel(2000, 5000), sibilanceRelDb: rel(5000, 9000),
      sibilancePeakDb: sibLv.length ? q(sibLv, 0.95) : null, sibilanceMedianDb: sibLv.length ? q(sibLv, 0.5) : null,
      rmsDb, integratedLufs: lu.integrated, loudnessRangeLu: lra, crestDb: D.linToDb(pk) - rmsDb, noiseFloorDb,
      syllablesPerSec: speechS > 0 ? syll / speechS : null,
      pausesPerMin: pauses.length / (durationS / 60), pauseMeanS: pauses.length ? pauses.reduce((a, b) => a + b, 0) / pauses.length : 0,
    };
  }

  /** Personalised DSP settings that move THIS voice toward the reference.
      Pitch is never changed here: the user's own F0 is kept. */
  function matchToReference(a, ref, base) {
    if (!a || !a.ok) return {};
    const c = D.clamp, p = {};
    const lowGap = ref.lowMidRelDb - a.lowMidRelDb;
    if (lowGap > 0) { p.warmthDb = c(Math.round(lowGap * 0.6 * 2) / 2, 0, 6); p.mudDb = 0; }
    else { p.warmthDb = 0; p.mudDb = c(Math.round(lowGap * 0.6 * 2) / 2, -6, 0); }
    p.presenceDb = c(Math.round((ref.presenceRelDb - a.presenceRelDb) * 0.6 * 2) / 2, -3, 6);
    const sibEx = a.sibilanceRelDb - ref.sibilanceRelDb;
    p.deEssEnabled = true;
    p.deEssMaxDb = c(Math.round(4 + Math.max(0, sibEx) * 1.2), 3, 12);
    const inGain = base.inputGainDb || 0;
    if (a.sibilancePeakDb !== null) p.deEssThresholdDb = c(Math.round(a.sibilanceMedianDb + inGain + 6 - Math.max(0, sibEx)), -60, -6);
    const lra = a.loudnessRangeLu ?? ref.loudnessRangeLu;
    p.compEnabled = true;
    p.compRatio = c(Math.round((2 + Math.max(0, lra - ref.loudnessRangeLu) * 0.35) * 10) / 10, 1.5, 6);
    p.compThresholdDb = c(Math.round(a.rmsDb + inGain - 2), -50, -6);
    p.makeupDb = c(Math.round((1 - 1 / p.compRatio) * 6 * 2) / 2, 0, 8);
    if (a.rmsDb < -40) p.inputGainDb = c(Math.round(-30 - a.rmsDb), 0, 18);
    p.nsEnabled = true;
    p.nsAmount = a.noiseFloorDb > -45 ? 0.85 : a.noiseFloorDb > ref.noiseFloorDb ? 0.65 : 0.4;
    p.loudEnabled = true; p.targetLufs = c(ref.integratedLufs, -30, -10);
    p.limiterEnabled = true; p.limiterCeilingDb = c(ref.truePeakDb, -6, 0);
    return p;
  }

  /** Plain-language description of a voice for the simple view (no jargon). */
  function plainSummary(a) {
    if (!a || !a.ok) return a && a.reason ? [a.reason] : [];
    const out = [];
    if (a.f0MedianHz) out.push(a.f0MedianHz < 145 ? 'Your voice sits in a lower range.' : a.f0MedianHz < 200 ? 'Your voice sits in a middle range.' : 'Your voice sits in a higher range.');
    if (a.lowMidRelDb < -9) out.push('It sounds a little thin, so there is room to add body.');
    else if (a.lowMidRelDb > -3) out.push('It has plenty of body, maybe slightly boomy.');
    if (a.sibilanceRelDb > -20) out.push('S sounds come through quite sharply.');
    if (a.noiseFloorDb !== null && a.noiseFloorDb > -55) out.push('There is noticeable background noise between phrases.');
    if (a.loudnessRangeLu !== null && a.loudnessRangeLu > 9) out.push('Your volume changes a lot from phrase to phrase.');
    if (a.rmsDb < -38) out.push('The recording is quiet; speaking closer to the microphone will help.');
    return out;
  }

  root.VoiceAnalysis = { DEFAULT_REFERENCE, analyzeVoice, matchToReference, plainSummary };
})(typeof window !== 'undefined' ? window : globalThis);
