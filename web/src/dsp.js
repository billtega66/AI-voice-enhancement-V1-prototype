/* ==========================================================================
   Real-time audio engine (browser reference backend, "browser-dsp").
   Every class processes mono Float32Array blocks in place and keeps its own
   state, so blocks of any size give the same result.
   ========================================================================== */
(function (root) {
  'use strict';
  const dbToLin = (db) => Math.pow(10, db / 20);
  const linToDb = (x) => 20 * Math.log10(Math.max(Math.abs(x), 1e-10));
  const powToDb = (p) => 10 * Math.log10(Math.max(p, 1e-20));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : () => Date.now();

  function rms(b, s = 0, e = b.length) { let a = 0; for (let i = s; i < e; i++) a += b[i] * b[i]; return Math.sqrt(a / Math.max(1, e - s)); }
  function peak(b) { let p = 0; for (let i = 0; i < b.length; i++) { const a = b[i] < 0 ? -b[i] : b[i]; if (a > p) p = a; } return p; }

  /* ---------------- Biquad (RBJ cookbook) ---------------- */
  class Biquad {
    constructor() { this.b0 = 1; this.b1 = this.b2 = this.a1 = this.a2 = 0; this.reset(); }
    reset() { this.x1 = this.x2 = this.y1 = this.y2 = 0; }
    set(type, fs, f0, Q, gainDb = 0) {
      f0 = Math.min(f0, fs * 0.45);
      const A = Math.pow(10, gainDb / 40), w = 2 * Math.PI * f0 / fs, c = Math.cos(w), s = Math.sin(w), al = s / (2 * Q);
      const sq = 2 * Math.sqrt(A) * al;
      let b0, b1, b2, a0, a1, a2;
      switch (type) {
        case 'highpass': b0 = (1 + c) / 2; b1 = -(1 + c); b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; break;
        case 'lowpass': b0 = (1 - c) / 2; b1 = 1 - c; b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; break;
        case 'bandpass': b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; break;
        case 'peaking': b0 = 1 + al * A; b1 = -2 * c; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * c; a2 = 1 - al / A; break;
        case 'lowshelf':
          b0 = A * ((A + 1) - (A - 1) * c + sq); b1 = 2 * A * ((A - 1) - (A + 1) * c); b2 = A * ((A + 1) - (A - 1) * c - sq);
          a0 = (A + 1) + (A - 1) * c + sq; a1 = -2 * ((A - 1) + (A + 1) * c); a2 = (A + 1) + (A - 1) * c - sq; break;
        case 'highshelf':
          b0 = A * ((A + 1) + (A - 1) * c + sq); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - sq);
          a0 = (A + 1) - (A - 1) * c + sq; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - sq; break;
        default: throw new Error('biquad type ' + type);
      }
      this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0; this.a1 = a1 / a0; this.a2 = a2 / a0;
      return this;
    }
    tick(x) {
      let y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
      if (y > -1e-25 && y < 1e-25) y = 0;
      this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = y; return y;
    }
    process(b) { for (let i = 0; i < b.length; i++) b[i] = this.tick(b[i]); }
  }

  /* ---------------- radix-2 FFT ---------------- */
  class FFT {
    constructor(n) {
      this.n = n; this.rev = new Uint32Array(n); this.cos = new Float64Array(n / 2); this.sin = new Float64Array(n / 2);
      const bits = Math.log2(n);
      for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); this.rev[i] = r; }
      for (let i = 0; i < n / 2; i++) { this.cos[i] = Math.cos(2 * Math.PI * i / n); this.sin[i] = -Math.sin(2 * Math.PI * i / n); }
    }
    transform(re, im, inverse = false) {
      const n = this.n, rev = this.rev;
      for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = 0, k = 0; j < half; j++, k += step) {
            const wr = this.cos[k], wi = inverse ? -this.sin[k] : this.sin[k];
            const a = i + j, b = a + half;
            const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
            re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          }
        }
      }
      if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
  }

  /* ---------------- Spectral noise suppression (STFT, min-tracking noise estimate) ----------------
     Streaming overlap-add with sqrt-Hann windows, 50% overlap. Latency = N samples. */
  class NoiseSuppressor {
    constructor(fs, N = 512) {
      this.fs = fs; this.N = N; this.hop = N / 2; this.fft = new FFT(N);
      this.win = new Float32Array(N);
      for (let i = 0; i < N; i++) this.win[i] = Math.sqrt(0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
      this.re = new Float64Array(N); this.im = new Float64Array(N);
      const B = N / 2 + 1;
      this.noise = new Float64Array(B); this.ps = new Float64Array(B); this.gPrev = new Float64Array(B);
      this.riseFactor = Math.pow(10, 3 * this.hop / fs / 10); // noise estimate may rise 3 dB/s
      this.amount = 0.6; this.reset();
    }
    get latency() { return this.N; }
    reset() {
      this.inBuf = new Float32Array(this.N); this.acc = new Float32Array(this.N); this.ready = new Float32Array(this.hop);
      this.k = 0; this.frames = 0; this.noise.fill(0); this.ps.fill(0); this.gPrev.fill(1);
      this.inE = 0; this.outE = 0; this.reductionDb = 0;
    }
    process(b) {
      const hop = this.hop, N = this.N;
      for (let i = 0; i < b.length; i++) {
        const x = b[i];
        this.inBuf[hop + this.k] = x;
        b[i] = this.ready[this.k];
        if (++this.k === hop) { this.k = 0; this._frame(); }
      }
    }
    _frame() {
      const N = this.N, hop = this.hop, re = this.re, im = this.im, w = this.win, B = N / 2 + 1;
      let ein = 0;
      for (let i = 0; i < N; i++) { re[i] = this.inBuf[i] * w[i]; im[i] = 0; ein += re[i] * re[i]; }
      this.fft.transform(re, im);
      const a = clamp(this.amount, 0, 1), alpha = 1 + 2 * a, gmin = dbToLin(-30 * a);
      const first = this.frames < 4;
      for (let j = 0; j < B; j++) {
        const P = re[j] * re[j] + im[j] * im[j];
        this.ps[j] = first ? P : 0.7 * this.ps[j] + 0.3 * P;
        if (first) this.noise[j] = this.frames === 0 ? this.ps[j] : Math.min(this.noise[j], this.ps[j]) || this.ps[j];
        else if (this.ps[j] < this.noise[j]) this.noise[j] += 0.25 * (this.ps[j] - this.noise[j]);
        else this.noise[j] *= this.riseFactor;
        let g = 1 - alpha * (this.noise[j] * 1.4) / (P + 1e-20);
        g = Math.max(gmin, g);
        g = 0.35 * this.gPrev[j] + 0.65 * g;
        if (a === 0) g = 1;
        this.gPrev[j] = g;
        re[j] *= g; im[j] *= g;
        if (j > 0 && j < N / 2) { re[N - j] *= g; im[N - j] *= g; }
      }
      this.fft.transform(re, im, true);
      let eout = 0;
      for (let i = 0; i < N; i++) { const y = re[i] * w[i]; this.acc[i] += y; eout += (re[i] * w[i]) ** 2; }
      this.ready.set(this.acc.subarray(0, hop));
      this.acc.copyWithin(0, hop); this.acc.fill(0, N - hop);
      this.inBuf.copyWithin(0, hop);
      this.frames++;
      this.inE = 0.8 * this.inE + 0.2 * ein; this.outE = 0.8 * this.outE + 0.2 * eout;
      this.reductionDb = Math.min(0, powToDb(this.outE + 1e-20) - powToDb(this.inE + 1e-20));
    }
  }

  /* ---------------- Speech detector + non-speech gate ----------------
     Uses periodicity (normalised autocorrelation in the 70-400 Hz pitch range,
     sustained over 30 ms) together with energy above an adaptive noise floor.
     Clicks, paper and fan noise have energy but little sustained pitch, so they
     do not open the gate. A short lookahead delay lets the gate open before the
     first syllable instead of clipping it. */
  class SpeechGate {
    constructor(fs) {
      this.fs = fs;
      this.dec = Math.max(1, Math.round(fs / 12000)); this.dfs = fs / this.dec;
      this.lp1 = new Biquad().set('lowpass', fs, 0.42 * this.dfs, 0.707);
      this.lp2 = new Biquad().set('lowpass', fs, 0.42 * this.dfs, 0.707);
      this.ring = new Float32Array(2048); this.rmask = 2047;
      this.frameLen = Math.round(0.03 * this.dfs); this.hop = Math.round(0.01 * fs);
      this.minLag = Math.floor(this.dfs / 400); this.maxLag = Math.ceil(this.dfs / 70);
      this.lookahead = Math.round(0.03 * fs);
      this.delay = new Float32Array(this.lookahead + 1);
      this.att = Math.exp(-1 / (0.004 * fs)); this.rel = Math.exp(-1 / (0.08 * fs));
      this.sensitivity = 0.5; this.holdMs = 300; this.attenuationDb = -30;
      this.reset();
    }
    reset() {
      this.ring.fill(0); this.w = 0; this.decPhase = 0; this.hopCount = 0; this.t = 0;
      this.floorDb = null; this.hang = 0; this.speech = false; this.active = false; this.voiced = false;
      this.voicedRun = 0; this.lastVoicedT = -10; this.periodicity = 0; this.f0 = 0; this.levelDb = -120;
      this.g = 1; this.delay.fill(0); this.dw = 0; this.lp1.reset(); this.lp2.reset(); this.frames = 0;
    }
    process(b, applyGate) {
      const target0 = dbToLin(this.attenuationDb), L = this.lookahead;
      if (!this.mask || this.mask.length !== b.length) this.mask = new Uint8Array(b.length);
      for (let i = 0; i < b.length; i++) {
        const x = b[i];
        const y = this.lp2.tick(this.lp1.tick(x));
        if (++this.decPhase >= this.dec) { this.decPhase = 0; this.ring[this.w] = y; this.w = (this.w + 1) & this.rmask; }
        if (++this.hopCount >= this.hop) { this.hopCount = 0; this._analyze(); }
        this.mask[i] = this.speech ? 1 : 0;
        if (applyGate) {
          const target = this.speech ? 1 : target0;
          this.g = target + (this.g - target) * (target > this.g ? this.att : this.rel);
          this.delay[this.dw] = x;
          this.dw = this.dw === L ? 0 : this.dw + 1;
          b[i] = this.delay[this.dw] * this.g; // sample from L samples ago
        }
      }
    }
    _analyze() {
      const L = this.frameLen, start = (this.w - L) & this.rmask, r = this.ring, m = this.rmask;
      const f = this._f || (this._f = new Float32Array(L));
      let e = 0;
      for (let i = 0; i < L; i++) { const v = r[(start + i) & m]; f[i] = v; e += v * v; }
      let lvl = powToDb(e / L);
      if (!Number.isFinite(lvl)) lvl = -120;
      this.levelDb = lvl; this.t += this.hop / this.fs; this.frames++;
      if (this.floorDb === null) this.floorDb = lvl;
      if (lvl < this.floorDb) this.floorDb += (lvl - this.floorDb) * 0.3;
      else this.floorDb += Math.min(lvl - this.floorDb, 2 * this.hop / this.fs);
      // normalised autocorrelation over the pitch range
      let best = 0, bestLag = 0;
      if (lvl > -75) {
        for (let lag = this.minLag; lag <= this.maxLag; lag++) {
          let s = 0, e1 = 0, e2 = 0;
          for (let i = 0; i + lag < L; i++) { const a = f[i], c = f[i + lag]; s += a * c; e1 += a * a; e2 += c * c; }
          const rr = s / Math.sqrt(e1 * e2 + 1e-20);
          if (rr > best) { best = rr; bestLag = lag; }
        }
      }
      const sens = clamp(this.sensitivity, 0, 1);
      const thrE = 18 - 14 * sens, thrP = 0.75 - 0.4 * sens, absMin = -50 - 15 * sens;
      const energetic = lvl > this.floorDb + thrE && lvl > absMin;
      const periodicNow = best > thrP && energetic;
      const f0 = bestLag ? this.dfs / bestLag : 0;
      if (periodicNow && this.voicedRun > 0 && this.f0 > 0 && Math.abs(Math.log2(f0 / this.f0)) > 0.25) this.voicedRun = 1;
      else this.voicedRun = periodicNow ? this.voicedRun + 1 : 0;
      this.periodicity = best; if (periodicNow) this.f0 = f0;
      this.voiced = this.voicedRun >= 3; // sustained for 30 ms
      if (this.voiced) this.lastVoicedT = this.t;
      this.active = energetic && (this.voiced || this.t - this.lastVoicedT < 0.15);
      if (this.active) this.hang = this.holdMs / 1000; else this.hang -= this.hop / this.fs;
      this.speech = this.active || this.hang > 0;
    }
  }

  /* ---------------- Energy-only gate (used only to show what a plain gate would do) ---------------- */
  function energyOnlyDecisions(data, fs, thrDb = 10) {
    const hop = Math.round(0.01 * fs), out = [];
    let floor = null;
    for (let i = 0; i + hop <= data.length; i += hop) {
      const l = linToDb(rms(data, i, i + hop));
      if (floor === null) floor = l;
      if (l < floor) floor += (l - floor) * 0.3; else floor += Math.min(l - floor, 0.02);
      out.push({ t: i / fs, on: l > floor + thrDb && l > -62 });
    }
    return out;
  }

  /* ---------------- De-esser: split-band, reduces only the sibilant band ---------------- */
  class DeEsser {
    constructor(fs) { this.fs = fs; this.bp = new Biquad(); this.det = new Biquad(); this.att = Math.exp(-1 / (0.001 * fs)); this.rel = Math.exp(-1 / (0.06 * fs)); this.set(6500, -32, 6); this.reset(); }
    set(freq, thr, maxDb) { this.freq = freq; this.thr = thr; this.maxDb = maxDb; this.bp.set('bandpass', this.fs, freq, 1.4); this.det.set('bandpass', this.fs, freq, 1.4); }
    reset() { this.bp.reset(); this.det.reset(); this.env = 0; this.grDb = 0; }
    process(b) {
      let minG = 0;
      for (let i = 0; i < b.length; i++) {
        const x = b[i], band = this.bp.tick(x), d = this.det.tick(x), a = d < 0 ? -d : d;
        this.env = a > this.env ? a + (this.env - a) * this.att : a + (this.env - a) * this.rel;
        const over = linToDb(this.env * 1.414) - this.thr;
        const red = over > 0 ? Math.min(this.maxDb, over * 0.75) : 0; // 4:1 above threshold
        const g = dbToLin(-red);
        b[i] = x - (1 - g) * band;
        if (-red < minG) minG = -red;
      }
      this.grDb = minG;
    }
  }

  /* ---------------- Compressor (soft knee, feed-forward) ---------------- */
  class Compressor {
    constructor(fs) { this.fs = fs; this.knee = 6; this.set(-26, 2.5, 8, 160); this.reset(); }
    set(thr, ratio, attMs, relMs) {
      this.thr = thr; this.ratio = Math.max(1, ratio);
      this.aA = Math.exp(-1 / (Math.max(0.1, attMs) / 1000 * this.fs)); this.aR = Math.exp(-1 / (Math.max(1, relMs) / 1000 * this.fs));
    }
    reset() { this.env = -120; this.gr = 0; this.grDb = 0; }
    gainFor(lvl) {
      const over = lvl - this.thr, k = this.knee, s = 1 / this.ratio - 1;
      if (over <= -k / 2) return 0;
      if (over < k / 2) return s * (over + k / 2) ** 2 / (2 * k);
      return s * over;
    }
    process(b) {
      let minG = 0;
      for (let i = 0; i < b.length; i++) {
        const lvl = linToDb(b[i]);
        this.env = lvl > this.env ? lvl + (this.env - lvl) * this.aA : lvl + (this.env - lvl) * this.aR;
        const g = this.gainFor(this.env);
        b[i] *= dbToLin(g);
        if (g < minG) minG = g;
      }
      this.grDb = minG;
    }
  }

  /* ---------------- K-weighting + loudness ---------------- */
  function kFilters(fs) { return [new Biquad().set('highshelf', fs, 1681.97, 0.7071, 4.0), new Biquad().set('highpass', fs, 38.13, 0.5)]; }
  class LoudnessMeter {
    constructor(fs) { this.fs = fs; this.k = kFilters(fs); this.ms = 0; this.started = false; this.buf = null; }
    reset() { this.k.forEach(f => f.reset()); this.ms = 0; this.started = false; }
    push(b) {
      if (!this.buf || this.buf.length < b.length) this.buf = new Float32Array(b.length);
      let s = 0;
      for (let i = 0; i < b.length; i++) { const y = this.k[1].tick(this.k[0].tick(b[i])); s += y * y; }
      const dt = b.length / this.fs, a = 1 - Math.exp(-dt / 0.4), cur = s / b.length;
      if (!this.started) { this.started = cur > 1e-12; this.ms = cur; }
      else this.ms += (cur - this.ms) * a;
      return this.momentary;
    }
    get momentary() { return -0.691 + powToDb(this.ms); }
  }
  /** Speech-gated loudness normaliser: slowly steers the gain so speech sits at the target LUFS.
      Works on fixed 256-sample frames internally, so the result does not depend on the host chunk size. */
  class LoudnessNormalizer {
    constructor(fs) { this.fs = fs; this.k = kFilters(fs); this.F = 256; this.target = -16; this.smooth = Math.exp(-1 / (0.02 * fs)); this.reset(); }
    reset() { this.k.forEach(f => f.reset()); this.ms = 0; this.started = false; this.acc = 0; this.n = 0; this.sp = 0; this.est = null; this.gainDb = 0; this.cur = 1; this.speechT = 0; }
    process(b, mask) {
      const F = this.F, dt = F / this.fs, a = 1 - Math.exp(-dt / 0.4);
      for (let i = 0; i < b.length; i++) {
        const y = this.k[1].tick(this.k[0].tick(b[i]));
        this.acc += y * y; this.sp += mask ? mask[i] : 1;
        if (++this.n === F) {
          const p = this.acc / F;
          if (!this.started) { this.started = p > 1e-12; this.ms = p; } else this.ms += (p - this.ms) * a;
          const m = -0.691 + powToDb(this.ms);
          if (this.sp > F / 2 && m > -70) {
            this.speechT += dt;
            const tau = this.speechT < 1.5 ? 0.5 : 2.5;
            this.est = this.est === null ? m : this.est + (m - this.est) * Math.min(1, dt / tau);
          }
          if (this.est !== null) this.gainDb += (clamp(this.target - this.est, -12, 24) - this.gainDb) * Math.min(1, dt / 0.8);
          this.acc = 0; this.n = 0; this.sp = 0;
        }
        const tgt = dbToLin(this.gainDb);
        this.cur = tgt + (this.cur - tgt) * this.smooth;
        b[i] *= this.cur;
      }
    }
  }
  function integratedLufs(data, fs) {
    const k = kFilters(fs), y = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) y[i] = k[1].tick(k[0].tick(data[i]));
    const W = Math.round(0.4 * fs), H = Math.round(0.1 * fs), blocks = [];
    for (let i = 0; i + W <= y.length; i += H) { let s = 0; for (let j = i; j < i + W; j++) s += y[j] * y[j]; blocks.push(s / W); }
    const L = (p) => -0.691 + powToDb(p);
    const abs = blocks.filter(p => L(p) > -70);
    if (!abs.length) return { integrated: -Infinity, blocks: blocks.map(L) };
    const mean1 = abs.reduce((a, b) => a + b, 0) / abs.length;
    const rel = abs.filter(p => L(p) > L(mean1) - 10);
    const mean2 = rel.reduce((a, b) => a + b, 0) / rel.length;
    return { integrated: L(mean2), blocks: blocks.map(L) };
  }

  /* ---------------- Pitch shifter (two-tap crossfaded delay line) ---------------- */
  class PitchShifter {
    constructor(fs, windowMs = 40) {
      this.fs = fs; this.W = Math.round(windowMs / 1000 * fs);
      let size = 1; while (size < this.W * 2 + 8) size <<= 1;
      this.mask = size - 1; this.buf = new Float32Array(size); this.setSemitones(0); this.reset();
    }
    reset() { this.buf.fill(0); this.w = 0; this.p = 0; }
    setSemitones(st) { this.semitones = st; this.ratio = Math.pow(2, st / 12); }
    get latency() { return Math.round(this.W / 2); }
    _read(d) { const pos = this.w - d, i = Math.floor(pos), fr = pos - i, a = this.buf[i & this.mask], b = this.buf[(i + 1) & this.mask]; return a + (b - a) * fr; }
    process(b) {
      const W = this.W, dp = (1 - this.ratio) / W; let p = this.p;
      for (let i = 0; i < b.length; i++) {
        this.buf[this.w] = b[i];
        let p2 = p + 0.5; if (p2 >= 1) p2 -= 1;
        b[i] = (1 - Math.abs(2 * p - 1)) * this._read(1 + p * W) + (1 - Math.abs(2 * p2 - 1)) * this._read(1 + p2 * W);
        p += dp; p -= Math.floor(p); this.w = (this.w + 1) & this.mask;
      }
      this.p = p;
    }
  }

  /* ---------------- Limiter (instant attack: output never exceeds the ceiling) ---------------- */
  class Limiter {
    constructor(fs) { this.rel = Math.exp(-1 / (0.06 * fs)); this.ceilingDb = -1; this.reset(); }
    reset() { this.g = 1; this.grDb = 0; }
    process(b) {
      const c = dbToLin(this.ceilingDb); let g = this.g, minG = 1;
      for (let i = 0; i < b.length; i++) {
        const a = Math.abs(b[i]), need = a > c ? c / a : 1;
        g = Math.min(need, 1 - (1 - g) * this.rel);
        let y = b[i] * g; if (y > c) y = c; else if (y < -c) y = -c;
        b[i] = y; if (g < minG) minG = g;
      }
      this.g = g; this.grDb = linToDb(minG);
    }
  }

  /* ---------------- Stats ---------------- */
  class Stats {
    constructor(cap = 3000) { this.cap = cap; this.arr = []; }
    push(v) { this.arr.push(v); if (this.arr.length > this.cap) this.arr.shift(); }
    clear() { this.arr = []; }
    get count() { return this.arr.length; }
    pct(q) { if (!this.arr.length) return NaN; const s = this.arr.slice().sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1) + 0.5))]; }
  }

  /* ---------------- The pipeline ----------------
     Input gain -> Noise suppression -> VAD (+ non-speech mute) -> High-pass -> Mud cut -> Warmth
     -> Presence -> Air -> Pitch -> De-esser -> Compressor -> Makeup -> Loudness -> Output gain -> Limiter */
  const STAGES = [
    ['input', 'Input gain'], ['ns', 'Noise suppression'], ['vad', 'Speech detection'], ['hpf', 'High-pass'],
    ['eq', 'Tone EQ'], ['pitch', 'Pitch'], ['creative', 'Creative effects'], ['deess', 'De-esser'], ['comp', 'Compressor'],
    ['loud', 'Loudness'], ['out', 'Output + limiter'],
  ];

  class EffectsChain {
    constructor(fs) {
      this.fs = fs; this.phase = 0; this.position = 0;
      this.delay = new Float32Array(Math.floor(fs) + 1);
      this.modules = {
        metallic: (b, p) => {
          if (!p.metallicMix) return;
          const step = 2 * Math.PI * p.metallicHz / fs;
          for (let i = 0; i < b.length; i++) {
            b[i] *= 1 - p.metallicMix + p.metallicMix * Math.sin(this.phase);
            this.phase = (this.phase + step) % (2 * Math.PI);
          }
        },
        distortion: (b, p) => {
          if (!p.distortionDrive) return;
          const gain = 1 + p.distortionDrive * 15;
          for (let i = 0; i < b.length; i++) b[i] = Math.tanh(b[i] * gain) / gain;
        },
        echo: (b, p) => {
          if (!p.echoMix) { this.delay.fill(0); return; }
          const delay = Math.max(1, Math.round(fs * p.echoMs / 1000)), size = this.delay.length;
          for (let i = 0; i < b.length; i++) {
            const wet = this.delay[(this.position - delay + size) % size], dry = b[i];
            this.delay[this.position] = dry + wet * p.echoFeedback;
            b[i] = dry * (1 - p.echoMix) + wet * p.echoMix;
            this.position = (this.position + 1) % size;
          }
        },
      };
    }
    process(b, p) {
      const order = p.echoBeforeTexture ? ['echo', 'metallic', 'distortion'] : ['metallic', 'distortion', 'echo'];
      for (const name of order) this.modules[name](b, p);
    }
  }

  class Pipeline {
    constructor(fs, params) {
      this.fs = fs;
      this.ns = new NoiseSuppressor(fs); this.gate = new SpeechGate(fs);
      this.hpf = new Biquad(); this.mud = new Biquad(); this.warm = new Biquad(); this.pres = new Biquad(); this.air = new Biquad();
      this.shifter = new PitchShifter(fs); this.deess = new DeEsser(fs); this.comp = new Compressor(fs);
      this.effects = new EffectsChain(fs);
      this.loud = new LoudnessNormalizer(fs); this.lim = new Limiter(fs);
      this.inMeter = new LoudnessMeter(fs); this.outMeter = new LoudnessMeter(fs);
      this.stageUs = Object.fromEntries(STAGES.map(s => [s[0], 0])); this.timeStages = true;
      this.p = { ...root.VoiceProfile.DEFAULTS };
      this.configure(params || {});
    }
    configure(params) {
      const prev = this.p, p = this.p = { ...this.p, ...params }, fs = this.fs;
      this.ns.amount = p.nsAmount;
      this.gate.sensitivity = p.vadSensitivity; this.gate.holdMs = p.vadHoldMs; this.gate.attenuationDb = p.vadAttenuationDb;
      this.hpf.set('highpass', fs, p.hpfHz, 0.707);
      this.mud.set('peaking', fs, 350, 1.0, p.mudDb);
      this.warm.set('peaking', fs, 180, 0.8, p.warmthDb);
      this.pres.set('peaking', fs, 4000, 0.9, p.presenceDb);
      this.air.set('highshelf', fs, 10000, 0.707, p.airDb);
      if (p.pitchSemitones !== prev.pitchSemitones) { if (Math.abs(p.pitchSemitones) < 0.01) this.shifter.reset(); this.shifter.setSemitones(p.pitchSemitones); }
      this.deess.set(p.deEssFreqHz, p.deEssThresholdDb, p.deEssMaxDb);
      this.comp.set(p.compThresholdDb, p.compRatio, p.compAttackMs, p.compReleaseMs);
      this.loud.target = p.targetLufs; this.lim.ceilingDb = p.limiterCeilingDb;
      return this;
    }
    get pitchActive() { return Math.abs(this.p.pitchSemitones) >= 0.01; }
    /** Delay between input and output in samples, for aligning Original vs Enhanced. */
    get latencySamples() { return (this.p.nsEnabled ? this.ns.latency : 0) + (this.p.vadEnabled ? this.gate.lookahead : 0) + (this.pitchActive ? this.shifter.latency : 0); }
    _t(id, t0) { const t1 = now(); if (this.timeStages) this.stageUs[id] += ((t1 - t0) * 1000 - this.stageUs[id]) * 0.05; return t1; }

    process(b, bypass = false) {
      const p = this.p;
      const inPk = peak(b), inM = this.inMeter.push(b);
      if (bypass) {
        const g = this.gate; g.process(b.slice(), false);
        return { bypass: true, speech: g.speech, voiced: g.voiced, f0: g.f0, periodicity: g.periodicity, inDb: linToDb(inPk), inLufs: inM, outDb: linToDb(inPk), outLufs: this.outMeter.push(b), nsDb: 0, deEssDb: 0, compDb: 0, limDb: 0, loudDb: 0 };
      }
      let t = now();
      if (p.inputGainDb !== 0) { const g = dbToLin(p.inputGainDb); for (let i = 0; i < b.length; i++) b[i] *= g; }
      t = this._t('input', t);
      if (p.nsEnabled) this.ns.process(b);
      t = this._t('ns', t);
      this.gate.process(b, p.vadEnabled);
      const speech = this.gate.speech;
      t = this._t('vad', t);
      if (p.hpfEnabled) this.hpf.process(b);
      t = this._t('hpf', t);
      if (p.mudDb !== 0) this.mud.process(b);
      if (p.warmthDb !== 0) this.warm.process(b);
      if (p.presenceDb !== 0) this.pres.process(b);
      if (p.airDb !== 0) this.air.process(b);
      t = this._t('eq', t);
      if (this.pitchActive) this.shifter.process(b);
      t = this._t('pitch', t);
      this.effects.process(b, p);
      t = this._t('creative', t);
      if (p.deEssEnabled) this.deess.process(b); else this.deess.grDb = 0;
      t = this._t('deess', t);
      if (p.compEnabled) {
        this.comp.process(b);
        if (p.makeupDb !== 0) { const g = dbToLin(p.makeupDb); for (let i = 0; i < b.length; i++) b[i] *= g; }
      } else this.comp.grDb = 0;
      t = this._t('comp', t);
      if (p.loudEnabled) this.loud.process(b, this.gate.mask);
      t = this._t('loud', t);
      if (p.outputGainDb !== 0) { const g = dbToLin(p.outputGainDb); for (let i = 0; i < b.length; i++) b[i] *= g; }
      if (p.limiterEnabled) this.lim.process(b); else this.lim.grDb = 0;
      this._t('out', t);
      return {
        bypass: false, speech, voiced: this.gate.voiced, f0: this.gate.f0, periodicity: this.gate.periodicity,
        inDb: linToDb(inPk), inLufs: inM, outDb: linToDb(peak(b)), outLufs: this.outMeter.push(b),
        nsDb: p.nsEnabled ? this.ns.reductionDb : 0, gateDb: p.vadEnabled ? linToDb(this.gate.g) : 0,
        deEssDb: this.deess.grDb, compDb: this.comp.grDb, limDb: this.lim.grDb, loudDb: p.loudEnabled ? this.loud.gainDb : 0,
      };
    }
  }

  /** Render a whole buffer offline, aligned with the input (latency removed). */
  function renderOffline(data, fs, params, chunk = 1024) {
    const pl = new Pipeline(fs, params); pl.timeStages = false;
    const lat = pl.latencySamples, total = data.length + lat;
    const out = new Float32Array(total), infos = [], blk = new Float32Array(chunk);
    for (let i = 0; i < total; i += chunk) {
      const n = Math.min(chunk, total - i), b = n === chunk ? blk : new Float32Array(n);
      b.fill(0);
      if (i < data.length) b.set(data.subarray(i, Math.min(data.length, i + n)));
      const info = pl.process(b);
      out.set(b, i);
      if (i < data.length) infos.push(info);
    }
    return { data: out.slice(lat, lat + data.length), infos, chunk, latency: lat };
  }

  /* ---------------- Synthetic sample voice with labelled noise events ---------------- */
  const _cache = new Map();
  function makeSampleVoice(fs, opts = {}) {
    const key = fs + JSON.stringify(opts);
    if (_cache.has(key)) return _cache.get(key);
    const r = _makeSample(fs, opts); if (_cache.size > 10) _cache.clear(); _cache.set(key, r); return r;
  }
  function _makeSample(fs, opts) {
    const seconds = opts.seconds || 14;
    const events = opts.events || [
      { type: 'speech', s: 0.6, e: 2.8, db: -24 }, { type: 'keys', s: 3.1, e: 3.9, db: -22 },
      { type: 'speech', s: 4.3, e: 6.3, db: -27 }, { type: 'paper', s: 6.7, e: 7.25, db: -36 },
      { type: 'speech', s: 7.7, e: 9.6, db: -38 }, { type: 'keys', s: 10.0, e: 10.8, db: -22 },
      { type: 'speech', s: 11.2, e: 13.4, db: -26 },
    ];
    const n = Math.round(fs * seconds), x = new Float32Array(n);
    let seed = opts.seed || 1234567;
    const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };
    const vowels = [[730, 1090, 2440], [270, 2290, 3010], [530, 1840, 2480], [300, 870, 2240], [660, 1720, 2410]];
    const f0base = opts.f0 || 135;
    events.forEach((ev, ei) => {
      const i0 = Math.round(ev.s * fs), i1 = Math.min(n, Math.round(ev.e * fs));
      if (i1 - i0 < fs * 0.05) return;
      const seg = new Float32Array(i1 - i0);
      if (ev.type === 'speech') {
        const ph = new Float64Array(64), sib = new Biquad().set('bandpass', fs, 6500, 1.2);
        for (let i = 0; i < seg.length; i++) {
          const t = i / fs, T = ev.s + t;
          const f0 = f0base + 20 * Math.sin(2 * Math.PI * 0.7 * T) - 10 * (t / (ev.e - ev.s)) + 4 * ei;
          const sp = t * 4.2, si = Math.floor(sp), fr = sp - si;
          const env = Math.pow(Math.max(0, Math.sin(Math.PI * fr)), 0.6);
          const F = vowels[(si + ei) % vowels.length], K = Math.min(60, Math.floor(5000 / f0));
          let v = 0;
          for (let k = 1; k <= K; k++) {
            ph[k] += 2 * Math.PI * k * f0 / fs; if (ph[k] > 1e6) ph[k] %= 2 * Math.PI;
            const fk = k * f0; let a = 0;
            for (let j = 0; j < 3; j++) a += Math.exp(-(((fk - F[j]) / (80 + 40 * j)) ** 2)) * (1 - 0.3 * j);
            v += (a * 0.9 + 0.12 / k) * Math.sin(ph[k]);
          }
          // a sibilant "s" at the start of every third syllable
          const s = sib.tick(rnd());
          const sEnv = si % 3 === 1 && fr < 0.3 ? Math.sin(Math.PI * fr / 0.3) : 0;
          seg[i] = v * env + s * sEnv * 9;
        }
      } else if (ev.type === 'keys') {
        let next = 0;
        for (let i = 0; i < seg.length; i++) {
          if (i >= next) { next = i + Math.round((0.11 + 0.08 * Math.abs(rnd())) * fs); ev._last = i; }
          const dt = (i - ev._last) / fs;
          seg[i] = rnd() * Math.exp(-dt / 0.0025);
        }
      } else if (ev.type === 'paper') {
        const bp = new Biquad().set('bandpass', fs, 3000, 0.6); let am = 0;
        for (let i = 0; i < seg.length; i++) { if (i % 200 === 0) am = Math.abs(rnd()); seg[i] = bp.tick(rnd()) * am * Math.sin(Math.PI * i / seg.length); }
      }
      const g = dbToLin(ev.db) / (rms(seg) || 1);
      for (let i = 0; i < seg.length; i++) x[i0 + i] += seg[i] * g;
    });
    // fan: low-passed noise plus 120 Hz hum
    const lp = new Biquad().set('lowpass', fs, 900, 0.7), fan = new Float32Array(n);
    for (let i = 0; i < n; i++) fan[i] = lp.tick(rnd()) + 0.15 * Math.sin(2 * Math.PI * 120 * i / fs);
    const fg = dbToLin(opts.noiseDb ?? -52) / rms(fan);
    for (let i = 0; i < n; i++) x[i] += fan[i] * fg;
    const at = (t) => { const e = events.find(ev => t >= ev.s && t < ev.e); return e ? e.type : 'none'; };
    return { data: x, fs, seconds, events, at };
  }

  root.DSP = {
    dbToLin, linToDb, powToDb, clamp, rms, peak, now, Biquad, FFT, NoiseSuppressor, SpeechGate, energyOnlyDecisions,
    DeEsser, Compressor, LoudnessMeter, LoudnessNormalizer, integratedLufs, kFilters, PitchShifter, Limiter, Stats,
    STAGES, EffectsChain, Pipeline, renderOffline, makeSampleVoice,
  };
})(typeof window !== 'undefined' ? window : globalThis);
