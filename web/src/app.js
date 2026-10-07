(function () {
  'use strict';
  const D = window.DSP, VP = window.VoiceProfile, VA = window.VoiceAnalysis, AI = window.VoiceAI;
  const $ = (s) => document.querySelector(s), $$ = (s) => Array.from(document.querySelectorAll(s));
  const f = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  const sg = (v, d = 1) => (Number.isFinite(v) ? (v > 0 ? '+' : '') + v.toFixed(d) : '–');
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  class UserError extends Error {}

  /* ============================ persistence ============================ */
  const LS = {
    get(k) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null; } catch (_) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (_) { return false; } },
  };

  /* ============================ state ============================ */
  const undoStack = [];
  const store = new VP.ProfileStore(LS.get('ve.working.v1') || undefined);
  const S = {
    view: 'simple', active: LS.get('ve.active.v1'), reference: LS.get('ve.reference.v1') || { ...VA.DEFAULT_REFERENCE },
    rec: null, analysis: null, chat: [], history: [], busy: false,
    caps: { sample: null, downloads: null }, interpreter: AI.OfflineInterpreter, engine: 'browser', server: null,
  };
  window.__ve = { S, store }; // for automated checks

  /* ============================ audio context ============================ */
  let ctx = null;
  async function audio() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new UserError('This browser does not support Web Audio, so audio cannot be processed here.');
      ctx = new AC({ latencyHint: 'interactive' });
    }
    if (ctx.state !== 'running') await ctx.resume();
    return ctx;
  }
  function micError(e) {
    const n = e && e.name;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return 'This browser or frame does not allow microphone access. Use the sample voice or upload a recording instead.';
    if (n === 'NotAllowedError' || n === 'SecurityError') return 'Microphone access was blocked. Allow the microphone for this page in your browser settings, or use the sample voice or upload a recording.';
    if (n === 'NotFoundError' || n === 'OverconstrainedError') return 'No microphone was found. Connect one, or use the sample voice or upload a recording.';
    if (n === 'NotSupportedError' || n === 'TypeError') return 'Microphone capture is not supported in this browser or frame. Use the sample voice or upload a recording.';
    if (n === 'NotReadableError') return 'The microphone is busy in another app. Close it there and try again.';
    return 'The microphone could not be opened (' + (n || 'unknown error') + ').';
  }
  async function openMic() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new UserError(micError(null));
    try { return await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } }); }
    catch (e) { throw new UserError(micError(e)); }
  }

  /* ============================ recorder ============================ */
  const recorder = {
    on: false, pending: false, stream: null, node: null, parts: [], frames: 0, max: 0, t0: 0, timer: null,
    async start() {
      if (this.on) return;
      const c = await audio();
      live.stop(); player.stop();
      this.stream = await openMic();
      const src = c.createMediaStreamSource(this.stream), sp = c.createScriptProcessor(4096, 1, 1), mute = c.createGain();
      mute.gain.value = 0; this.parts = []; this.frames = 0; this.max = c.sampleRate * 20;
      sp.onaudioprocess = (e) => {
        if (!this.on) return;
        const d = e.inputBuffer.getChannelData(0); this.parts.push(d.slice()); this.frames += d.length;
        if (this.frames >= this.max) this.stop();
      };
      src.connect(sp); sp.connect(mute); mute.connect(c.destination);
      Object.assign(this, { on: true, node: sp, src, mute, t0: performance.now() });
      this.timer = setInterval(() => ui.recTick((performance.now() - this.t0) / 1000), 200);
      ui.recState(true);
    },
    stop() {
      if (!this.on) return;
      this.on = false; clearInterval(this.timer);
      try { this.node.disconnect(); this.src.disconnect(); this.mute.disconnect(); } catch (_) {}
      this.stream.getTracks().forEach(t => t.stop());
      const data = new Float32Array(this.frames); let o = 0; for (const p of this.parts) { data.set(p, o); o += p.length; }
      ui.recState(false);
      if (data.length < ctx.sampleRate * 1) { ui.recMsg('That recording was shorter than a second. Record a few sentences.', 'warn'); return; }
      setRecording(data, ctx.sampleRate, 'Your recording');
    },
  };

  async function setRecording(data, fs, label) {
    player.stop();
    if (recorder.on) recorder.stop();
    live.stop();
    preview.orig = preview.enh = preview.enhData = null;
    $('#playBtn').disabled = $('#mxPlay').disabled = $('#downloadAudio').disabled = true;
    S.rec = { data, fs, label };
    const a = VA.analyzeVoice(data, fs);
    S.analysis = a;
    ui.recMsg(`${label}: ${f(data.length / fs, 1)} s loaded.` + (a.ok ? '' : ' ' + a.reason), a.ok ? '' : 'warn');
    $('#sampleBadge').textContent = f(data.length / fs, 1) + ' s · Ready';
    $('#sampleBadge').classList.add('ready');
    $('#waveEmpty').hidden = true;
    ui.renderSummary(); ui.renderAnalysis();
    await preview.render(true);
  }

  /* ============================ preview (offline render of the same engine) ============================ */
  const preview = {
    orig: null, enh: null, infos: null, chunk: 1024, version: -1, timer: null, renders: 0, request: 0,
    async render(immediate) {
      clearTimeout(this.timer);
      const request = ++this.request;
      if (!S.rec) { ui.previewMsg('Load a recording to hear your sound.'); return; }
      $('#downloadAudio').disabled = true;
      $('#previewBadge').textContent = 'Updating…';
      $('#previewBadge').classList.remove('ready');
      const go = async () => {
        const rec = S.rec, params = store.get(), version = store.version, engine = S.engine;
        const c = await audio().catch(() => null);
        const t0 = performance.now();
        let r, where = 'in the browser', fallback = '';
        try {
          if (engine === 'server') {
            try { r = await window.VoiceServer.render(rec.data, rec.fs, params, this.chunk); where = 'on the server (' + r.backend + ')'; }
            catch (e) { fallback = 'Server unavailable; using local processing. '; }
          }
          if (!r) r = D.renderOffline(rec.data, rec.fs, params, this.chunk);
          if (request !== this.request || version !== store.version || rec !== S.rec || engine !== S.engine) return;
          const ms = performance.now() - t0;
          if (c) {
            const ob = c.createBuffer(1, rec.data.length, rec.fs); ob.getChannelData(0).set(rec.data);
            const eb = c.createBuffer(1, r.data.length, rec.fs); eb.getChannelData(0).set(r.data);
            this.orig = ob; this.enh = eb;
          }
          this.enhData = r.data; this.infos = r.infos; this.version = version; this.renders++;
          ui.previewMsg(`${fallback}Preview ready · ${f(rec.data.length / rec.fs, 1)} s · processed ${where} in ${f(ms, 0)} ms.`);
          $('#previewBadge').textContent = 'Ready to listen';
          $('#previewBadge').classList.add('ready');
          $('#playBtn').disabled = $('#mxPlay').disabled = !this.enh;
          $('#downloadAudio').disabled = false;
          player.swapBuffers(); drawBar();
        } catch (e) {
          if (request !== this.request) return;
          $('#previewBadge').textContent = 'Could not update';
          ui.previewMsg('Preview failed: ' + e.message + '. Try loading your recording again.');
        }
      };
      if (immediate) await go(); else { ui.previewMsg('Updating preview…'); this.timer = setTimeout(go, 180); }
    },
  };

  /* Both versions play in sync through two gains; Original/Enhanced crossfades without restarting. */
  const player = {
    playing: false, which: 'enhanced', startAt: 0, offset: 0, nodes: null,
    get duration() { return preview.orig ? preview.orig.duration : 0; },
    get position() { return this.playing ? Math.min(this.duration, ctx.currentTime - this.startAt) : this.offset; },
    async play(from) {
      if (!preview.enh) return;
      const c = await audio(); live.stop();
      this.stopNodes();
      let pos = from ?? this.offset;
      if (!(pos >= 0) || pos >= this.duration - 0.05) pos = 0;
      const mk = (buf, on) => { const s = c.createBufferSource(), g = c.createGain(); s.buffer = buf; g.gain.value = on ? 1 : 0; s.connect(g); g.connect(c.destination); return { s, g }; };
      const o = mk(preview.orig, this.which === 'original'), e = mk(preview.enh, this.which === 'enhanced');
      const t = c.currentTime + 0.03;
      o.s.start(t, pos); e.s.start(t, pos);
      e.s.onended = () => { if (this.nodes && this.nodes.e === e) { this.playing = false; this.offset = 0; this.nodes = null; ui.playState(); } };
      this.nodes = { o, e }; this.startAt = t - pos; this.playing = true; ui.playState();
    },
    pause() { if (!this.playing) return; this.offset = this.position; this.stopNodes(); this.playing = false; ui.playState(); },
    stop() { this.stopNodes(); this.playing = false; this.offset = 0; ui.playState(); },
    stopNodes() { if (this.nodes) { const n = this.nodes; this.nodes = null; for (const x of [n.o, n.e]) { try { x.s.onended = null; x.s.stop(); } catch (_) {} } } },
    setWhich(w) {
      this.which = w;
      if (this.nodes && ctx) {
        const t = ctx.currentTime;
        this.nodes.o.g.gain.setTargetAtTime(w === 'original' ? 1 : 0, t, 0.01);
        this.nodes.e.g.gain.setTargetAtTime(w === 'enhanced' ? 1 : 0, t, 0.01);
      }
      $$('[data-ab]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.ab === w)));
    },
    swapBuffers() { if (this.playing) { const p = this.position; this.play(p); } },
    info() {
      if (!this.playing || !preview.infos) return null;
      const i = Math.floor(this.position * S.rec.fs / preview.chunk);
      const inf = preview.infos[Math.min(preview.infos.length - 1, Math.max(0, i))];
      return inf ? (this.which === 'original' ? { ...inf, bypass: true, outDb: inf.inDb, outLufs: inf.inLufs, nsDb: 0, deEssDb: 0, compDb: 0, limDb: 0 } : inf) : null;
    },
  };

  /* ============================ live engine ============================ */
  const live = {
    on: false, pending: false, which: 'enhanced', stream: null, pl: null, remote: null, last: null, stats: new D.Stats(2000), blocks: 0, over: 0, chunk: 1024, vol: 0, profileName: '', mode: 'browser',
    async start() {
      if (this.on) return;
      const c = await audio(); player.pause(); if (recorder.on) recorder.stop();
      this.stream = await openMic();
      const params = S.active ? S.active.params : store.get();
      this.profileName = S.active ? S.active.name : 'unsaved settings';
      this.mode = S.engine;
      this.stats.clear(); this.blocks = 0; this.over = 0; this.last = null;
      if (this.mode === 'server') {
        try { this.remote = await window.VoiceServer.openStream(c.sampleRate, params, this.bypassed()).opened; }
        catch (e) { this.stream.getTracks().forEach(t => t.stop()); if (this.remote) this.remote.close(); this.remote = null; throw new UserError(e.message + ' Switch the engine to Browser in the Mixer to run without the server.'); }
        this.remote.onInfo = (m) => { this.last = m; };
        this.remote.onError = (m) => { this.stop(); ui.liveMsg('Server: ' + m + ' Try restarting live monitoring.', 'bad'); };
        this._byp = this.bypassed();
      } else this.pl = new D.Pipeline(c.sampleRate, params);
      const src = c.createMediaStreamSource(this.stream), sp = c.createScriptProcessor(this.chunk, 1, 1), g = c.createGain();
      g.gain.value = this.vol;
      sp.onaudioprocess = (e) => {
        const inp = e.inputBuffer.getChannelData(0), out = e.outputBuffer.getChannelData(0), w = inp.slice();
        const bypass = this.bypassed();
        if (this.mode === 'server') {
          if (bypass !== this._byp) { this._byp = bypass; this.remote.setBypass(bypass); }
          this.remote.send(w);
          const back = this.remote.take(inp.length);
          if (back) out.set(back); else out.fill(0);
          this.blocks++;
          return;
        }
        const t0 = performance.now();
        this.last = this.pl.process(w, bypass);
        const dt = performance.now() - t0; this.stats.push(dt); this.blocks++; if (dt > inp.length / c.sampleRate * 1000) this.over++;
        out.set(w);
      };
      src.connect(sp); sp.connect(g); g.connect(c.destination);
      Object.assign(this, { on: true, src, sp, g });
      ui.liveState();
    },
    bypassed() { return this.which === 'original' || $('#bypass').checked; },
    stop() {
      if (!this.on) return;
      this.on = false;
      try { this.sp.onaudioprocess = null; this.src.disconnect(); this.sp.disconnect(); this.g.disconnect(); } catch (_) {}
      this.stream.getTracks().forEach(t => t.stop()); this.last = null;
      if (this.remote) { this.remote.close(); this.remote = null; }
      ui.liveState();
    },
    setVol(v) { this.vol = v; if (this.g) this.g.gain.setTargetAtTime(v, ctx.currentTime, 0.02); },
    apply(params) { if (this.pl && this.mode === 'browser') this.pl.configure(params); if (this.remote) this.remote.setProfile(params); },
    latencyMs() {
      if (!ctx) return NaN;
      const dev = ((ctx.baseLatency || 0) + (ctx.outputLatency || 0)) * 1000, chunkMs = this.chunk / ctx.sampleRate * 1000;
      if (this.mode === 'server' && this.remote) { // one extra block: the reply for block k is played in callback k+1
        return 3 * chunkMs + chunkMs * this.remote.queue.length + this.remote.latencySamples / ctx.sampleRate * 1000 + dev; }
      if (!this.pl) return NaN;
      return (2 * this.chunk + this.pl.latencySamples) / ctx.sampleRate * 1000 + dev;
    },
  };

  /* ============================ profile changes ============================ */
  store.on((diff, source) => {
    if (source !== 'undo') {
      undoStack.push(Object.fromEntries(diff.map(d => [d.key, d.from])));
      if (undoStack.length > 30) undoStack.shift();
    }
    $('#undoBtn').disabled = !undoStack.length;
    LS.set('ve.working.v1', store.get());
    ui.renderDescs(); ui.renderMixerValues(diff, source); ui.renderJson(); ui.renderLog(); ui.renderUseState();
    if (live.on && !S.active) live.apply(store.get());
    if (S.rec) preview.render(false);
  });

  function useVoice() {
    const name = ($('#profName').value || '').trim().slice(0, 40) || 'My voice';
    $('#profName').value = name;
    S.active = { name, params: store.get(), savedAt: new Date().toISOString() };
    const persisted = LS.set('ve.active.v1', S.active);
    if (live.on) { live.apply(S.active.params); live.profileName = name; }
    ui.renderActive();
    const msg = `“${name}” is now your active voice profile` + (persisted ? '.' : ', for this session only (this browser blocked saving it).');
    ui.useMsg(msg + ' Live microphone uses it.', persisted ? 'good' : 'warn');
  }
  function editActive() {
    if (!S.active) return;
    store.replace(S.active.params, 'load'); $('#profName').value = S.active.name;
    ui.useMsg(`Editing “${S.active.name}”. Press Use voice again to save your changes.`, '');
  }

  /* ============================ AI chat ============================ */
  async function send(text) {
    text = (text || '').trim(); if (!text || S.busy) return;
    S.busy = true; $('#send').disabled = true; $('#prompt').value = '';
    try {
    addMsg('user', text);
    $$('#suggest button').forEach(b => b.disabled = true);
    $('#msgs').setAttribute('aria-busy', 'true');
    $('#send').textContent = 'Working…';
    const thinking = addMsg('ai', 'Working out the settings…');
    thinking.classList.add('thinking');
    const ctxAI = { profile: store.get(), analysis: S.analysis, reference: S.reference, history: S.history };
    let res, note = '';
    try { res = await S.interpreter.interpret(text, ctxAI); }
    catch (e) {
      const code = e && e.code;
      const who = S.interpreter.name === 'server' ? 'The voice server' : 'Claude';
      if (code === 'not_granted') { S.interpreter = AI.OfflineInterpreter; ui.renderWho(); note = 'Claude was not allowed for this page, so the offline interpreter handled this.'; }
      else note = who + ' could not answer (' + (code || (e && e.message) || 'error') + '), so the offline interpreter handled this.';
      res = await AI.OfflineInterpreter.interpret(text, ctxAI);
    }
    if (res.serverNote) note = res.serverNote;
    // Preserve fields manually edited while the model was answering.
    const current = store.get(), changes = {}, skipped = [];
    for (const [k, v] of Object.entries(res.changes || {})) {
      if (current[k] !== ctxAI.profile[k]) skipped.push(k); else changes[k] = v;
    }
    if (skipped.length) note += (note ? ' ' : '') + 'Kept the settings you edited while I was answering: ' + skipped.map(k => VP.SCHEMA[k]?.label || k).join(', ') + '.';
    const { diff, rejected } = store.set(changes, 'ai');
    const plain = AI.describeDiff(diff);
    let reply = res.reply || (diff.length ? 'Done: ' + plain.join(', ') + '.' : 'Nothing needed to change.');
    if (!diff.length && res.reply && Object.keys(res.changes || {}).length) reply += ' (Those settings were already in place.)';
    if (!S.rec && diff.length) reply += '\n\nRecord a sample or use the sample voice to hear it.';
    thinking.remove();
    const el = addMsg('ai', reply, { diff, rejected: (rejected || []).concat(res.rejected || []), note });
    if (note) el.classList.add('err');
    S.history.push({ role: 'user', content: text }, { role: 'assistant', content: JSON.stringify({ reply, changes: Object.fromEntries(diff.map(d => [d.key, d.to])) }) });
    S.history = S.history.slice(-24);
    return { diff, reply };
    } catch (e) {
      $('#msgs .thinking')?.remove();
      $('#prompt').value = text;
      addMsg('ai', 'I could not apply that request. Your text is still in the box; please try again.', { note: e.message });
    } finally {
      S.busy = false; $('#send').disabled = false; $('#send').innerHTML = 'Send <span aria-hidden="true">↗</span>';
      $$('#suggest button').forEach(b => b.disabled = false);
      $('#msgs').setAttribute('aria-busy', 'false');
    }
  }

  function addMsg(role, text, meta) {
    $('#chatEmpty').hidden = true;
    const el = document.createElement('div'); el.className = 'msg ' + (role === 'user' ? 'user' : 'ai'); el.textContent = text;
    if (meta && meta.note) { const n = document.createElement('div'); n.className = 'small warn'; n.style.marginTop = '6px'; n.textContent = meta.note; el.appendChild(n); }
    if (meta && meta.diff && meta.diff.length) {
      const d = document.createElement('details'); d.innerHTML = '<summary>Settings changed (' + meta.diff.length + ')</summary><ul></ul>';
      for (const x of meta.diff) { const li = document.createElement('li'); li.textContent = `${VP.SCHEMA[x.key].label}: ${VP.formatValue(x.key, x.from)} → ${VP.formatValue(x.key, x.to)}`; d.querySelector('ul').appendChild(li); }
      el.appendChild(d);
    }
    if (meta && meta.rejected && meta.rejected.length) { const n = document.createElement('div'); n.className = 'small muted'; n.textContent = 'Ignored invalid settings: ' + meta.rejected.join(', '); el.appendChild(n); }
    $('#msgs').appendChild(el); $('#msgs').scrollTop = $('#msgs').scrollHeight;
    return el;
  }

  /* ============================ UI ============================ */
  const ui = {
    recMsg(t, k = '') { const el = $('#recStatus'); el.textContent = t; el.className = 'status ' + k; },
    previewMsg(t) { $('#previewStatus').textContent = t; },
    useMsg(t, k) { const el = $('#useStatus'); el.textContent = t; el.className = 'status small ' + (k || ''); },
    recState(on) {
      const b = $('#recBtn'); b.textContent = on ? 'Stop recording' : 'Record'; b.classList.toggle('stop', on); b.classList.toggle('primary', !on);
      if (on) { $('#recStatus').innerHTML = '<span class="recdot"></span>Recording… 0.0 s (stops at 20 s)'; $('#recStatus').className = 'status'; }
    },
    recTick(s) { if (recorder.on) { $('#recStatus').innerHTML = `<span class="recdot"></span>Recording… ${f(s, 1)} s (stops at 20 s)`; if (s >= 10 && !recorder._hinted) { recorder._hinted = true; } } },
    renderSummary() {
      const ul = $('#plainSummary'), lines = VA.plainSummary(S.analysis);
      ul.hidden = !lines.length || !S.analysis.ok; ul.innerHTML = '';
      for (const l of lines) { const li = document.createElement('li'); li.textContent = l; ul.appendChild(li); }
      $('#matchBtn').disabled = !(S.analysis && S.analysis.ok);
    },
    playState() {
      const p = player.playing;
      $('#playIcon').innerHTML = p ? '<path d="M6 4h4v16H6zM14 4h4v16h-4z"/>' : '<path d="M7 4v16l13-8z"/>';
      $('#playBtn').setAttribute('aria-label', p ? 'Pause preview' : 'Play preview');
      $('#mxPlay').textContent = p ? 'Pause preview' : 'Play preview';
    },
    liveState() {
      const b = $('#liveBtn'), m = $('#mxLive');
      b.textContent = live.on ? 'Stop live voice' : 'Start live voice'; b.classList.toggle('stop', live.on);
      m.textContent = live.on ? 'Stop live' : 'Start live'; m.classList.toggle('stop', live.on);
      $('#liveBadge').textContent = live.on ? 'Live' : 'Off';
      $('#liveBadge').classList.toggle('ready', live.on);
      if (live.on) ui.liveMsg(`Live with “${live.profileName}”.` + (live.vol === 0 ? ' Monitoring is muted: put on headphones, then raise the volume.' : ''), '');
    },
    liveMsg(t, k) { const el = $('#liveStatus'); el.textContent = t; el.className = 'status small ' + (k || ''); },
    renderDescs() {
      const box = $('#descs'), ds = VP.describe(store.get());
      if (!box.children.length) box.innerHTML = ds.map(d => `<div class="desc" data-d="${d.id}"><div class="row"><span>${d.label}</span><output></output></div><div class="track"><i></i></div></div>`).join('');
      for (const d of ds) {
        const row = box.querySelector(`[data-d="${d.id}"]`), value = Math.round(d.v * 100);
        row.querySelector('i').style.width = value + '%';
        row.querySelector('output').textContent = value + '%';
      }
    },
    renderActive() {
      const c = $('#activeChip');
      if (S.active) {
        c.className = 'chip'; c.innerHTML = '<span class="dot"></span><span>Active voice profile: <b></b></span><button class="btn sm" id="editActive">Edit</button>';
        c.querySelector('b').textContent = S.active.name; $('#editActive').onclick = editActive;
        $('#liveNote').textContent = `Runs “${S.active.name}” on the microphone in real time. Wear headphones to avoid feedback.`;
      } else { c.className = 'chip none'; c.textContent = 'No active voice yet'; $('#liveNote').textContent = 'Press Use voice to choose the profile that live voice runs. Until then it uses the current settings. Wear headphones to avoid feedback.'; }
      ui.renderUseState();
    },
    renderUseState() {
      const same = S.active && JSON.stringify(S.active.params) === JSON.stringify(store.get());
      for (const b of [$('#useBtn'), $('#mxUse')]) b.textContent = same ? 'Saved & active' : 'Use voice';
      if (same) ui.useMsg('Your saved profile is ready for live monitoring.', 'good');
      if (S.active && !same) ui.useMsg('You have unsaved changes. Press Use voice to update your live profile.', 'warn');
    },
    renderWho() {
      const n = S.interpreter.name;
      $('#connectionStatus').innerHTML = '<span class="dot"></span>' + (S.server?.interpreter === 'gateway' ? 'AI connected · Local audio' : 'Local studio · Offline AI');
      $('#aiWho').textContent = n === 'claude' ? 'AI: Claude' : n === 'server' ? 'AI: voice server (' + (S.server ? S.server.interpreter + (S.server.model ? ': ' + S.server.model : '') : '?') + ')' : 'AI: offline interpreter (keyword rules)';
    },
    renderEngine() {
      const sel = $('#engineSel'); if (!sel) return;
      sel.querySelector('[value=server]').disabled = !S.server;
      sel.querySelector('[value=server]').textContent = S.server ? 'Server (' + S.server.backend.name + ')' : 'Server (not connected)';
      sel.value = S.engine;
      $('#engineNote').textContent = S.server ? S.server.backend.detail : 'Open the app from `voice-engine serve` to use the CPU/ROCm server.';
    },

    /* ---------- mixer ---------- */
    buildMixer() {
      const chain = $('#chain'); chain.innerHTML = '';
      const ACT = { ns: 'mNsAct', vad: 'mVadAct', deess: 'mDeAct', comp: 'mCompAct', loud: 'mLoudAct', out: 'mLimAct' };
      VP.GROUPS.forEach((g, gi) => {
        const keys = Object.keys(VP.SCHEMA).filter(k => VP.SCHEMA[k].group === g.id);
        const mod = document.createElement('div'); mod.className = 'mod'; mod.dataset.group = g.id;
        mod.innerHTML = `<div class="mod-head"><span class="n">${gi + 1}</span><h3>${esc(g.name)}</h3><span class="act" id="${ACT[g.id] || ''}"></span></div>`;
        for (const k of keys) {
          const s = VP.SCHEMA[k], row = document.createElement('div'); row.className = 'param'; row.dataset.key = k;
          if (s.type === 'bool') {
            row.innerHTML = `<div class="row"><label for="p-${k}">${esc(s.label)}</label><input type="checkbox" class="switch" id="p-${k}"></div>`;
            row.querySelector('input').addEventListener('change', e => store.set({ [k]: e.target.checked }, 'mixer'));
          } else {
            const scale = s.pct ? 100 : 1;
            row.innerHTML = `<div class="row"><label for="r-${k}">${esc(s.label)}</label><span><input type="number" id="n-${k}" min="${s.min * scale}" max="${s.max * scale}" step="${s.step * scale}" aria-label="${esc(s.label)} value"><span class="u">${s.pct ? '%' : esc(s.unit)}</span></span></div><input type="range" id="r-${k}" min="${s.min}" max="${s.max}" step="${s.step}">`;
            row.querySelector('input[type=range]').addEventListener('input', e => store.set({ [k]: parseFloat(e.target.value) }, 'mixer'));
            row.querySelector('input[type=number]').addEventListener('change', e => {
              store.set({ [k]: parseFloat(e.target.value) / scale }, 'mixer');
              e.target.value = store.get()[k] * scale;
            });
          }
          mod.appendChild(row);
        }
        chain.appendChild(mod);
      });
      ui.renderMixerValues();
    },
    renderMixerValues(diff, source) {
      const p = store.get();
      for (const [k, s] of Object.entries(VP.SCHEMA)) {
        if (s.type === 'bool') { const el = $('#p-' + k); if (el) el.checked = p[k]; }
        else { const r = $('#r-' + k), n = $('#n-' + k); if (r && document.activeElement !== r) r.value = p[k]; if (n && document.activeElement !== n) n.value = s.pct ? Math.round(p[k] * 100) : p[k]; }
      }
      const en = { ns: p.nsEnabled, vad: p.vadEnabled, hpf: p.hpfEnabled, deess: p.deEssEnabled, comp: p.compEnabled, loud: p.loudEnabled };
      $$('.mod').forEach(m => m.classList.toggle('off', en[m.dataset.group] === false));
      if (diff && source === 'ai' || source === 'load' || source === 'match') for (const d of diff || []) { const row = document.querySelector(`.param[data-key="${d.key}"]`); if (row) { row.classList.remove('hl'); void row.offsetWidth; row.classList.add('hl'); } }
    },
    renderJson() { $('#profJson').textContent = JSON.stringify(store.get(), null, 2); },
    renderLog() {
      const box = $('#log'), items = store.log.slice(-60).reverse();
      if (!items.length) { box.innerHTML = '<span class="muted">No changes yet.</span>'; return; }
      box.innerHTML = items.map(d => `<div><span class="tag ${d.source === 'ai' ? 'ai' : ''}">${esc(d.source)}</span>${esc(VP.SCHEMA[d.key].label)} ${esc(VP.formatValue(d.key, d.from))} → ${esc(VP.formatValue(d.key, d.to))}</div>`).join('');
    },
    renderAnalysis() {
      const a = S.analysis, r = S.reference, t = $('#anaTable');
      if (!a) return;
      if (!a.ok) { t.innerHTML = `<tbody><tr><td class="warn">${esc(a.reason)}</td></tr></tbody>`; return; }
      const rows = [
        ['Median pitch (F0)', f(a.f0MedianHz, 0) + ' Hz', 'kept as yours'], ['Mean pitch', f(a.f0MeanHz, 0) + ' Hz', ''], ['Pitch range (p10–p90)', f(a.f0RangeSt, 1) + ' st', ''],
        ['Formants', 'not measured in browser build', ''], ['Spectral centroid', f(a.spectralCentroidHz, 0) + ' Hz', ''], ['Spectral slope', f(a.spectralSlopeDbOct, 1) + ' dB/oct', ''],
        ['Low-mid energy (150–500 Hz)', f(a.lowMidRelDb, 1) + ' dB', f(r.lowMidRelDb, 1) + ' dB'], ['Presence (2–5 kHz)', f(a.presenceRelDb, 1) + ' dB', f(r.presenceRelDb, 1) + ' dB'],
        ['Sibilance (5–9 kHz)', f(a.sibilanceRelDb, 1) + ' dB', '≤ ' + f(r.sibilanceRelDb, 1) + ' dB'], ['HNR (estimate)', f(a.hnrDb, 1) + ' dB', ''],
        ['Speech RMS', f(a.rmsDb, 1) + ' dBFS', ''], ['Integrated loudness', f(a.integratedLufs, 1) + ' LUFS', f(r.integratedLufs, 1) + ' LUFS'],
        ['Loudness spread (p95–p10)', f(a.loudnessRangeLu, 1) + ' LU', f(r.loudnessRangeLu, 1) + ' LU'], ['Crest factor', f(a.crestDb, 1) + ' dB', ''],
        ['Noise floor', f(a.noiseFloorDb, 0) + ' dBFS', '≤ ' + f(r.noiseFloorDb, 0) + ' dBFS'], ['Speaking rate', f(a.syllablesPerSec, 1) + ' syllables/s', ''],
        ['Pauses', f(a.pausesPerMin, 0) + ' per min, mean ' + f(a.pauseMeanS, 2) + ' s', ''],
      ];
      t.innerHTML = '<thead><tr><th>Measure</th><th class="num">You</th><th class="num">Reference</th></tr></thead><tbody>' + rows.map(x => `<tr><td>${x[0]}</td><td class="num">${x[1]}</td><td class="num">${x[2]}</td></tr>`).join('') + '</tbody>';
    },
    renderRef() {
      $('#refText').value = JSON.stringify(S.reference, null, 2);
      $('#refNote').hidden = S.reference.source !== VA.DEFAULT_REFERENCE.source;
    },
  };

  /* ============================ meters + bar ============================ */
  function meterSource() {
    if (live.on && live.last) return { info: live.last, label: `Live microphone (${live.which})` };
    const pi = player.info(); if (pi) return { info: pi, label: `Preview (${player.which})` };
    return null;
  }
  function renderMeters() {
    const m = meterSource();
    $('#mxSrc').textContent = m ? 'Meters: ' + m.label : 'Meters idle: play the preview or start live';
    const I = m && m.info;
    const set = (id, v) => { $(id).textContent = v; };
    set('#mInPk', I ? f(Math.max(-99, I.inDb), 0) : '–'); set('#mOutPk', I ? f(Math.max(-99, I.outDb), 0) : '–');
    $('#mInBar').style.width = I ? Math.max(0, Math.min(100, (I.inDb + 60) / 60 * 100)) + '%' : '0';
    $('#mOutBar').style.width = I ? Math.max(0, Math.min(100, (I.outDb + 60) / 60 * 100)) + '%' : '0';
    set('#mLufs', I ? f(Math.max(-99, I.outLufs), 0) : '–');
    const v = $('#mVad'); v.textContent = I ? (I.speech ? 'Speech' : 'No speech') : '–'; v.classList.toggle('on', !!(I && I.speech));
    set('#mNs', I ? f(I.nsDb, 1) : '–'); set('#mDe', I ? f(I.deEssDb, 1) : '–'); set('#mComp', I ? f(I.compDb, 1) : '–'); set('#mLim', I ? f(I.limDb, 1) : '–');
    const act = (id, txt, on) => { const el = $(id); if (el) { el.textContent = txt; el.classList.toggle('on', !!on); } };
    act('#mNsAct', I && !I.bypass ? f(I.nsDb, 1) + ' dB' : '', I && I.nsDb < -1);
    act('#mVadAct', I ? (I.speech ? 'speech' : 'no speech') : '', I && I.speech);
    act('#mDeAct', I && !I.bypass ? f(I.deEssDb, 1) + ' dB' : '', I && I.deEssDb < -0.5);
    act('#mCompAct', I && !I.bypass ? f(I.compDb, 1) + ' dB' : '', I && I.compDb < -0.5);
    act('#mLoudAct', I && !I.bypass ? sg(I.loudDb, 1) + ' dB' : '', I && Math.abs(I.loudDb) > 0.5);
    act('#mLimAct', I && !I.bypass ? f(I.limDb, 1) + ' dB' : '', I && I.limDb < -0.3);
  }
  let waveform = { data: null, n: 0, peaks: null };
  function drawBar() {
    const cv = $('#barCv'), g = cv.getContext('2d'), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const cs = getComputedStyle(document.documentElement), data = player.which === 'original' || !preview.enhData ? (S.rec && S.rec.data) : preview.enhData;
    g.fillStyle = cs.getPropertyValue('--sunk'); g.fillRect(0, 0, W, H);
    const seek = $('#bar');
    seek.setAttribute('aria-valuemax', String(player.duration));
    seek.setAttribute('aria-valuenow', String(Math.max(0, player.position)));
    seek.setAttribute('aria-disabled', String(!player.duration));
    seek.setAttribute('aria-valuetext', f(Math.max(0, player.position), 1) + ' of ' + f(player.duration, 1) + ' seconds');
    if (!data) return;
    const n = Math.floor(W / 2), step = data.length / n, pos = player.duration ? player.position / player.duration : 0;
    if (waveform.data !== data || waveform.n !== n) {
      const peaks = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const a = Math.floor(i * step), b = Math.floor((i + 1) * step);
        for (let j = a; j < b; j += 4) peaks[i] = Math.max(peaks[i], Math.abs(data[j]));
      }
      waveform = { data, n, peaks };
    }
    for (let i = 0; i < n; i++) {
      const pk = waveform.peaks[i];
      const h = Math.max(1, Math.min(1, pk) * (H - 4));
      g.fillStyle = i / n <= pos ? cs.getPropertyValue('--enh') : cs.getPropertyValue('--orig');
      g.fillRect(i * 2, (H - h) / 2, 1.4, h);
    }
  }
  let lastPerf = 0, lastFrame = 0;
  function frame(t) {
    if (t - lastFrame < 50) { requestAnimationFrame(frame); return; }
    lastFrame = t;
    if (S.view === 'mixer') renderMeters();
    if (player.playing || recorder.on) drawBar();
    if (player.duration) $('#time').textContent = `${f(player.position, 1)} / ${f(player.duration, 1)} s`;
    if (t - lastPerf > 500) { lastPerf = t; renderPerf(); }
    requestAnimationFrame(frame);
  }
  function renderPerf() {
    if (!live.on) return;
    const fs = ctx.sampleRate, budget = live.chunk / fs * 1000, p50 = live.stats.pct(0.5), p95 = live.stats.pct(0.95);
    let rows;
    if (live.mode === 'server' && live.remote) {
      const R = live.remote, rt = R.rtt.slice().sort((a, b) => a - b), q = (p) => rt.length ? rt[Math.min(rt.length - 1, Math.floor(p * (rt.length - 1)))] : NaN;
      rows = [['Backend', 'server: ' + (S.server ? S.server.backend.name + ' (' + S.server.backend.detail + ')' : '?')], ['Sample rate', fs + ' Hz'], ['Chunk', `${live.chunk} samples (${f(budget, 1)} ms budget)`],
        ['Server processing (last)', f(live.last ? live.last.processMs : NaN, 2) + ' ms'], ['Round trip p50 / p95', `${f(q(0.5), 1)} / ${f(q(0.95), 1)} ms`],
        ['Blocks / underruns', `${live.blocks} / ${R.underruns}`], ['Pipeline latency', f(R.latencySamples / fs * 1000, 1) + ' ms'], ['Estimated end-to-end', f(live.latencyMs(), 0) + ' ms']];
    } else rows = [['Backend', 'browser-dsp (CPU, main thread)'], ['Sample rate', fs + ' Hz'], ['Chunk', `${live.chunk} samples (${f(budget, 1)} ms budget)`],
      ['Processing p50 / p95', `${f(p50, 2)} / ${f(p95, 2)} ms`], ['Real-time factor (p95)', f(p95 / budget, 3)], ['Overruns', `${live.over} of ${live.blocks}`],
      ['Pipeline latency', f(live.pl.latencySamples / fs * 1000, 1) + ' ms'], ['Estimated end-to-end', f(live.latencyMs(), 0) + ' ms']];
    $('#perfBody').innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td></tr>`).join('');
  }

  /* ============================ self-test ============================ */
  async function runTests() {
    const btn = $('#runTests'), body = $('#testTable tbody'), msg = $('#testMsg'); btn.disabled = true; body.innerHTML = '';
    const results = [];
    for (const t of window.SelfTest.TESTS) {
      msg.textContent = 'Running: ' + t.name; await new Promise(r => setTimeout(r, 10));
      let r; try { r = t.run(); } catch (e) { r = { pass: false, detail: 'threw: ' + e.message }; }
      results.push({ id: t.id, ...r });
      const tr = document.createElement('tr');
      tr.innerHTML = `<td class="${r.pass ? 'pass' : 'fail'}">${r.pass ? 'Pass' : 'Fail'}</td><td></td><td>${esc(t.req)}</td><td></td>`;
      tr.children[1].textContent = t.name; tr.children[3].textContent = r.detail; body.appendChild(tr);
    }
    const nf = results.filter(r => !r.pass).length;
    msg.textContent = nf ? `${nf} of ${results.length} failed` : `All ${results.length} passed`; msg.className = 'small ' + (nf ? 'bad' : 'good');
    btn.disabled = false; window.__ve.testResults = results; return results;
  }

  function setTheme(theme) {
    theme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    $('#themeBtn').textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    $('#themeBtn').setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' theme');
    requestAnimationFrame(drawBar);
  }
  function download(blob, filename) {
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function encodeWav(data, fs) {
    const buf = new ArrayBuffer(44 + data.length * 2), view = new DataView(buf);
    const str = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
    str(0, 'RIFF'); view.setUint32(4, 36 + data.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, fs, true); view.setUint32(28, fs * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    str(36, 'data'); view.setUint32(40, data.length * 2, true);
    for (let i = 0; i < data.length; i++) { const x = Math.max(-1, Math.min(1, data[i])); view.setInt16(44 + i * 2, Math.round(x * (x < 0 ? 32768 : 32767)), true); }
    return new Blob([buf], { type: 'audio/wav' });
  }

  /* ============================ wiring ============================ */
  function setView(v) {
    S.view = v;
    $$('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === v)));
    $('#simpleView').hidden = v !== 'simple'; $('#mixerView').hidden = v !== 'mixer';
    $('.studio-intro').hidden = v !== 'simple';
    if (v === 'simple') requestAnimationFrame(drawBar);
  }
  function wire() {
    $$('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
    $('#mxBack').addEventListener('click', () => setView('simple'));
    $('#composer').addEventListener('submit', (e) => { e.preventDefault(); send($('#prompt').value); });
    $('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send($('#prompt').value); } });
    $$('#suggest button').forEach(b => b.addEventListener('click', () => send(b.textContent)));
    $('#recBtn').addEventListener('click', async () => {
      if (recorder.on) { recorder.stop(); return; }
      if (recorder.pending || live.pending) return;
      recorder.pending = true; $('#recBtn').disabled = true;
      ui.recMsg('Opening microphone…');
      try { await recorder.start(); } catch (e) { ui.recMsg(e instanceof UserError ? e.message : 'Recording failed: ' + e.message, 'bad'); }
      finally { recorder.pending = false; $('#recBtn').disabled = false; }
    });
    $('#sampleBtn').addEventListener('click', async () => {
      if (recorder.pending || live.pending) return;
      if (recorder.on) recorder.stop();
      try { const c = await audio(); const s = D.makeSampleVoice(c.sampleRate); await setRecording(s.data.slice(), c.sampleRate, 'Sample voice (speech, keyboard, paper and fan noise)'); }
      catch (e) { ui.recMsg(e instanceof UserError ? e.message : e.message, 'bad'); }
    });
    $('#uploadBtn').addEventListener('click', () => $('#fileIn').click());
    $('#fileIn').addEventListener('change', async (e) => {
      const file = e.target.files && e.target.files[0]; if (!file) return; e.target.value = '';
      if (file.size > 50 * 1024 * 1024) { ui.recMsg('Choose an audio file smaller than 50 MB.', 'bad'); return; }
      if (recorder.pending || live.pending) return;
      if (recorder.on) recorder.stop();
      ui.recMsg('Decoding ' + file.name + '…');
      try {
        const c = await audio(), buf = await c.decodeAudioData(await file.arrayBuffer());
        const n = Math.min(buf.length, buf.sampleRate * 60), mono = new Float32Array(n);
        for (let ch = 0; ch < buf.numberOfChannels; ch++) { const d = buf.getChannelData(ch); for (let i = 0; i < n; i++) mono[i] += d[i] / buf.numberOfChannels; }
        await setRecording(mono, buf.sampleRate, file.name + (buf.duration > 60 ? ' (first 60 s)' : ''));
      } catch (err) { ui.recMsg(err instanceof UserError ? err.message : 'That file could not be read as audio. Try WAV, MP3 or M4A.', 'bad'); }
    });
    const toggle = () => player.playing ? player.pause() : player.play().catch(e => ui.previewMsg(e.message));
    $('#playBtn').addEventListener('click', toggle); $('#mxPlay').addEventListener('click', toggle);
    $$('[data-ab]').forEach(b => b.addEventListener('click', () => { player.setWhich(b.dataset.ab); drawBar(); }));
    $('#bar').addEventListener('click', (e) => {
      if (!player.duration) return; const r = e.currentTarget.getBoundingClientRect(), pos = (e.clientX - r.left) / r.width * player.duration;
      if (player.playing) player.play(pos); else { player.offset = pos; drawBar(); }
    });
    $('#bar').addEventListener('keydown', e => {
      if (!player.duration || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      e.preventDefault();
      const pos = e.key === 'Home' ? 0 : e.key === 'End' ? player.duration : Math.max(0, Math.min(player.duration, player.position + (e.key === 'ArrowRight' ? 1 : -1) * 5));
      if (player.playing) player.play(pos); else { player.offset = pos; drawBar(); }
    });
    $('#openMixer').addEventListener('click', () => setView('mixer'));
    $('#undoBtn').addEventListener('click', () => {
      const changes = undoStack.pop(); if (changes) store.set(changes, 'undo');
      $('#undoBtn').disabled = !undoStack.length;
    });
    $('#themeBtn').addEventListener('click', () => {
      const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      setTheme(theme); LS.set('ve.theme.v1', theme);
    });
    $('#downloadAudio').addEventListener('click', () => {
      if (!preview.enhData || !S.rec || $('#downloadAudio').disabled) return;
      download(encodeWav(preview.enhData, S.rec.fs), 'enhanced-voice.wav');
    });
    $('#useBtn').addEventListener('click', useVoice); $('#mxUse').addEventListener('click', useVoice);
    const liveToggle = async () => {
      if (live.on) { live.stop(); ui.liveMsg('Live voice stopped.'); return; }
      if (live.pending || recorder.pending) return;
      live.pending = true; $('#liveBtn').disabled = $('#mxLive').disabled = true;
      ui.liveMsg('Opening live microphone…');
      try { await live.start(); } catch (e) { ui.liveMsg(e instanceof UserError ? e.message : 'Live voice failed: ' + e.message, 'bad'); }
      finally { live.pending = false; $('#liveBtn').disabled = $('#mxLive').disabled = false; }
    };
    $('#liveBtn').addEventListener('click', liveToggle); $('#mxLive').addEventListener('click', liveToggle);
    $$('[data-live]').forEach(b => b.addEventListener('click', () => { live.which = b.dataset.live; $$('[data-live]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); }));
    $('#liveVol').addEventListener('input', (e) => { live.setVol(e.target.value / 100); $('#volumeValue').textContent = +e.target.value ? e.target.value + '%' : 'Muted'; if (live.on) ui.liveState(); });
    $('#chunkSel').addEventListener('change', async (e) => { live.chunk = parseInt(e.target.value, 10); if (live.on) { live.stop(); try { await live.start(); } catch (err) { ui.liveMsg(err.message, 'bad'); } } });
    $('#matchBtn').addEventListener('click', () => {
      const ch = VA.matchToReference(S.analysis, S.reference, store.get()); const { diff } = store.set(ch, 'match');
      addMsg('ai', diff.length ? 'I matched your voice to the reference: ' + AI.describeDiff(diff).join(', ') + '.' : 'Your settings already match the reference.', { diff });
    });
    $('#refApply').addEventListener('click', () => {
      try {
        const obj = JSON.parse($('#refText').value), need = ['integratedLufs', 'truePeakDb', 'lowMidRelDb', 'presenceRelDb', 'sibilanceRelDb', 'loudnessRangeLu', 'noiseFloorDb'];
        const missing = need.filter(k => !Number.isFinite(obj[k]));
        if (missing.length) throw new Error('missing numeric ' + missing.join(', '));
        if (obj.source === VA.DEFAULT_REFERENCE.source) obj.source = 'Edited reference';
        S.reference = obj; LS.set('ve.reference.v1', obj); ui.renderRef(); ui.renderAnalysis();
        $('#refMsg').textContent = 'Reference applied. Use “Match my voice” or ask the AI for a podcast sound to use it.'; $('#refMsg').className = 'small good';
      } catch (e) { $('#refMsg').textContent = 'Not applied: ' + e.message + '.'; $('#refMsg').className = 'small bad'; }
    });
    $('#refReset').addEventListener('click', () => { S.reference = { ...VA.DEFAULT_REFERENCE }; LS.set('ve.reference.v1', S.reference); ui.renderRef(); ui.renderAnalysis(); $('#refMsg').textContent = 'Defaults restored.'; $('#refMsg').className = 'small'; });
    $('#resetProf').addEventListener('click', () => store.replace(VP.DEFAULTS, 'reset'));
    $('#copyJson').addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(JSON.stringify(store.get(), null, 2)); $('#jsonMsg').textContent = 'Copied.'; }
      catch (_) { $('#jsonMsg').textContent = 'Copying is blocked here; select the text above instead.'; }
    });
    $('#saveJson').addEventListener('click', async () => {
      try {
        const data = JSON.stringify({ name: $('#profName').value, params: store.get() }, null, 2);
        if (S.caps.downloads) await S.caps.downloads.save({ filename: 'voice-profile.json', data });
        else download(new Blob([data], { type: 'application/json' }), 'voice-profile.json'); $('#jsonMsg').textContent = 'Saved.'; }
      catch (e) { $('#jsonMsg').textContent = e && e.code === 'declined' ? 'Save cancelled.' : 'Could not save (' + ((e && e.code) || 'error') + ').'; }
    });
    $('#bypass').addEventListener('change', (e) => { $('#mxSrc').title = e.target.checked ? 'Bypass is on: live output is the untouched input' : ''; if (e.target.checked) player.setWhich('original'); else player.setWhich('enhanced'); });
    $('#runTests').addEventListener('click', runTests);
    $('#engineSel').addEventListener('change', async (e) => {
      S.engine = e.target.value === 'server' && S.server ? 'server' : 'browser'; LS.set('ve.engine.v1', S.engine); ui.renderEngine();
      if (live.on) { live.stop(); try { await live.start(); } catch (err) { ui.liveMsg(err.message, 'bad'); } }
      if (S.rec) preview.render(true);
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.view === 'mixer') setView('simple'); });
    new ResizeObserver(() => drawBar()).observe($('#bar'));
  }

  function init() {
    setTheme(LS.get('ve.theme.v1') || 'dark');
    ui.buildMixer(); ui.renderDescs(); ui.renderJson(); ui.renderLog(); ui.renderActive(); ui.renderRef(); ui.renderWho(); wire();
    if (S.active) $('#profName').value = S.active.name;
    requestAnimationFrame(frame);
    if (window.claude && typeof window.claude.use === 'function') {
      window.claude.use('sample').then(s => { if (s) { S.caps.sample = s; S.interpreter = AI.makeClaudeInterpreter(s); ui.renderWho(); } }).catch(() => {});
      window.claude.use('downloads').then(d => { S.caps.downloads = d; $('#saveJson').hidden = false; }).catch(() => {});
    }
    if (window.VoiceServer) window.VoiceServer.detect().then(h => {
      if (!h) return;
      S.server = h; S.engine = LS.get('ve.engine.v1') === 'browser' ? 'browser' : 'server';
      if (!S.caps.sample) { S.interpreter = window.VoiceServer.interpreter(); }
      ui.renderWho(); ui.renderEngine(); if (S.rec) preview.render(true);
    });
    ui.renderEngine();
    window.addEventListener('pagehide', () => { recorder.stop(); live.stop(); player.stop(); });
    Object.assign(window.__ve, { send, useVoice, editActive, preview, player, live, recorder, runTests, setView, ready: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
