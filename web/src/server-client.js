/* ==========================================================================
   Client for the voice_engine server (server/voice_engine/server.py).
   When the page is served by `voice-engine serve`, the app can run preview
   rendering, live processing and the AI interpreter on the server backend
   (CPU or AMD ROCm) instead of in the browser. Same Voice Profile, same meters.
   ========================================================================== */
(function (root) {
  'use strict';
  const VoiceServer = {
    base: '', health: null,

    /** Resolves to the /api/health payload, or null when no server is reachable. */
    async detect(timeoutMs = 2500) {
      if (typeof location === 'undefined' || !/^https?:$/.test(location.protocol)) return null;
      const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), timeoutMs);
      try {
        const r = await fetch(this.base + '/api/health', { signal: ctl.signal });
        this.health = r.ok ? await r.json() : null;
      } catch (_) { this.health = null; } finally { clearTimeout(t); }
      return this.health;
    },

    async request(path, options = {}, timeoutMs = 45000) {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), timeoutMs);
      try { return await fetch(this.base + path, { ...options, signal: ctl.signal }); }
      catch (e) { if (e.name === 'AbortError') throw new Error('The server took too long to respond. Please try again.'); throw e; }
      finally { clearTimeout(timer); }
    },

    async _err(r) { let m = r.status + ' ' + r.statusText; try { const j = await r.json(); if (j.detail) m = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {} return new Error(m); },

    /** Offline render on the server. Returns {data: Float32Array, infos, latency, processingMs, backend}. */
    async render(data, fs, params, chunk = 1024) {
      const fd = new FormData();
      fd.append('audio', new Blob([data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)], { type: 'application/octet-stream' }), 'take.f32');
      fd.append('fs', String(fs)); fd.append('profile', JSON.stringify(params)); fd.append('chunk', String(chunk));
      const r = await this.request('/api/render', { method: 'POST', body: fd });
      if (!r.ok) throw await this._err(r);
      const j = await r.json();
      const bin = atob(j.audio), u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return { data: new Float32Array(u8.buffer), infos: j.infos, latency: j.latencySamples, processingMs: j.processingMs, backend: j.backend };
    },

    /** Interpreter with the same interface as VoiceAI.OfflineInterpreter. */
    interpreter() {
      const self = this;
      return {
        name: 'server', label: 'server',
        async interpret(text, ctx) {
          const r = await self.request('/api/interpret', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, profile: ctx.profile, analysis: ctx.analysis && ctx.analysis.ok ? ctx.analysis : null, reference: ctx.reference, history: ctx.history || [] }),
          });
          if (!r.ok) throw await self._err(r);
          const j = await r.json();
          return { reply: j.reply, changes: j.changes, rejected: j.rejected, serverNote: j.note, via: j.interpreter };
        },
      };
    },

    /** Real-time stream. Blocks go out as float32; processed blocks come back in order. */
    openStream(fs, params, bypass) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
      ws.binaryType = 'arraybuffer';
      const s = {
        ws, ready: false, queue: [], sentAt: [], rtt: [], underruns: 0, onInfo: null, onError: null, latencySamples: 0,
        send(block) {
          if (!this.ready || ws.readyState !== 1) return false;
          if (this.sentAt.length >= 8 || ws.bufferedAmount > 1 << 18) return false; // back-pressure: drop rather than build delay
          ws.send(block.buffer.slice(block.byteOffset, block.byteOffset + block.byteLength)); this.sentAt.push(performance.now()); return true;
        },
        take(n) {
          const b = this.queue.shift();
          if (b && b.length === n) return b;
          this.underruns++; return null;
        },
        setProfile(p) { if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'profile', profile: p })); },
        setBypass(v) { if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'bypass', value: !!v })); },
        close() { this.onError = null; try { ws.close(); } catch (_) {} },
      };
      s.opened = new Promise((resolve, reject) => {
        const timer = setTimeout(() => { reject(new Error('The live server did not respond.')); ws.close(); }, 10000);
        ws.onclose = () => {
          clearTimeout(timer);
          if (!s.ready) reject(new Error('The live server disconnected before it was ready.'));
          else { s.ready = false; if (s.onError) s.onError('Live connection closed.'); }
        };
        ws.onopen = () => ws.send(JSON.stringify({ type: 'start', sampleRate: fs, profile: params, bypass }));
        ws.onerror = () => { clearTimeout(timer); reject(new Error('Could not connect to the voice server stream.')); };
        ws.onmessage = (ev) => {
          if (typeof ev.data === 'string') {
            const m = JSON.parse(ev.data);
            if (m.type === 'ready') { s.ready = true; s.latencySamples = m.latencySamples; clearTimeout(timer); resolve(s); }
            else if (m.type === 'info' && s.onInfo) s.onInfo(m);
            else if (m.type === 'error') { if (!s.ready) reject(new Error(m.message)); else if (s.onError) s.onError(m.message); }
            return;
          }
          const t = s.sentAt.shift(); if (t) { s.rtt.push(performance.now() - t); if (s.rtt.length > 500) s.rtt.shift(); }
          s.queue.push(new Float32Array(ev.data));
          while (s.queue.length > 3) s.queue.shift(); // keep latency bounded
        };
      });
      return s;
    },
  };
  root.VoiceServer = VoiceServer;
})(typeof window !== 'undefined' ? window : globalThis);
