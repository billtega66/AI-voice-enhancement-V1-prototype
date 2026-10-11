(function (root) {
  'use strict';
  const VP = root.VoiceProfile;
  function make(base, changes) {
    const valid = VP.validateChanges(changes).ok;
    const gentler = {};
    for (const [key, value] of Object.entries(valid)) {
      gentler[key] = VP.SCHEMA[key].type === 'bool' ? value : base[key] + (value - base[key]) * 0.5;
    }
    const variants = [{ label: 'Suggested', changes: valid }, { label: 'Gentler', changes: VP.validateChanges(gentler).ok }];
    const seen = new Set();
    return variants.map(v => ({ ...v, params: { ...base, ...v.changes } })).filter(v => {
      const key = JSON.stringify(v.params);
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
  }
  function measure(data, fs) {
    let clipped = 0, finite = true;
    for (const value of data) { if (!Number.isFinite(value)) finite = false; if (Math.abs(value) >= 1) clipped++; }
    return { finite, clipped, peakDb: root.DSP.linToDb(root.DSP.peak(data)), loudness: root.DSP.integratedLufs(data, fs) };
  }
  root.VoiceCandidates = { make, measure };
})(typeof window !== 'undefined' ? window : globalThis);
