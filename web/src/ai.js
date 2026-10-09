/* ==========================================================================
   AI Voice Profile Generator.
   Natural language -> structured change set -> validated -> ProfileStore.
   Two interchangeable interpreters behind one interface:
     ClaudeInterpreter  (when the page runs inside Claude and the viewer allows it)
     OfflineInterpreter (keyword rules; always available, clearly labelled)
   Both start from the CURRENT profile and only return changes.
   ========================================================================== */
(function (root) {
  'use strict';
  const VP = root.VoiceProfile, VA = root.VoiceAnalysis;

  /* ---------------- shared helpers ---------------- */
  const PLAIN = {
    warmthDb: ['more warmth', 'less warmth'], mudDb: ['more low-mid body', 'less boominess'], presenceDb: ['more clarity', 'less edge'],
    airDb: ['more brightness', 'less brightness'], pitchSemitones: ['a slightly higher pitch', 'a slightly deeper pitch'],
    deEssThresholdDb: ['gentler S control', 'softer S sounds'], deEssMaxDb: ['softer S sounds', 'gentler S control'], deEssEnabled: ['S-sound control on', 'S-sound control off'],
    compRatio: ['steadier volume', 'more natural dynamics'], compThresholdDb: ['more natural dynamics', 'steadier volume'], compEnabled: ['volume levelling on', 'volume levelling off'],
    makeupDb: ['a little more level', 'a little less level'], targetLufs: ['louder output', 'quieter output'], outputGainDb: ['louder output', 'quieter output'],
    inputGainDb: ['more input level', 'less input level'], nsAmount: ['more background cleanup', 'less background cleanup'], nsEnabled: ['noise suppression on', 'noise suppression off'],
    vadAttenuationDb: ['less muting between words', 'quieter pauses'], vadSensitivity: ['easier speech detection', 'stricter speech detection'], vadHoldMs: ['longer word tails', 'tighter pauses'],
    vadEnabled: ['non-speech muting on', 'non-speech muting off'], hpfHz: ['less rumble', 'more low end'], limiterCeilingDb: ['more headroom', 'less headroom'],
  };
  function describeDiff(diff) {
    const parts = [];
    for (const d of diff) {
      const pl = PLAIN[d.key]; if (!pl) continue;
      const up = typeof d.to === 'boolean' ? d.to : d.to > d.from;
      const phrase = pl[up ? 0 : 1];
      if (!parts.includes(phrase)) parts.push(phrase);
    }
    return parts;
  }

  /* ---------------- Offline interpreter ---------------- */
  const RULES = [
    // [regex, kind, fn(k, p, sign) -> changes]; kind 'quality' can be negated by less/not so/too, 'problem' always means fix it
    [/\b(s sounds?|s's|sibilan\w*|ess(es)?|hiss\w*|lisp)\b/, 'problem', (k, p) => ({ deEssEnabled: true, deEssThresholdDb: p.deEssThresholdDb - 5 * k, deEssMaxDb: p.deEssMaxDb + 3 * k })],
    [/\b(thin|weak|tinny)\b/, 'problem', (k, p) => ({ warmthDb: p.warmthDb + 2 * k, mudDb: Math.min(0, p.mudDb + 1 * k) })],
    [/\b(muffled|muddy|boomy|boxy|dull|unclear|mumbl\w*)\b/, 'problem', (k, p) => ({ mudDb: p.mudDb - 2 * k, presenceDb: p.presenceDb + 1.5 * k })],
    [/\b(harsh|shrill|piercing|edgy|nasal)\b|\bsharp\b(?! s\b| s sound| sibil| ess)/, 'problem', (k, p) => ({ presenceDb: p.presenceDb - 1.5 * k, airDb: p.airDb - 1.5 * k })],
    [/\b(noise|noisy|background|keyboard|typing|keys|fan|hum|buzz|paper|room|traffic|clicks?|air ?con\w*)\b/, 'problem', (k, p) => ({ nsEnabled: true, nsAmount: p.nsAmount + 0.2 * k, vadEnabled: true, vadAttenuationDb: p.vadAttenuationDb - 8 * k })],
    [/\b(inconsistent|uneven|jumps?|varies|fluctuat\w*|up and down)\b/, 'problem', (k, p) => ({ compEnabled: true, compRatio: p.compRatio + 1 * k, compThresholdDb: p.compThresholdDb - 4 * k, makeupDb: p.makeupDb + 1 * k })],
    [/\b(cut off|cuts off|chopp\w*|clipped words|missing words)\b/, 'problem', (k, p) => ({ vadHoldMs: p.vadHoldMs + 150 * k, vadSensitivity: p.vadSensitivity + 0.15 * k })],
    [/\b(too quiet|quiet|soft spoken|can't hear)\b/, 'problem', (k, p) => ({ targetLufs: p.targetLufs + 2 * k })],
    [/\b(warm\w*|body|fuller|full|rich\w*)\b/, 'quality', (k, p, s) => ({ warmthDb: p.warmthDb + 2 * k * s })],
    [/\b(deep\w*|bass\w*|lower (voice|pitch|tone))\b/, 'quality', (k, p, s) => ({ warmthDb: p.warmthDb + 1.5 * k * s, pitchSemitones: p.pitchSemitones - 0.5 * k * s })],
    [/\b(higher|lighter)\b/, 'quality', (k, p, s) => ({ pitchSemitones: p.pitchSemitones + 0.5 * k * s, warmthDb: p.warmthDb - 1 * k * s })],
    [/\b(clear\w*|crisp\w*|articulat\w*|intelligib\w*|understandable|present)\b/, 'quality', (k, p, s) => ({ presenceDb: p.presenceDb + 2 * k * s, mudDb: Math.min(0, p.mudDb - 1 * k * s) })],
    [/\b(bright\w*|airy|sparkl\w*)\b|\bair\b(?! ?con)/, 'quality', (k, p, s) => ({ airDb: p.airDb + 2 * k * s })],
    [/\b(consistent|steady|steadier|stable|controlled|compress\w*|more even|evenly|evener)\b/, 'quality', (k, p, s) => ({ compEnabled: true, compRatio: p.compRatio + 1 * k * s, compThresholdDb: p.compThresholdDb - 3 * k * s })],
    [/\b(loud\w*)\b/, 'quality', (k, p, s) => ({ targetLufs: p.targetLufs + 2 * k * s })],
    [/\b(quieter)\b/, 'quality', (k, p, s) => ({ targetLufs: p.targetLufs - 2 * k * s })],
  ];
  function intensity(c) {
    if (/\b(slightly|slight|a bit|bit|a little|little|touch|tad|somewhat|subtly)\b/.test(c)) return 0.5;
    if (/\b(much|a lot|lots|very|really|way|significantly|strongly|heavily)\b/.test(c)) return 1.6;
    return 1;
  }
  function podcastChanges(p, analysis, ref) {
    const base = analysis && analysis.ok ? VA.matchToReference(analysis, ref, p) : {
      warmthDb: 2, presenceDb: 2, mudDb: -2, deEssEnabled: true, deEssMaxDb: 6, compEnabled: true, compRatio: 3, compThresholdDb: -24, makeupDb: 3,
      nsEnabled: true, nsAmount: 0.6, loudEnabled: true, targetLufs: ref.integratedLufs, limiterEnabled: true, limiterCeilingDb: ref.truePeakDb,
    };
    // keep any warmth/presence the user already added if it is more than the match suggests
    const out = { ...base };
    for (const k of ['warmthDb', 'presenceDb']) if (p[k] > (out[k] ?? -99)) out[k] = p[k];
    return out;
  }
  function localInterpret(text, profile, analysis, ref) {
    const t = (text || '').toLowerCase().replace(/[’']/g, "'");
    let p = { ...profile }; const touched = new Set(); let matched = false, special = '';
    if (/\b(reset|start over|from scratch|default settings)\b/.test(t)) {
      return { reply: 'I reset the voice to the neutral starting point.', changes: { ...VP.DEFAULTS }, matched: true };
    }
    if (/\b(natural|less processed|subtle|untouched|original sound)\b/.test(t) && !/\b(more processed)\b/.test(t)) {
      for (const k of ['warmthDb', 'presenceDb', 'airDb', 'mudDb', 'pitchSemitones', 'makeupDb']) { p[k] = p[k] / 2; touched.add(k); }
      p.compRatio = 1 + (p.compRatio - 1) / 2; touched.add('compRatio'); matched = true; special = 'natural';
    }
    if (/\b(podcast|professional|broadcast|radio|studio|announcer|presenter)\b/.test(t)) {
      Object.assign(p, podcastChanges(p, analysis, ref)); Object.keys(podcastChanges(p, analysis, ref)).forEach(k => touched.add(k)); matched = true; special = 'podcast';
    }
    const clauses = t.split(/[,.;!?]|\bbut\b|\band\b|\bthen\b|\balso\b/).map(s => s.trim()).filter(Boolean);
    for (const c of clauses) {
      if (/\b(helium|chipmunks?|cartoon voices?)\b/.test(c)) {
        if (!/\b(not|no|without|avoid|remove|stop|don't|do not)\b/.test(c)) {
          p.pitchSemitones = Math.min(3, p.pitchSemitones + 3 * intensity(c));
          touched.add('pitchSemitones'); matched = true; special = 'creative';
        }
        continue;
      }
      const k = intensity(c);
      const neg = /\b(less|not so|not as|too|without|reduce|cut|tone down|remove|decrease|turn down|lower the)\b/.test(c);
      for (const [re, kind, fn] of RULES) {
        if (!re.test(c)) continue;
        if (kind === 'quality' && /\btoo quiet\b/.test(c)) continue;
        const ch = fn(k, p, kind === 'problem' ? 1 : (neg ? -1 : 1));
        Object.assign(p, ch); Object.keys(ch).forEach(x => touched.add(x)); matched = true;
      }
    }
    const changes = {}; for (const key of touched) changes[key] = p[key];
    return { changes, matched, special };
  }
  const OfflineInterpreter = {
    name: 'offline',
    label: 'Offline interpreter (keyword rules)',
    async interpret(text, ctx) {
      const r = localInterpret(text, ctx.profile, ctx.analysis, ctx.reference);
      if (ctx.mode !== 'creative' && (r.special === 'creative' || Math.abs((r.changes.pitchSemitones ?? ctx.profile.pitchSemitones) - ctx.profile.pitchSemitones) > 1.5)) {
        return { reply: 'Select Creative mode to try stronger pitch effects.', changes: {}, rejected: [] };
      }
      if (!r.matched) return { reply: 'I could not map that to a sound change yet. Try words like warmer, clearer, deeper, steadier volume, less background noise, or softer S sounds.', changes: {} };
      return { reply: r.reply || null, changes: r.changes, special: r.special };
    },
  };

  /* ---------------- Claude interpreter ---------------- */
  function schemaText(profile) {
    return Object.entries(VP.SCHEMA).map(([k, s]) => s.type === 'bool'
      ? `${k} (${s.label}): boolean, now ${profile[k]}`
      : `${k} (${s.label}): ${s.min}..${s.max}${s.unit ? ' ' + s.unit : ''}, step ${s.step}, now ${profile[k]}`).join('\n');
  }
  function buildTurns(history, text, ctx) {
    const a = ctx.analysis && ctx.analysis.ok ? ctx.analysis : null;
    const round = (o) => o ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === 'number' ? Math.round(v * 100) / 100 : v])) : o;
    const instructions = [
      'You are the voice-profile generator inside a real-time voice enhancement app. You never edit audio. You edit a parameter set that a DSP chain applies:',
      'input gain -> noise suppression -> speech detection (mutes non-speech) -> high-pass -> EQ (mud 350 Hz, warmth 180 Hz, presence 4 kHz, air 10 kHz shelf) -> pitch -> de-esser -> compressor + makeup -> loudness normalisation -> output gain -> limiter.',
      '', 'PARAMETERS (current values are the starting point):', schemaText(ctx.profile), '',
      'MEASURED USER VOICE: ' + (a ? JSON.stringify(round(a)) : 'not analysed yet'),
      'PRODUCTION REFERENCE (podcast target): ' + JSON.stringify(ctx.reference),
      'PERSONALISED MATCH TO REFERENCE (deterministic suggestion from the measurements): ' + JSON.stringify(a ? VA.matchToReference(a, ctx.reference, ctx.profile) : {}), '',
      'RULES:',
      '- Start from the current values and change only what the request needs. Never reset settings the user did not mention, including ones they set by hand in the Mixer.',
      '- "slightly"/"a bit" means small steps (EQ about 1 dB, ratio about 0.5); no qualifier means moderate (EQ 2-3 dB); "much" means larger.',
      '- Keep the speaker recognisable. Prefer warmth EQ for "deeper". Use pitchSemitones only when the user clearly asks for a deeper/higher voice, and stay within +/-1.5 unless asked for more.',
      '- For "podcast", "professional" or "broadcast", move toward the personalised match above rather than generic numbers.',
      '- Sharp S sounds: de-esser (lower threshold and/or more max reduction). Background, keyboard, paper or fan noise: noise suppression and speech detection. Uneven volume: compressor.',
      '- reply: one or two short sentences in the same language the user wrote in, plain words, no units or jargon (no dB, Hz, LUFS, ratio, compressor, EQ).',
      'Reply with only a JSON object: {"reply": string, "changes": {parameterKey: value}}. Include only keys you change. Use numbers for numeric keys and true/false for booleans.',
    ].join('\n');
    const turns = [{ role: 'user', content: instructions }];
    for (const h of history.slice(-12)) turns.push({ role: h.role, content: h.content });
    turns.push({ role: 'user', content: text });
    return turns;
  }
  function parseAIResponse(obj) {
    if (!obj || typeof obj !== 'object') return { reply: null, ok: {}, rejected: ['response was not a JSON object'] };
    const { ok, rejected } = VP.validateChanges(obj.changes || {});
    return { reply: typeof obj.reply === 'string' ? obj.reply.slice(0, 600) : null, ok, rejected };
  }
  function makeClaudeInterpreter(sample) {
    return {
      name: 'claude', label: 'Claude',
      async interpret(text, ctx) {
        const turns = buildTurns(ctx.history || [], text, ctx);
        const obj = await sample.json(turns, { modelTier: 'quick', cache: false, signal: ctx.signal });
        const r = parseAIResponse(obj);
        return { reply: r.reply, changes: r.ok, rejected: r.rejected };
      },
    };
  }

  root.VoiceAI = { OfflineInterpreter, makeClaudeInterpreter, localInterpret, buildTurns, parseAIResponse, describeDiff, podcastChanges };
})(typeof window !== 'undefined' ? window : globalThis);
