// Renders the JS engine on its sample voice for the browser/server parity test.
// usage: node tests/js/render_fixture.js <outDir> '<profiles JSON array>' [chunk]
const fs = require('fs'), path = require('path');
const src = path.join(__dirname, '../../web/src/');
for (const f of ['profile', 'dsp']) require(src + f + '.js');
const [outDir, profilesJson, chunkArg] = process.argv.slice(2);
const FS = 48000, chunk = parseInt(chunkArg || '1024', 10);
const sample = DSP.makeSampleVoice(FS);
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, 'input.f32'), Buffer.from(sample.data.buffer));
JSON.parse(profilesJson).forEach((p, i) => {
  const r = DSP.renderOffline(sample.data, FS, { ...VoiceProfile.DEFAULTS, ...p }, chunk);
  fs.writeFileSync(path.join(outDir, `out${i}.f32`), Buffer.from(r.data.buffer));
  fs.writeFileSync(path.join(outDir, `info${i}.json`), JSON.stringify(r.infos.map(x => ({ speech: x.speech, compDb: x.compDb, loudDb: x.loudDb }))));
});
