// Runs the browser engine's self-test suite in Node: `node tests/js/run_selftest.js`
const path = require('path');
const src = path.join(__dirname, '../../web/src/');
for (const f of ['profile', 'dsp', 'analysis', 'ai', 'selftest']) require(src + f + '.js');
let failed = 0;
for (const t of globalThis.SelfTest.TESTS) {
  const t0 = Date.now(); let r;
  try { r = t.run(); } catch (e) { r = { pass: false, detail: 'threw ' + e.message }; }
  console.log(`${r.pass ? 'PASS' : 'FAIL'} ${t.id.padEnd(9)} ${r.detail} (${Date.now() - t0} ms)`);
  if (!r.pass) failed++;
}
console.log(failed ? `${failed} FAILED` : 'ALL PASS');
process.exit(failed ? 1 : 0);
