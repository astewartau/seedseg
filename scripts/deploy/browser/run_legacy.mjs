// Drive the web app headless with the legacy v1 model (Advanced settings) on one NIfTI and
// report whether it completes and which stages it produced (wiring check of the legacy path).
//   node run_legacy.mjs --url http://localhost:8080/ --input t1.nii
import fs from 'fs';
import os from 'os';
import path from 'path';
import { chromium } from 'playwright';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? d : process.argv[i + 1]; };
const url = arg('url', 'http://localhost:8080/'), input = arg('input');
const browser = await chromium.launch({ args: ['--js-flags=--max-old-space-size=8192'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
await page.goto(url);
await page.waitForFunction(() => window.app && window.app.legacyExecutor, null, { timeout: 120000 });
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'seedseg-'));
const named = path.join(tmp, 'T1w_' + path.basename(input));
fs.copyFileSync(input, named);
await page.setInputFiles('#unifiedFiles', named);
await page.waitForFunction(() => !document.getElementById('runSegmentation').disabled, null, { timeout: 60000 });
await page.evaluate(() => {
  const cb = document.getElementById('legacyModel');
  cb.checked = true; cb.dispatchEvent(new Event('change'));
  const ex = window.app.legacyExecutor;
  window.__done = null;
  const done = ex.onComplete, fail = ex.onError;
  ex.onComplete = () => { done(); window.__done = 'ok'; };
  ex.onError = (m) => { fail(m); window.__done = 'error: ' + m; };
});
const t0 = Date.now();
await page.click('#runSegmentation');
await page.waitForFunction(() => window.__done !== null, null, { timeout: 3600 * 1000, polling: 2000 });
const r = await page.evaluate(() => ({
  status: window.__done, stages: window.app.legacyExecutor.stageOrder,
  foldsDisabled: document.getElementById('fold0').disabled,
  panel: document.getElementById('detectionsPanel').textContent.trim()
}));
console.log(JSON.stringify({ ...r, wall_s: (Date.now() - t0) / 1000 }));
await browser.close();
process.exit(r.status === 'ok' ? 0 : 1);
