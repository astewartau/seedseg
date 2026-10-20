// Drive the web app headless (Playwright Chromium): load a NIfTI, run SeedSeg, save the
// marker probability map and detections. Used for browser parity and timing.
//   node run_app.mjs --url http://localhost:8080/ --input t1.nii --out prefix
//        [--folds 0,1,2,3,4] [--mirror] [--backend wasm|webgpu] [--bias makehomogeneous|none] [--k 3]
// Needs `npm i playwright` and PLAYWRIGHT_BROWSERS_PATH with Chromium installed.
import fs from 'fs';
import os from 'os';
import path from 'path';
import { chromium } from 'playwright';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? d : (process.argv[i + 1] ?? true); };
const url = arg('url', 'http://localhost:8080/'), input = arg('input'), out = arg('out');
const folds = String(arg('folds', '0')).split(','), mirror = process.argv.includes('--mirror');
const backend = arg('backend', 'wasm'), bias = arg('bias', 'makehomogeneous'), k = +arg('k', 3);

const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
  '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader', '--js-flags=--max-old-space-size=8192'] });
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.error('[page]', m.text()); });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
await page.goto(url);
await page.waitForFunction(() => window.app && window.app.inferenceExecutor && window.app.inferenceExecutor.isReady(), null, { timeout: 120000 });
const env = await page.evaluate(() => ({ coi: self.crossOriginIsolated, cores: navigator.hardwareConcurrency, gpu: !!navigator.gpu,
  adapter: navigator.gpu ? !!(1) : false }));
console.log('page', JSON.stringify(env));
// the app sorts dropped NIfTIs into the T1w slot by name
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'seedseg-'));
const named = path.join(tmp, 'T1w_' + path.basename(input));
fs.copyFileSync(input, named);
await page.setInputFiles('#unifiedFiles', named);
await page.waitForFunction(() => !document.getElementById('runSegmentation').disabled, null, { timeout: 60000 });
await page.evaluate(({ folds, mirror, backend, bias, k }) => {
  for (const f of ['0', '1', '2', '3', '4']) document.getElementById(`fold${f}`).checked = folds.includes(f);
  document.getElementById('mirrorTTA').checked = mirror;
  document.getElementById('nMarkers').value = String(k);
  const ex = window.app.inferenceExecutor, run = ex.run.bind(ex);
  ex.run = (cfg) => { cfg.settings.biasCorrection = bias; cfg.settings.backend = backend; return run(cfg); };
  window.__done = null;
  const done = ex.onComplete, fail = ex.onError;
  ex.onComplete = () => { done(); window.__done = 'ok'; };
  ex.onError = (m) => { fail(m); window.__done = 'error: ' + m; };
}, { folds, mirror, backend, bias, k });
const t0 = Date.now();
await page.click('#runSegmentation');
await page.waitForFunction(() => window.__done !== null, null, { timeout: 4 * 3600 * 1000, polling: 2000 });
const wall = (Date.now() - t0) / 1000;
const status = await page.evaluate(() => window.__done);
if (status !== 'ok') { console.error(status); await browser.close(); process.exit(1); }
const det = await page.evaluate(() => window.app.inferenceExecutor.detections);
const b64 = await page.evaluate(async () => {
  const buf = new Uint8Array(await window.app.inferenceExecutor.getResult('probability').file.arrayBuffer());
  let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return btoa(s);
});
fs.writeFileSync(`${out}_marker_prob.nii`, Buffer.from(b64, 'base64'));
fs.writeFileSync(`${out}_detections.json`, JSON.stringify({ ...det, wall_s: wall, env }, null, 1));
console.log(JSON.stringify({ wall_s: wall, timing: det.timing, countFlag: det.countFlag,
  markers: det.markers.map((m) => [m.meanProb.toFixed(4), m.centroidVox.map((v) => v.toFixed(1))]) }));
await browser.close();
