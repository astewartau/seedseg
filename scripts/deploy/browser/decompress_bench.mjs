// In-browser decompression time of compressed model files (headless Chromium, Playwright),
// with DecompressionStream for each format this Chromium supports.
//   node decompress_bench.mjs --url http://localhost:8742/ --file seedseg-v2_fold0.wg16.onnx [--runs 3]
// The server must serve <file> and <file>.{gzip,deflate,br,zstd} (from compress_sizes.py --write-dir).
import { chromium } from 'playwright';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? d : process.argv[i + 1]; };
const url = arg('url', 'http://localhost:8742/'), file = arg('file'), runs = +arg('runs', 3);
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(url);
const res = await page.evaluate(async ({ file, runs }) => {
  const out = { userAgent: navigator.userAgent, results: [] };
  // file suffix -> DecompressionStream format names to try
  const formats = [['', null], ['gzip', ['gzip']], ['deflate', ['deflate']], ['br', ['brotli', 'br']], ['zstd', ['zstd']]];
  for (const [suffix, names] of formats) {
    let fmt = null;
    if (names) {
      for (const n of names) { try { new DecompressionStream(n); fmt = n; break; } catch (e) { /* unsupported */ } }
      if (!fmt) { out.results.push({ suffix, supported: false }); continue; }
    }
    const name = suffix ? `${file}.${suffix}` : file;
    const r = { suffix, format: fmt, supported: true, fetch_s: [], decompress_s: [], total_s: [] };
    for (let i = 0; i < runs; i++) {
      // fetch fully first (network excluded), then decompress from memory
      const t0 = performance.now();
      const buf = await (await fetch(name, { cache: 'no-store' })).arrayBuffer();
      const t1 = performance.now();
      let n = 0;
      if (fmt) {
        const reader = new Blob([buf]).stream().pipeThrough(new DecompressionStream(fmt)).getReader();
        for (;;) { const { done, value } = await reader.read(); if (done) break; n += value.length; }
      } else n = buf.byteLength;
      const t2 = performance.now();
      r.compressed = buf.byteLength; r.bytes = n;
      r.fetch_s.push((t1 - t0) / 1000); r.decompress_s.push((t2 - t1) / 1000); r.total_s.push((t2 - t0) / 1000);
    }
    out.results.push(r);
  }
  return out;
}, { file, runs });
console.log(JSON.stringify(res, null, 1));
await browser.close();
