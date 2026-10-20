/**
 * SeedSeg Inference Worker
 *
 * nnU-Net v2 (ResEnc) fold ensemble with ONNX Runtime Web, following
 * scripts/deploy/seedseg_onnx.py (checked against nnUNetv2_predict):
 * NIfTI parse -> bias field correction -> crop to nonzero -> z-score -> [resample]
 * -> Gaussian sliding window (optional test-time mirroring) per fold -> softmax
 * -> mean over folds -> candidates (components of p > threshold, ranked by mean
 * probability) -> top K markers + count flag.
 */

/* global importScripts, ort, localforage, SeedSegPipeline, nifti */

importScripts('../wasm/ort.min.js');
importScripts('https://cdn.jsdelivr.net/npm/localforage@1.10.0/dist/localforage.min.js');
importScripts('../nifti-js/index.js');
importScripts('modules/inference/nnunet-pipeline.js');

const P = SeedSegPipeline;
let qsmWasm = null;

// ==================== Message helpers ====================

const postProgress = (value, text) => self.postMessage({ type: 'progress', value, text });
const postLog = (message) => self.postMessage({ type: 'log', message });
const postError = (message) => self.postMessage({ type: 'error', message });
const postStageData = (stage, niftiData, description) =>
  self.postMessage({ type: 'stageData', stage, niftiData, description }, [niftiData]);

// ==================== NIfTI ====================

function decompressIfNeeded(data) {
  const bytes = new Uint8Array(data);
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (typeof nifti !== 'undefined' && nifti.isCompressed && nifti.isCompressed(bytes.buffer)) {
      return new Uint8Array(nifti.decompress(bytes.buffer));
    }
    throw new Error('Gzipped NIfTI detected but decompression not available');
  }
  return bytes;
}

/** Voxel -> world (mm) affine rows from the header: sform if set, else qform, else pixdim. */
function headerAffine(view, pix) {
  const qcode = view.getInt16(252, true), scode = view.getInt16(254, true);
  if (scode > 0) {
    const r = (o) => [0, 1, 2, 3].map((i) => view.getFloat32(o + 4 * i, true));
    return [r(280), r(296), r(312)];
  }
  if (qcode > 0) {
    const b = view.getFloat32(256, true), c = view.getFloat32(260, true), d = view.getFloat32(264, true);
    const a = Math.sqrt(Math.max(0, 1 - b * b - c * c - d * d));
    const q = [view.getFloat32(268, true), view.getFloat32(272, true), view.getFloat32(276, true)];
    const qfac = pix[0] < 0 ? -1 : 1;
    const R = [[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
      [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
      [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b]];
    const s = [pix[1], pix[2], pix[3] * qfac];
    return [0, 1, 2].map((i) => [R[i][0] * s[0], R[i][1] * s[1], R[i][2] * s[2], q[i]]);
  }
  return [[pix[1], 0, 0, 0], [0, pix[2], 0, 0], [0, 0, pix[3], 0]];
}

function parseNiftiInput(arrayBuffer) {
  const data = decompressIfNeeded(arrayBuffer);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const dims = [];
  for (let i = 0; i < 8; i++) dims.push(view.getInt16(40 + i * 2, true));
  const [nx, ny, nz] = [dims[1], dims[2], dims[3]];
  const pix = [];
  for (let i = 0; i < 8; i++) pix.push(view.getFloat32(76 + i * 4, true));
  const datatype = view.getInt16(70, true);
  const dataStart = Math.ceil(view.getFloat32(108, true));
  const slope = view.getFloat32(112, true) || 1, inter = view.getFloat32(116, true) || 0;
  const n = nx * ny * nz; // first volume only
  const img = new Float32Array(n);
  const rd = {
    2: (i) => data[dataStart + i], 4: (i) => view.getInt16(dataStart + i * 2, true),
    8: (i) => view.getInt32(dataStart + i * 4, true), 16: (i) => view.getFloat32(dataStart + i * 4, true),
    64: (i) => view.getFloat64(dataStart + i * 8, true), 512: (i) => view.getUint16(dataStart + i * 2, true)
  }[datatype];
  if (!rd) throw new Error(`Unsupported NIfTI datatype: ${datatype}`);
  for (let i = 0; i < n; i++) img[i] = rd(i) * slope + inter;
  const headerBytes = new ArrayBuffer(dataStart);
  new Uint8Array(headerBytes).set(data.slice(0, dataStart));
  return { imageData: img, dims: [nx, ny, nz], voxelSize: [pix[1] || 1, pix[2] || 1, pix[3] || 1],
    affine: headerAffine(view, pix), headerBytes };
}

function createOutputNifti(float32Data, sourceHeader) {
  const headerSize = Math.ceil(new DataView(sourceHeader).getFloat32(108, true));
  const buffer = new ArrayBuffer(headerSize + float32Data.length * 4);
  const v = new DataView(buffer);
  new Uint8Array(buffer).set(new Uint8Array(sourceHeader).slice(0, headerSize));
  v.setInt16(70, 16, true); v.setInt16(72, 32, true); // float32
  v.setInt16(40, 3, true); v.setInt16(48, 1, true);   // 3D
  v.setFloat32(112, 1, true); v.setFloat32(116, 0, true);
  new Float32Array(buffer, headerSize).set(float32Data);
  return buffer;
}

// ==================== Models ====================

async function sha256Hex(buf) {
  const h = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(h)).map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Fetch a model file, caching it in IndexedDB under its SHA-256 (so updates invalidate). */
async function fetchModel(baseUrl, entry, label, progressBase, progressSpan) {
  const key = `seedseg-v2:${entry.sha256}`;
  try {
    const cached = await localforage.getItem(key);
    if (cached && cached.byteLength === entry.bytes) {
      postLog(`Model loaded from cache: ${label}`);
      return cached;
    }
  } catch (e) { /* cache miss */ }
  const url = `${baseUrl}/${entry.file}`;
  postLog(`Downloading ${label} (${(entry.bytes / 1048576).toFixed(0)} MB)...`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch model ${url}: ${response.status} ${response.statusText}`);
  const total = entry.bytes;
  const reader = response.body.getReader();
  const data = new Uint8Array(total);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > total) throw new Error(`${entry.file}: larger than expected`);
    data.set(value, received);
    received += value.length;
    postProgress(progressBase + (received / total) * progressSpan,
      `Downloading ${label} (${(received / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)} MB)`);
  }
  if (received !== total) throw new Error(`${entry.file}: got ${received} bytes, expected ${total}`);
  if ((await sha256Hex(data.buffer)) !== entry.sha256) throw new Error(`${entry.file}: checksum mismatch`);
  try {
    await localforage.setItem(key, data.buffer);
  } catch (e) {
    postLog('Warning: could not cache model (storage full?)');
  }
  return data.buffer;
}

// ==================== Pipeline ====================

async function runInference(config) {
  const { inputData, settings } = config;
  const {
    folds = ['0'], mirror = false, threshold = 0.1, nMarkers = 3, modelBaseUrl, modelDir = 'browser',
    model = 'web', backend = 'auto', biasCorrection = 'makehomogeneous'
  } = settings;
  const tStart = performance.now();
  const timing = {};

  postProgress(0.01, 'Loading model description...');
  const manifest = await (await fetch(`${modelBaseUrl}/seedseg-v2.json`, { cache: 'no-cache' })).json();
  const patch = manifest.patch_size; // [z, y, x]
  const target = manifest.spacing_zyx;
  const cls = manifest.marker_class;

  postProgress(0.02, 'Reading NIfTI...');
  const { imageData, dims: niiDims, voxelSize, affine, headerBytes } = parseNiftiInput(inputData);
  const [nx, ny, nz] = niiDims;
  const dims = [nz, ny, nx];
  const spacing = [voxelSize[2], voxelSize[1], voxelSize[0]];
  postLog(`Volume ${nx} x ${ny} x ${nz}, voxel ${voxelSize.map((v) => v.toFixed(3)).join(' x ')} mm`);

  // 1. Bias field correction. The training images were N4-corrected (SimpleITK); N4 is not
  // available in the browser, so makehomogeneous (QSM.rs) is used instead.
  let vol = imageData;
  if (biasCorrection === 'makehomogeneous') {
    postProgress(0.04, 'Bias field correction...');
    const t = performance.now();
    vol = new Float32Array(qsmWasm.makehomogeneous_wasm(new Float64Array(imageData), nx, ny, nz,
      voxelSize[0], voxelSize[1], voxelSize[2], 7.0, 15));
    timing.biasCorrection = (performance.now() - t) / 1000;
    postLog(`Bias field correction (makehomogeneous): ${timing.biasCorrection.toFixed(1)} s`);
  }

  // 2. nnU-Net preprocessing: crop to nonzero, z-score, resample if needed, pad to patch.
  const box = P.nonzeroBBox(vol, dims);
  const cropped = P.cropBox(vol, dims, box);
  P.zscoreInPlace(cropped.data);
  let work = cropped.data, wdims = cropped.dims;
  const ns = P.newShape(wdims, spacing, target);
  const resampled = ns.some((n, i) => n !== wdims[i]);
  if (resampled) {
    postLog(`Warning: spacing ${spacing.join('x')} differs from the training spacing ${target.join('x')}; ` +
      `resampling ${wdims.join('x')} -> ${ns.join('x')} (linear)`);
    work = P.resampleLinear(work, wdims, ns);
    wdims = ns;
  }
  const padded = P.padToPatch(work, wdims, patch);
  const gaussian = P.gaussianMap(patch, manifest.gaussian.sigma_scale, manifest.gaussian.value_scaling_factor);
  const mirrorAxes = mirror ? manifest.mirror_axes : [];
  const nFlips = P.mirrorSets(mirrorAxes).length;
  const steps = P.windowSteps(padded.dims, patch, manifest.tile_step_size);
  const nTiles = steps[0].length * steps[1].length * steps[2].length;
  postLog(`Sliding window: ${nTiles} patches x ${nFlips} mirrorings x ${folds.length} network(s)`);

  // 3. Folds.
  const Npad = padded.dims[0] * padded.dims[1] * padded.dims[2];
  const ens = new Float32Array(Npad);
  const perFoldSpan = 0.85 / folds.length;
  timing.folds = [];
  let activeBackend = backend === 'wasm' ? 'wasm' : ((typeof navigator.gpu !== 'undefined' && await navigator.gpu.requestAdapter()) ? 'webgpu' : 'wasm');
  postLog(`Backend: ${activeBackend === 'webgpu' ? 'WebGPU (WebAssembly fallback)' : 'WebAssembly'}`);
  for (let fi = 0; fi < folds.length; fi++) {
    const f = String(folds[fi]);
    const base = 0.08 + fi * perFoldSpan;
    const onTile = (i, n) => postProgress(base + perFoldSpan * (0.3 + 0.7 * i / n), `Fold ${f}: patch ${i}/${n}`);
    let result = null;
    // One file for both backends (the graph is rewritten for WebGPU, scripts/deploy/rewrite_for_webgpu.py):
    // WebGPU first, WebAssembly if it is unavailable or fails.
    const entry = manifest.folds[f][model];
    const modelData = await fetchModel(`${modelBaseUrl}/${modelDir}`, entry, `network ${Number(f) + 1}`, base, perFoldSpan * 0.3);
    for (const be of (activeBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'])) {
      postProgress(base + perFoldSpan * 0.3, `Loading network ${Number(f) + 1}...`);
      const t0 = performance.now();
      let session = null;
      try {
        session = await ort.InferenceSession.create(modelData, { executionProviders: [be], graphOptimizationLevel: 'all' });
        const t1 = performance.now();
        const runNet = async (x) => {
          const out = await session.run({ [manifest.input]: new ort.Tensor('float32', x, [1, 1, ...patch]) });
          const t = out[manifest.output];
          const d = t.location && t.location !== 'cpu' ? await t.getData() : t.data;
          if (t.dispose) t.dispose();
          return d;
        };
        const r = await P.slidingWindowLogits(padded.data, padded.dims, patch, runNet,
          { mirrorAxes, gaussian, step: manifest.tile_step_size, onTile });
        await session.release();
        const t2 = performance.now();
        result = r;
        timing.folds.push({ fold: f, backend: be, sessionCreate: (t1 - t0) / 1000, inference: (t2 - t1) / 1000 });
        postLog(`Network ${Number(f) + 1} (${be}): session ${((t1 - t0) / 1000).toFixed(1)} s, inference ${((t2 - t1) / 1000).toFixed(1)} s`);
        break;
      } catch (err) {
        if (session) { try { await session.release(); } catch (e) { /* ignore */ } }
        if (be !== 'webgpu') throw err;
        postLog(`WebGPU failed (${err?.message || err}); using WebAssembly`);
        activeBackend = 'wasm';
      }
    }
    const { logits, C } = result;
    const p = P.softmaxClass(logits, C, Npad, cls);
    for (let i = 0; i < Npad; i++) ens[i] += p[i] / folds.length;
  }

  // 4. Back to the input grid: unpad, (resample back), uncrop (outside the crop: probability 0).
  let pm = P.unpad(ens, padded.dims, wdims, padded.lo);
  if (resampled) pm = P.resampleLinear(pm, wdims, cropped.dims);
  pm = P.uncropBox(pm, dims, box);

  // 5. Candidates.
  postProgress(0.95, 'Selecting markers...');
  const det = P.detect(pm, dims, nMarkers, threshold, voxelSize[0] * voxelSize[1] * voxelSize[2]);
  const world = (c) => [0, 1, 2].map((r) => affine[r][0] * c[0] + affine[r][1] * c[1] + affine[r][2] * c[2] + affine[r][3]);
  for (const m of det.markers.concat(det.others)) m.centroidWorldMm = world(m.centroidVox);
  timing.total = (performance.now() - tStart) / 1000;

  postStageData('probability', createOutputNifti(pm, headerBytes), `Marker probability (mean of ${folds.length} fold(s))`);
  postStageData('markers', createOutputNifti(det.markerMap, headerBytes), `Top ${nMarkers} markers (value = rank)`);
  const { markerMap, ...summary } = det; // eslint-disable-line no-unused-vars
  const usedBackends = [...new Set(timing.folds.map((t) => t.backend))];
  self.postMessage({ type: 'detections', ...summary, folds, mirror, backend: usedBackends.join('+'), timing });
  postLog(`Detected ${det.markers.length}/${nMarkers} markers (${det.nCandidates} candidates)` +
    (det.countFlag ? ': fewer than the implant count, review the image' : ''));
  postLog(`Total time ${timing.total.toFixed(1)} s`);
  postProgress(1.0, 'Complete');
  self.postMessage({ type: 'complete' });
}

// ==================== Message handler ====================

self.onmessage = async (e) => {
  const { type, data } = e.data;
  if (type === 'init') {
    try {
      const threads = (self.crossOriginIsolated && navigator.hardwareConcurrency)
        ? Math.min(navigator.hardwareConcurrency, 16) : 1;
      ort.env.wasm.numThreads = threads;
      ort.env.wasm.wasmPaths = new URL('../wasm/', self.location.href).href;
      localforage.config({ name: 'SeedSegModelCache', storeName: 'models' });
      const baseUrl = self.location.href.replace(/\/js\/.*$/, '');
      qsmWasm = await import(`${baseUrl}/wasm/qsm_wasm.js`);
      await qsmWasm.default(`${baseUrl}/wasm/qsm_wasm_bg.wasm`);
      self.postMessage({ type: 'initialized', threads, webgpu: typeof navigator.gpu !== 'undefined' });
    } catch (error) {
      postError(`Initialization failed: ${error.message}`);
    }
  } else if (type === 'run') {
    try {
      await runInference(data);
    } catch (error) {
      console.error('Inference error:', error);
      postError(error?.message || String(error));
    }
  }
};
