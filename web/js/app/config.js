export const VERSION = '0.2.6';

// Models: nnU-Net v2 ResEnc folds exported to ONNX (scripts/deploy/export_onnx.py).
// MODEL_BASE_URL holds seedseg-v2.json (patch size, spacing, file names, sizes, SHA-256)
// and the fold files; each fold is downloaded when first used and cached in IndexedDB.
// Model files: one place to fill in. MODEL_FILE_URLS maps a file name from seedseg-v2.json
// (and 'seedseg-v2.json' itself) to a full download URL (e.g. an OSF file URL); files not
// listed are fetched from MODEL_BASE_URL/<file name>.
export const MODEL_BASE_URL = './models';
export const MODEL_FILE_URLS = {
  // 'seedseg-v2.json': 'https://files.au-1.osf.io/v1/resources/f7hjv/providers/osfstorage/<id>',
  // 'seedseg-v2_fold0.wg16.onnx': '...', 'seedseg-v2_fold0.w16.onnx': '...', ...
};

export const FOLDS = ['0', '1', '2', '3', '4'];

export const INFERENCE_DEFAULTS = {
  threshold: 0.1,
  nMarkers: 3,
  folds: ['0'],
  mirror: false,
  precision: 'w16',   // fp16-stored weights, fp32 compute (half the download of fp32)
  backend: 'auto'     // WebGPU (rewritten graph, *.wg16.onnx) when available, else WebAssembly
};

export const VIEWER_CONFIG = {
  loadingText: "",
  dragToMeasure: false,
  isColorbar: false,
  textHeight: 0.03,
  show3Dcrosshair: false,
  crosshairColor: [0.23, 0.51, 0.96, 1.0],
  crosshairWidth: 0.75
};

export const PROGRESS_CONFIG = {
  animationSpeed: 0.5
};

export const STAGE_NAMES = {
  'input': 'Input T1w',
  'probability': 'Probability',
  'markers': 'Markers'
};

export const CACHE_CONFIG = {
  name: 'SeedSegModelCache',
  storeName: 'models',
  maxSizeMB: 1100
};

if (typeof self !== 'undefined') self.SeedSegConfig = { VERSION, MODEL_BASE_URL, FOLDS, INFERENCE_DEFAULTS, VIEWER_CONFIG, PROGRESS_CONFIG, STAGE_NAMES, CACHE_CONFIG };
