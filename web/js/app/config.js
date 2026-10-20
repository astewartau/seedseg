export const VERSION = '0.2.6';

// Models: nnU-Net v2 ResEnc folds exported to ONNX (scripts/deploy/export_onnx.py), served
// same-origin from ./models (the deploy workflow downloads them from OSF). seedseg-v2.json
// holds the patch size, spacing, file names, sizes and SHA-256 digests; each network is
// downloaded when first used and cached in IndexedDB.
export const MODEL_BASE_URL = './models';

export const FOLDS = ['0', '1', '2', '3', '4'];

// Legacy v1 model (the previously released 4-network 3D U-Net consensus), Advanced settings only.
export const LEGACY_MODEL_BASE_URL = './models/legacy';
export const LEGACY_MODELS = ['seedseg-model-seed42.onnx', 'seedseg-model-seed123.onnx',
  'seedseg-model-seed456.onnx', 'seedseg-model-seed789.onnx'];

export const INFERENCE_DEFAULTS = {
  threshold: 0.1,
  nMarkers: 3,
  folds: ['0'],
  mirror: false,
  model: 'web',      // seedseg-v2.json entry: *.wg16.onnx, fp16-stored weights, fp32 compute (WebGPU and WebAssembly)
  backend: 'auto'    // WebGPU when available, else WebAssembly
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
