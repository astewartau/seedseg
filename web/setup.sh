#!/bin/bash
# One-time setup: download ONNX Runtime Web and QSM WASM files
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/wasm"

# ONNX Runtime Web
ORT_VERSION="1.30.0"
ORT_BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"

echo "Downloading ONNX Runtime Web v${ORT_VERSION}..."

# JSEP build: the 'webgpu' (with 3D Conv) and 'wasm' execution providers. The newer
# ort.webgpu.min.js build has no 3D Conv on WebGPU.
ORT_FILES=(
  ort.min.js
  ort-wasm-simd-threaded.jsep.mjs
  ort-wasm-simd-threaded.jsep.wasm
)

for f in "${ORT_FILES[@]}"; do
  echo "  $f"
  curl -fsSL -o "$SCRIPT_DIR/wasm/$f" "$ORT_BASE/$f"
done

# QSM WASM (bias field correction from QSMbly)
QSM_WASM_VERSION="v0.9.2"
QSM_BASE="https://github.com/astewartau/qsmbly/releases/download/${QSM_WASM_VERSION}"

echo "Downloading QSM WASM ${QSM_WASM_VERSION}..."

QSM_FILES=(
  qsm_wasm.js
  qsm_wasm_bg.wasm
)

for f in "${QSM_FILES[@]}"; do
  echo "  $f"
  curl -fsSL -o "$SCRIPT_DIR/wasm/$f" "$QSM_BASE/$f"
done

echo "Done. Files saved to wasm/"
