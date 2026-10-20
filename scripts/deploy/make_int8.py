#!/usr/bin/env python
"""Evaluation only (not used by the app): weight-only int8 versions of the WebGPU-compatible
fold graphs (*.wg.onnx). Each Conv weight with >= --min-size elements is stored as int8 with a
symmetric per-output-channel scale and a DequantizeLinear (axis 0) back to fp32 in the graph,
so compute stays fp32. Writes <out>/seedseg-v2_fold{k}.wg8.onnx and <out>/seedseg-v2.json
(a copy of the input metadata with only the "int8" entries).
  make_int8.py <dir with seedseg-v2.json> <out dir> [--min-size 1024]"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def to_int8(m, min_size):
    g = m.graph
    conv_w = {nd.input[1] for nd in g.node if nd.op_type == 'Conv'}
    new_init, dq = [], []
    for t in g.initializer:
        if t.name in conv_w and t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= min_size:
            w = numpy_helper.to_array(t)
            amax = np.abs(w.reshape(w.shape[0], -1)).max(1)
            scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(w / scale.reshape(-1, *[1] * (w.ndim - 1))), -127, 127).astype(np.int8)
            new_init += [numpy_helper.from_array(q, t.name + '__q'),
                         numpy_helper.from_array(scale, t.name + '__s'),
                         numpy_helper.from_array(np.zeros_like(q[:, 0, 0, 0, 0]), t.name + '__z')]
            dq.append(helper.make_node('DequantizeLinear', [t.name + '__q', t.name + '__s', t.name + '__z'], [t.name],
                                       axis=0, name=t.name + '__dq'))
        else:
            new_init.append(t)
    del g.initializer[:]; g.initializer.extend(new_init)
    nodes = dq + list(g.node); del g.node[:]; g.node.extend(nodes)
    return m, len(dq)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('dir', type=Path); ap.add_argument('out', type=Path)
ap.add_argument('--min-size', type=int, default=1024)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
meta = json.loads((a.dir / 'seedseg-v2.json').read_text())
for f, files in meta['folds'].items():
    src = a.dir / files['webgpu']['file']
    m, n = to_int8(onnx.load(str(src)), a.min_size)
    onnx.checker.check_model(m)
    dst = a.out / src.name.replace('.wg.onnx', '.wg8.onnx'); onnx.save(m, str(dst))
    meta['folds'][f] = {'int8': dict(file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst))}
    print(f, n, 'weights quantized', meta['folds'][f]['int8'], flush=True)
(a.out / 'seedseg-v2.json').write_text(json.dumps(meta, indent=1) + '\n')
