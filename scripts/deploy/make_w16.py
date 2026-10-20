#!/usr/bin/env python
"""Store an fp32 ONNX model's large weights as fp16 with a Cast back to fp32 in the graph
("w16"): half the download, fp32 compute on every execution provider (ORT folds the
casts at load time). Also records the file in the models' seedseg-v2.json under "w16".
  make_w16.py <dir with seedseg-v2.json> [--min-size 1024]"""
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


def to_w16(src, dst, min_size):
    m = onnx.load(str(src))
    g = m.graph
    new_init, casts = [], []
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= min_size:
            a = numpy_helper.to_array(t).astype(np.float16)
            new_init.append(numpy_helper.from_array(a, t.name + '__fp16'))
            casts.append(helper.make_node('Cast', [t.name + '__fp16'], [t.name], to=TensorProto.FLOAT,
                                          name=t.name + '__cast'))
        else:
            new_init.append(t)
    del g.initializer[:]
    g.initializer.extend(new_init)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('dir', type=Path)
ap.add_argument('--min-size', type=int, default=1024)
a = ap.parse_args()
meta_p = a.dir / 'seedseg-v2.json'
meta = json.loads(meta_p.read_text())
for f, files in meta['folds'].items():
    src = a.dir / files['fp32']['file']
    dst = src.with_name(src.name.replace('.onnx', '.w16.onnx'))
    to_w16(src, dst, a.min_size)
    files['w16'] = dict(file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst))
    print(f, files['w16'], flush=True)
meta_p.write_text(json.dumps(meta, indent=1) + '\n')
