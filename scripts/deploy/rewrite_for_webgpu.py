#!/usr/bin/env python
"""Make the fold graphs runnable by ONNX Runtime Web's WebGPU provider (JSEP build,
ort.min.js), with exactly equivalent operations:
- 3D ConvTranspose with kernel == stride (no padding/overlap, as in the nnU-Net decoder)
  -> 1x1x1 Conv + Reshape/Transpose (depth-to-space) + bias Add (no 3D ConvTranspose kernel);
- 3D Conv with unequal pads (the 1x3x3 convs, pads 0,1,1) -> Conv with equal pads (1) and a
  Slice removing the extra output slices (the 3D Conv kernel only takes equal pads);
- 3D AveragePool with kernel == stride and no padding (the residual-skip downsampling)
  -> depthwise Conv with weights 1/prod(kernel) (3D pooling only runs in NCHW layout).
Writes {fold}.wg.onnx (fp32) and {fold}.wg16.onnx (fp16-stored weights, fp32 compute) and
records them as "webgpu" / "webgpu16" in seedseg-v2.json; "web" (the file the web app
loads, on WebGPU and WebAssembly) is "webgpu16".
  rewrite_convtranspose.py <dir with seedseg-v2.json> [--check]"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rewrite(m):
    m = shape_inference.infer_shapes(m)
    g = m.graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.input)}
    init = {t.name: t for t in g.initializer}
    nodes, n_rw = [], 0
    for nd in g.node:
        if nd.op_type == 'Conv':
            at = {a.name: helper.get_attribute_value(a) for a in nd.attribute}
            pads = list(at.get('pads', []))
            if pads and len(set(pads)) > 1:
                # Equal pads on all sides (the largest), then slice off the extra output rows
                # (stride-1 axes only). A separate Pad node would be fused back by ORT.
                pm, nsp = max(pads), len(pads) // 2
                assert all(st == 1 for st, lo, hi in zip(at.get('strides', [1] * nsp), pads[:nsp], pads[nsp:])
                           if lo != pm or hi != pm), nd.name
                p = nd.name.replace('/', '_')
                for a_ in nd.attribute:
                    if a_.name == 'pads':
                        del a_.ints[:]; a_.ints.extend([pm] * len(pads))
                axes = [i for i in range(nsp) if pads[i] != pm or pads[nsp + i] != pm]
                g.initializer.extend([
                    numpy_helper.from_array(np.array([pm - pads[i] for i in axes], np.int64), p + '_s0'),
                    numpy_helper.from_array(np.array([-(pm - pads[nsp + i]) or np.iinfo(np.int64).max for i in axes], np.int64), p + '_s1'),
                    numpy_helper.from_array(np.array([i + 2 for i in axes], np.int64), p + '_sax')])
                y = nd.output[0]
                nd.output[0] = p + '_full'
                nodes += [nd, helper.make_node('Slice', [p + '_full', p + '_s0', p + '_s1', p + '_sax'], [y], name=p + '_slice')]
                n_rw += 1
                continue
            nodes.append(nd); continue
        if nd.op_type == 'AveragePool':
            at = {a.name: helper.get_attribute_value(a) for a in nd.attribute}
            k = list(at['kernel_shape'])
            assert k == list(at.get('strides', k)) and not any(at.get('pads', [0])) and not at.get('ceil_mode', 0), nd.name
            c = shapes[nd.input[0]][1]
            assert c, f'{nd.name}: unknown channel count'
            p = nd.name.replace('/', '_')
            g.initializer.append(numpy_helper.from_array(np.full((c, 1, *k), 1.0 / np.prod(k), np.float32), p + '_w'))
            nodes.append(helper.make_node('Conv', [nd.input[0], p + '_w'], list(nd.output), name=p + '_conv',
                                          kernel_shape=k, strides=k, group=c))
            n_rw += 1
            continue
        if nd.op_type != 'ConvTranspose':
            nodes.append(nd); continue
        at = {a.name: helper.get_attribute_value(a) for a in nd.attribute}
        W = numpy_helper.to_array(init[nd.input[1]])  # [Cin, Cout, kz, ky, kx]
        k = list(W.shape[2:]); s = list(at.get('strides', [1] * 3))
        assert k == s and not any(at.get('pads', [0] * 6)) and at.get('group', 1) == 1, nd.name
        cin, cout = W.shape[:2]
        D, H, Wd = shapes[nd.input[0]][2:]
        assert D and H and Wd, f'{nd.name}: unknown spatial shape'
        p = nd.name.replace('/', '_')
        w1 = np.ascontiguousarray(W.transpose(1, 2, 3, 4, 0).reshape(cout * int(np.prod(k)), cin, 1, 1, 1))
        g.initializer.append(numpy_helper.from_array(w1, p + '_w1'))
        sh1 = numpy_helper.from_array(np.array([0, cout, *k, D, H, Wd], np.int64), p + '_sh1')
        sh2 = numpy_helper.from_array(np.array([0, cout, D * k[0], H * k[1], Wd * k[2]], np.int64), p + '_sh2')
        g.initializer.extend([sh1, sh2])
        y = nd.output[0]
        nodes += [helper.make_node('Conv', [nd.input[0], p + '_w1'], [p + '_c'], name=p + '_conv', kernel_shape=[1, 1, 1]),
                  helper.make_node('Reshape', [p + '_c', p + '_sh1'], [p + '_r1'], name=p + '_r1'),
                  # [N, Co, kz, ky, kx, D, H, W] -> [N, Co, D, kz, H, ky, W, kx]
                  helper.make_node('Transpose', [p + '_r1'], [p + '_t'], name=p + '_t', perm=[0, 1, 5, 2, 6, 3, 7, 4]),
                  helper.make_node('Reshape', [p + '_t', p + '_sh2'], [p + '_r2' if len(nd.input) > 2 else y], name=p + '_r2')]
        if len(nd.input) > 2:
            b = numpy_helper.to_array(init[nd.input[2]]).reshape(1, cout, 1, 1, 1)
            g.initializer.append(numpy_helper.from_array(b.astype(np.float32), p + '_b'))
            nodes.append(helper.make_node('Add', [p + '_r2', p + '_b'], [y], name=p + '_add'))
        n_rw += 1
    del g.node[:]; g.node.extend(nodes)
    used = {i for nd in g.node for i in nd.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]; g.initializer.extend(keep)
    del g.value_info[:]
    return m, n_rw


def to_w16(m, min_size=1024):
    g = m.graph; new, casts = [], []
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT and np.prod(t.dims) >= min_size:
            new.append(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), t.name + '__fp16'))
            casts.append(helper.make_node('Cast', [t.name + '__fp16'], [t.name], to=TensorProto.FLOAT, name=t.name + '__cast'))
        else:
            new.append(t)
    del g.initializer[:]; g.initializer.extend(new)
    nodes = casts + list(g.node); del g.node[:]; g.node.extend(nodes)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dir', type=Path); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    mp = a.dir / 'seedseg-v2.json'; meta = json.loads(mp.read_text())
    for f, files in meta['folds'].items():
        src = a.dir / files['fp32']['file']
        m, n = rewrite(onnx.load(str(src)))
        onnx.checker.check_model(m)
        dst = src.with_name(src.name.replace('.onnx', '.wg.onnx')); onnx.save(m, str(dst))
        files['webgpu'] = dict(file=dst.name, bytes=dst.stat().st_size, sha256=sha256(dst))
        m16 = to_w16(m); dst16 = src.with_name(src.name.replace('.onnx', '.wg16.onnx')); onnx.save(m16, str(dst16))
        files['webgpu16'] = dict(file=dst16.name, bytes=dst16.stat().st_size, sha256=sha256(dst16))
        files['web'] = files['webgpu16']  # the browser file (WebGPU and WebAssembly)
        msg = f'fold {f}: {n} nodes rewritten'
        if a.check:
            import onnxruntime as ort
            x = np.random.default_rng(0).standard_normal((1, 1, *meta['patch_size'])).astype(np.float32)
            so = ort.SessionOptions(); so.intra_op_num_threads = 4
            r = ort.InferenceSession(str(src), so).run(None, {'input': x})[0]
            o = ort.InferenceSession(str(dst), so).run(None, {'input': x})[0]
            msg += f', max |dlogit| vs original {np.abs(o - r).max():.2e}'
        print(msg, flush=True)
    mp.write_text(json.dumps(meta, indent=1) + '\n')


if __name__ == '__main__':
    main()
