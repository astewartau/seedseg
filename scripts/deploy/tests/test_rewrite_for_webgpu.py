"""rewrite_for_webgpu.py on a tiny graph (pytest; needs onnx + onnxruntime): the rewritten
graph has only the ops the WebGPU provider takes and gives the same output on CPU."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
import rewrite_for_webgpu as rw  # noqa: E402


def tiny_model():
    rng = np.random.default_rng(0)
    init = [numpy_helper.from_array(rng.standard_normal(s).astype(np.float32), n) for n, s in
            [('w1', (3, 1, 1, 3, 3)), ('wt', (3, 2, 1, 2, 2)), ('bt', (2,))]]
    nodes = [helper.make_node('Conv', ['input', 'w1'], ['c'], name='stem', kernel_shape=[1, 3, 3], pads=[0, 1, 1, 0, 1, 1]),
             helper.make_node('AveragePool', ['c'], ['p'], name='pool', kernel_shape=[1, 2, 2], strides=[1, 2, 2]),
             helper.make_node('ConvTranspose', ['p', 'wt', 'bt'], ['logits'], name='up', kernel_shape=[1, 2, 2], strides=[1, 2, 2])]
    g = helper.make_graph(nodes, 'g', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 4, 6])],
                          [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 2, 2, 4, 6])], init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    m.ir_version = 8
    return m


def run(m, x):
    return ort.InferenceSession(m.SerializeToString()).run(None, {'input': x})[0]


def test_rewrite_is_equivalent():
    m = tiny_model()
    x = np.random.default_rng(1).standard_normal((1, 1, 2, 4, 6)).astype(np.float32)
    ref = run(m, x)
    r, n = rw.rewrite(tiny_model())
    onnx.checker.check_model(r)
    ops = [nd.op_type for nd in r.graph.node]
    assert 'ConvTranspose' not in ops and 'AveragePool' not in ops and n == 3
    for nd in r.graph.node:
        if nd.op_type == 'Conv':
            pads = [a for a in nd.attribute if a.name == 'pads']
            assert not pads or len(set(pads[0].ints)) == 1
    np.testing.assert_allclose(run(r, x), ref, atol=1e-5)
    np.testing.assert_allclose(run(rw.to_w16(r), x), ref, atol=1e-2)
