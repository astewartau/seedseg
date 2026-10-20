"""Wiring tests for seedseg_onnx.py with a tiny fake network (pytest; needs onnx + onnxruntime).
The numerics are checked against nnUNetv2_predict by compare_parity.py, not here."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import seedseg_onnx as so  # noqa: E402

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


@pytest.fixture
def model_dir(tmp_path):
    """1x1x1 conv: logits (background, marker, prostate) = (0, x - 3, 0)."""
    w = numpy_helper.from_array(np.array([0, 1, 0], np.float32).reshape(3, 1, 1, 1, 1), 'w')
    b = numpy_helper.from_array(np.array([0, -3, 0], np.float32), 'b')
    node = helper.make_node('Conv', ['input', 'w', 'b'], ['logits'])
    g = helper.make_graph([node], 'g', [helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 1, 4, 8, 8])],
                          [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['N', 3, 4, 8, 8])], [w, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
    m.ir_version = 8
    onnx.save(m, str(tmp_path / 'f0.onnx'))
    meta = dict(patch_size=[4, 8, 8], spacing_zyx=[2.0, 0.6875, 0.6875], mirror_axes=[0, 1, 2], tile_step_size=0.5,
                gaussian=dict(sigma_scale=1 / 8, value_scaling_factor=10), marker_class=1,
                candidates=dict(threshold=0.1), folds={'0': {'fp32': {'file': 'f0.onnx'}}})
    (tmp_path / 'seedseg-v2.json').write_text(json.dumps(meta))
    return tmp_path


def test_predict_and_detect(model_dir):
    vol = np.full((6, 12, 10), -2.0, np.float32)
    vol[:, :, 0] = 0  # zero border is cropped and reported as background
    vol[2:4, 5:7, 4:6] = 6.0  # one bright blob
    m = so.SeedSegONNX(model_dir, mirror=True, threads=1, log=None)
    assert m.folds == ['0'] and len(so.mirror_sets(m.mirror_axes)) == 8
    p = m.predict(vol, spacing_zyx=(2.0, 0.6875, 0.6875))
    assert p.shape == (3, 6, 12, 10)
    np.testing.assert_allclose(p.sum(0), 1, atol=1e-5)
    assert np.all(p[0, :, :, 0] == 1)
    d = so.detect(p[1], zooms=(2.0, 0.6875, 0.6875), k=3)
    assert d['n_candidates'] == 1 and d['count_flag']
    np.testing.assert_allclose(d['markers'][0]['centroid_vox'], [2.5, 5.5, 4.5], atol=0.2)
    mask = so.marker_mask(p[1], d)
    assert mask.max() == 1 and mask[2, 5, 4] == 1


def test_steps_and_gaussian():
    assert so.window_steps((40, 320, 320), (32, 256, 224)) == [[0, 8], [0, 64], [0, 96]]
    g = so.gaussian_map((4, 8, 8))
    assert g.max() == pytest.approx(10) and g.min() > 0
