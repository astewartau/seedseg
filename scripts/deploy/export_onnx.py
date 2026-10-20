#!/usr/bin/env python
"""Export trained nnU-Net v2 fold networks to ONNX for the web app and OpenRecon.

Each fold of a trained nnU-Net model folder becomes one ONNX file
``{prefix}_fold{k}.onnx`` that maps a z-scored image patch
``input`` [N, 1, Z, Y, X] (the plans' patch size, axes in SimpleITK order) to raw
logits ``logits`` [N, C, Z, Y, X]. Deep supervision is off. The batch axis is
dynamic. A ``{prefix}.json`` file next to the models records the inference
settings the pipeline (``seedseg_onnx.py``, ``web/js/inference-worker.js``) needs:
patch size, target spacing, mirroring axes, classes, per-file SHA-256 and size.

  python export_onnx.py --model <nnUNet_results/Dataset.../Trainer__Plans__3d_fullres> \
      --folds 0 1 2 3 4 --out <dir> [--fp16] [--check]

``--fp16`` also writes ``{prefix}_fold{k}.fp16.onnx`` (fp16 weights and compute,
fp32 inputs/outputs). ``--check`` compares ONNX Runtime (CPU) logits with PyTorch on
a random patch.
"""
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np
import torch


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--checkpoint', default='checkpoint_final.pth')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--prefix', default='seedseg-v2')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    a.out.mkdir(parents=True, exist_ok=True)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           device=torch.device('cpu'), verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(a.model), use_folds=tuple(a.folds), checkpoint_name=a.checkpoint)
    net = pred.network.eval()
    cm, pm = pred.configuration_manager, pred.plans_manager
    patch = list(cm.patch_size)
    labels = pred.dataset_json['labels']
    n_cls = pred.label_manager.num_segmentation_heads
    meta = dict(
        format='seedseg-onnx-1',
        input='input', output='logits', layout='N,C,Z,Y,X (SimpleITK array order: z slowest, x fastest)',
        patch_size=patch, spacing_zyx=list(cm.spacing), transpose_forward=list(pm.transpose_forward),
        normalization=cm.normalization_schemes[0], use_mask_for_norm=bool(cm.use_mask_for_norm[0]),
        mirror_axes=list(pred.allowed_mirroring_axes or []), tile_step_size=0.5,
        gaussian=dict(sigma_scale=1 / 8, value_scaling_factor=10),
        classes={str(v): k for k, v in labels.items() if v < n_cls}, marker_class=int(labels['marker']),
        candidates=dict(threshold=0.1, connectivity=26, score='mean', top_k=3),
        ensemble='mean of per-fold softmax probabilities', checkpoint=a.checkpoint, folds={})
    x = torch.randn(1, 1, *patch)
    for f, params in zip(a.folds, pred.list_of_parameters):
        net.load_state_dict(params)
        net.eval()
        fn = a.out / f'{a.prefix}_fold{f}.onnx'
        torch.onnx.export(net, x, str(fn), input_names=['input'], output_names=['logits'], opset_version=a.opset,
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, do_constant_folding=True,
                          dynamo=False)
        files = {'fp32': dict(file=fn.name, bytes=fn.stat().st_size, sha256=sha256(fn))}
        if a.fp16:
            import onnx
            from onnxconverter_common import float16
            m16 = float16.convert_float_to_float16(onnx.load(str(fn)), keep_io_types=True)
            f16 = a.out / f'{a.prefix}_fold{f}.fp16.onnx'
            onnx.save(m16, str(f16))
            files['fp16'] = dict(file=f16.name, bytes=f16.stat().st_size, sha256=sha256(f16))
        meta['folds'][str(f)] = files
        if a.check:
            import onnxruntime as ort
            with torch.inference_mode():
                ref = net(x).numpy()
            for k, v in files.items():
                s = ort.InferenceSession(str(a.out / v['file']), providers=['CPUExecutionProvider'])
                o = s.run(None, {'input': x.numpy()})[0]
                print(f'fold {f} {k}: max |onnx - torch| = {np.abs(o - ref).max():.3g} (logit range {ref.min():.1f}..{ref.max():.1f})')
        print(f'fold {f}: {files}')
    (a.out / f'{a.prefix}.json').write_text(json.dumps(meta, indent=1) + '\n')


if __name__ == '__main__':
    main()
