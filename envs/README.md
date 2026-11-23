# Python environments

Lock files for the deployment scripts in `scripts/deploy/` (ONNX export, parity, benchmarks), built with [uv](https://docs.astral.sh/uv/).

| Lock file | Used by | Notes |
|---|---|---|
| `seedseg-deploy.lock.txt` | `export_onnx`, `make_w16`, `make_int8`, `logit_parity`, `parity_cpu`, `bench_cpu`, the tests in `scripts/deploy/tests/` | CPU `onnxruntime` 1.31.0, torch 2.6.0+cu124 |
| `seedseg-deploy-gpu.lock.txt` | `parity_gpu` | `onnxruntime-gpu` 1.23.2 (CUDA and TensorRT providers), torch 2.6.0+cu124 |

Both use Python 3.11.9. Every package is pinned with its sha256 hashes, for Linux x86_64 (manylinux_2_28 or newer, so Bunya's Rocky 9 works).

## Rebuild

Run this in a job, not on a login node:

```bash
export UV_CACHE_DIR=$SCR/cache/uv            # any scratch path; safe to delete
export UV_INDEX_STRATEGY=unsafe-best-match   # torch comes from the PyTorch cu124 index, the rest from PyPI
ENV=$SCR/projects/prostate/env/seedseg-deploy   # or seedseg-deploy-gpu
uv venv --python 3.11.9 "$ENV"
uv pip sync --python "$ENV/bin/python" envs/seedseg-deploy.lock.txt
"$ENV/bin/python" -m pytest -q scripts/deploy/tests
```

`uv pip sync` installs exactly the locked set and checks every hash. The index URLs are in the lock files.

## Where the lock files came from

Captured on 23 Nov 2026 with `pip freeze --all` from the venvs on Bunya at `/scratch/user/uqaste15/envs/seedseg-deploy` and `seedseg-deploy-gpu`, then turned into hashed lock files with:

```bash
uv pip compile <freeze>.in -o envs/<name>.lock.txt --generate-hashes \
  --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match --emit-index-url --no-header
```

`pip`, `setuptools` and `wheel` are left out. The pins match the freezes exactly.

Those venvs were made with `--system-site-packages` on top of the conda env `seedseg-nnunet`, so most of each freeze (torch, numpy, scipy, nibabel, pandas and others) came from that base env. The base env was rebuilt with newer package versions on 23 Nov 2026, so these lock files record the package set as it was on that date. They are not the exact versions used for the October ONNX export and parity runs. The lock files are self-contained and don't depend on any conda env.

To change a version, edit the pin, run the compile command again and commit both changes.
