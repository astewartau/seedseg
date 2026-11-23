# Python environments

Self-contained environments for the v2 work on Bunya, built with [uv](https://docs.astral.sh/uv/) from hashed lock files. They pin the package versions the October 2026 v2 work used (training, inference, ONNX export and parity). None of them depends on a conda env or uses `--system-site-packages`.

| Env | Lock file | Used by |
|---|---|---|
| `seedseg-train` | `seedseg-train.lock.txt` | `scripts/nnunet/` (convert, preprocess, train, predict, `export_probs.py`), `scripts/evaluation/`, `scripts/rerank/`, `scripts/labels/` |
| `seedseg-deploy` | `seedseg-deploy.lock.txt` | `scripts/deploy/` on CPU: `export_onnx`, `make_w16`, `make_int8`, `logit_parity`, `run_parity`/`parity_cpu`, `bench_cpu`, and the tests in `scripts/deploy/tests/` |
| `seedseg-deploy-gpu` | `seedseg-deploy-gpu.lock.txt` | `parity_gpu` (`onnxruntime-gpu`, CUDA and TensorRT providers) |

All three use Python 3.11.9, torch 2.6.0+cu124, torchvision 0.21.0+cu124, nnunetv2 2.8.1, numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2, pandas 3.0.6 and SimpleITK 2.5.6. The deploy envs add `onnx` 1.23.2 and either `onnxruntime` 1.31.0 (CPU) or `onnxruntime-gpu` 1.23.2. Every package is pinned with its sha256 hashes for Linux x86_64 (manylinux_2_28 or newer, so Bunya's Rocky 9 works).

`seedseg-synth` (`scripts/synth/`) is not covered. It needs only numpy, scipy, nibabel, matplotlib, pandas and pydicom, so `seedseg-train` runs it. The v1 scripts that activate the old conda env `prostate39` (`scripts/training/`, some of `scripts/evaluation/` and `scripts/analysis/`) are not covered either.

## Rebuild

Run this in a job, not on a login node. Envs live in `$SCR/projects/prostate/env/<name>`. The 90-day scratch purge may delete them, so rebuild them from the lock files when that happens.

```bash
# uv as a standalone binary (it isn't installed on Bunya)
UVV=0.13.9; UVD=$SCR/cache/tools/uv-$UVV
[ -x $UVD/uv ] || { mkdir -p $UVD && curl -fsSL https://github.com/astral-sh/uv/releases/download/$UVV/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz --strip-components=1 -C $UVD; }
export UV_CACHE_DIR=$SCR/cache/uv                             # safe to delete
export UV_PYTHON_INSTALL_DIR=$SCR/projects/prostate/env/python # the interpreter the envs link to; keep it with them
export UV_INDEX_STRATEGY=unsafe-best-match                    # torch comes from the PyTorch cu124 index, the rest from PyPI

NAME=seedseg-train   # or seedseg-deploy, seedseg-deploy-gpu
ENV=$SCR/projects/prostate/env/$NAME
$UVD/uv venv --python 3.11.9 "$ENV"
$UVD/uv pip sync --python "$ENV/bin/python" envs/$NAME.lock.txt
```

`uv pip sync` installs exactly the locked set and checks every hash. The index URLs are in the lock files. Check the result with `"$ENV/bin/python" -m pytest -q scripts/deploy/tests` (deploy envs) or `"$ENV/bin/nnUNetv2_predict" -h` (train env). Use an env by calling `$ENV/bin/python`, or put `$ENV/bin` first on `PATH`.

## Where the versions come from

Each `.in` file lists the pins with a comment on where they came from. `.lock.txt` is compiled from it.

**`seedseg-train`**, the October `seedseg-nnunet` conda env (built 16 Oct 2026, deleted 23 Nov 2026). It was made in two steps, both recorded in the session log of the agent that built it:

1. `conda create -c conda-forge python=3.11`, which gave Python 3.11.9, then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`. The log shows torch 2.6.0+cu124. torchvision 0.21.0 is the only build that matches it.
2. `pip install nnunetv2 nibabel scipy pandas`. Its full `Successfully installed …` line was logged, so every package from this step is pinned exactly as installed.

Step 1's own dependencies were not logged. They came only from the PyTorch cu124 index, so each one is pinned to the newest version on that index that was published on PyPI before 16 Oct 2026 and supports Python 3.11:

| Package | Pin | Confidence |
|---|---|---|
| `sympy`, `triton`, `nvidia-*` | pinned exactly by torch 2.6.0 | certain |
| `numpy` | 2.4.6 (May 2026) | high: 2.5 needs Python 3.12 |
| `mpmath` | 1.3.0 | certain: sympy 1.13.1 requires < 1.4 |
| `networkx` | 3.6.1 | high: 3.7 needs Python 3.12 |
| `jinja2` 3.1.6, `markupsafe` 3.0.3, `typing-extensions` 4.16.0 | | high: released months earlier |
| `pillow` | 12.3.0 | medium: 13.0.0 came out on 16 Oct, about 18 h after the build |
| `fsspec` | 2026.9.0 | medium: released 4 weeks earlier |
| `filelock` | 4.2.0 | low: released 2 days earlier and may not have been on the PyTorch index yet (4.1.0 otherwise) |
| `packaging` | 26.3 | high: came with conda-forge's Python; the newest release since August |

**`seedseg-deploy` and `seedseg-deploy-gpu`**, the venvs `/scratch/user/uqaste15/envs/seedseg-deploy{,-gpu}` (made on 20 Oct 2026 with `--system-site-packages` on top of `seedseg-nnunet`). They are `seedseg-train` plus the packages installed in each venv itself. Those are still on disk from October, so their pins are exact. The deploy lock files needed no packages beyond these, which matches how the venvs were built.

What was left out:
- `pip` and `setuptools`;
- `zstandard` and `brotli`, which were installed with `--target` into a scratch folder for one size check, not into the venv.

uv's Python 3.11.9 is a python-build-standalone build, not conda-forge's, but the version is the same.

To change a version, edit the `.in` file, compile it again with the command below and commit both files:

```bash
uv pip compile envs/<name>.in -o envs/<name>.lock.txt --generate-hashes \
  --python-version 3.11 --python-platform x86_64-manylinux_2_28 \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match --emit-index-url --no-header
```
