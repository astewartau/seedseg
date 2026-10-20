#!/usr/bin/env python
"""Evaluation only: lossless compressed sizes of model files (gzip -9, zlib/deflate -9,
brotli 11, zstd 19) and, with --write-dir, the compressed copies of the first file for the
in-browser decompression timing (scripts/deploy/browser/decompress_bench.mjs).
Needs the `brotli` and `zstandard` Python packages.
  compress_sizes.py <file>... --out <json> [--write-dir <dir>] [--jobs N]"""
import argparse, gzip, json, os, time, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def one(job):
    path, method, write_dir = job
    import brotli, zstandard
    data = Path(path).read_bytes()
    t = time.time()
    if method == 'gzip':
        c = gzip.compress(data, 9, mtime=0)
    elif method == 'deflate':
        c = zlib.compress(data, 9)  # zlib-wrapped deflate (DecompressionStream('deflate'))
    elif method == 'br':
        c = brotli.compress(data, quality=11, lgwin=24)
    elif method == 'zstd':
        c = zstandard.ZstdCompressor(level=19, threads=-1).compress(data)
    dt = time.time() - t
    if write_dir:
        (Path(write_dir) / (Path(path).name + '.' + method)).write_bytes(c)
    return dict(file=Path(path).name, method=method, bytes=len(data), compressed=len(c),
                ratio=len(c) / len(data), compress_s=dt)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('files', nargs='+')
ap.add_argument('--out', required=True, type=Path)
ap.add_argument('--write-dir')
ap.add_argument('--methods', default='gzip,deflate,br,zstd')
ap.add_argument('--jobs', type=int, default=len(os.sched_getaffinity(0)))
a = ap.parse_args()
if a.write_dir:
    os.makedirs(a.write_dir, exist_ok=True)
jobs = [(f, m, a.write_dir if i == 0 else None) for i, f in enumerate(a.files) for m in a.methods.split(',')]
with ProcessPoolExecutor(a.jobs) as ex:
    res = list(ex.map(one, jobs))
for r in res:
    print(f"{r['file']:40s} {r['method']:8s} {r['bytes'] / 1e6:8.1f} MB -> {r['compressed'] / 1e6:8.1f} MB "
          f"({100 * r['ratio']:.1f}%), {r['compress_s']:.0f} s", flush=True)
a.out.write_text(json.dumps(res, indent=1))
