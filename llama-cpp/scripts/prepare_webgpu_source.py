#!/usr/bin/env python3
"""Compose explicitly enabled WebGPU changes into one source-bound translation unit."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import prepare_webgpu_tensor_copy as tensor_copy

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = tensor_copy.SOURCE_PATH
PATCH_DIRECTORY = tensor_copy.PATCH_DIRECTORY
PARAM_PATCH_NAME = 'ggml-webgpu-batch-param-uploads.patch'
PARAM_PATCH_SHA256 = 'f1fee64844addb636621412ef7227d6d5fd3ac4210cf7e001689550a321a8a26'
OUTPUT_SHA256 = {
    (False, False): 'a37cb03d2d85b3ecdcd203dd515f35f22b1ca895934b2aab17066d575299ba84',
    (True, False): '0100a766829adeb06df66664646ddd2bb0f50763272f99fa0ea2f5deafda4dea',
    (False, True): '8cf2c5c2a5021518e7af3fd59c15e42e48591f35db4d61345c7627a4e6a50725',
    (True, True): 'd1fd96c0113ac2ffd60c6175fbe9ab8115a0a58d5c17bfd209a5a64f288a1f56',
}


def prepare(source: Path, output: Path, *, copy: bool, batch: bool, patch_root: Path = ROOT / PATCH_DIRECTORY) -> Path:
    if type(copy) is not bool or type(batch) is not bool:
        raise ValueError('WebGPU source options must be explicit booleans')
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('WebGPU overlay must be outside the upstream source tree')
    tensor_copy.verify_reviewed_source(source)
    selected = []
    if copy:
        selected.append((patch_root / tensor_copy.PATCH_NAME, tensor_copy.PATCH_SHA256))
    if batch:
        selected.append((patch_root / PARAM_PATCH_NAME, PARAM_PATCH_SHA256))
    for patch, expected in selected:
        if tensor_copy.digest(patch) != expected:
            raise ValueError('WebGPU patch identity changed: ' + patch.name)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='webgpu-source-', dir=output) as temporary:
        work = Path(temporary)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        shutil.copyfile(source / SOURCE_PATH, target)
        for patch, _ in selected:
            for flags in (['--check'], []):
                subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                               cwd=work, check=True, capture_output=True)
        if tensor_copy.digest(target) != OUTPUT_SHA256[(copy, batch)]:
            raise ValueError('Unexpected combined WebGPU source identity')
        result = output / target.name
        if not result.exists() or result.read_bytes() != target.read_bytes():
            target.replace(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensor-copy', choices=('ON', 'OFF'), required=True)
    parser.add_argument('--param-upload-batching', choices=('ON', 'OFF'), required=True)
    parser.add_argument('--dawn-package', type=Path, required=True)
    args = parser.parse_args()
    tensor_copy.verify_dawn(args.dawn_package)
    prepare(args.source, args.output, copy=args.tensor_copy == 'ON', batch=args.param_upload_batching == 'ON')


if __name__ == '__main__':
    main()
