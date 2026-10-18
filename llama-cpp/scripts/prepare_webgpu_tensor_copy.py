#!/usr/bin/env python3
"""Prepare and validate the same-device copy translation unit."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
PATCH_NAME = 'ggml-webgpu-same-device-tensor-copy.patch'
SOURCE_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu.cpp'
REVIEWED_REVISIONS = {
    "620d2225cc436ab0b001e3200e590e3b286ef9ab": {
        "ggml/src/ggml-webgpu/ggml-webgpu.cpp": "995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf",
        "ggml/src/ggml-backend.cpp": "43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff",
        "ggml/include/ggml.h": "475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3",
        "ggml/include/ggml-backend.h": "0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2",
        "ggml/src/ggml-backend-impl.h": "4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7",
        "ggml/src/ggml-webgpu/CMakeLists.txt": "aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a",
        "ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp": "ab446e25ba92b0733bbe957b32adab70be2c3973cd8f412fbbcf2dd86818a289"
    }
}
PATCH_SHA256 = '7e37ea0704f53117b9d3876b34b1ffc58970ce7356385509a989e40d8381e97a'
DAWN_RELEASE = 'v20260908.214631'
DAWN_HEADERS = {
    "webgpu/include/webgpu/webgpu.h": "da1bbcb1c6e0f65d917c4829b465454914b44ea16fa49c3146fc44e606ac989a",
    "webgpu_cpp/include/webgpu/webgpu_cpp.h": "dbad68fcf60e9b9948331bf388cfbe0317e7628ff3bb28246c8b6bb1ce4e9701",
    "webgpu_cpp/include/webgpu/webgpu_cpp_chained_struct.h": "582e62d371f29391d78a7356c634c6b76e1d58b6257a6f5be51701d15797fce1",
    "webgpu_cpp/include/webgpu/webgpu_enum_class_bitmasks.h": "fd436dc17d050e156ac073b6148d90d388e2585de924ae0274b6aed3067c2a96"
}
PATCHED_SOURCE_SHA256 = '62f0421c24eacd78591d5ba26b90d887c82e8a60ac9b5696b5b245acc59dc467'


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked tensor-copy input: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reviewed_source(source: Path) -> str:
    source = source.resolve()
    top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=source, text=True).strip()
    if Path(top).resolve() != source:
        raise ValueError('Tensor-copy requires a real upstream Git checkout')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    # Unrelated upstream commits are compatible when every reviewed contract
    # input is byte-identical. A changed dependency still requires review.
    for inputs in REVIEWED_REVISIONS.values():
        if all(digest(source / relative) == expected for relative, expected in inputs.items()):
            return commit
    raise ValueError('Tensor-copy needs semantic review: upstream input changed')


def prepare(source: Path, output: Path, patch: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Tensor-copy overlay must be outside the upstream source tree')
    if digest(patch) != PATCH_SHA256:
        raise ValueError('Tensor-copy patch identity changed; review before building')
    verify_reviewed_source(source)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tensor-copy-', dir=output) as temporary:
        work = Path(temporary)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        shutil.copyfile(source / SOURCE_PATH, target)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                           cwd=work, check=True, capture_output=True)
        if digest(target) != PATCHED_SOURCE_SHA256:
            raise ValueError('Unexpected patched tensor-copy identity')
        result = output / target.name
        if not result.exists() or result.read_bytes() != target.read_bytes():
            target.replace(result)
        return result


def verify_dawn(package: Path) -> None:
    for relative, expected in DAWN_HEADERS.items():
        if digest(package / relative) != expected:
            raise ValueError('Tensor-copy needs semantic review for Dawn header ' + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dawn-package', type=Path)
    args = parser.parse_args()
    if args.dawn_package is not None:
        verify_dawn(args.dawn_package)
    prepare(args.source, args.output, ROOT / PATCH_DIRECTORY / PATCH_NAME)


if __name__ == '__main__':
    main()
