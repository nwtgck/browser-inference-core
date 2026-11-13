#!/usr/bin/env python3
"""Prepare and validate the same-device copy translation unit."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from build_tools.project import runtime_root
from build_tools.toolchain import dawn
from build_tools.output import output_directories, preserve_unchanged
from build_tools.source.files import absolute, directory
from build_tools.source.checkout import git
from build_tools.source.patches import patch_environment
import shutil
import subprocess
import tempfile

ROOT = runtime_root('llama-cpp')
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
PATCH_NAME = 'ggml-webgpu-same-device-tensor-copy.patch'
SOURCE_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu.cpp'
REVIEWED_REVISIONS = {
    "9309bf1bcfe8d52b0fb7ef906474db94041a5e0c": {
        "ggml/src/ggml-webgpu/ggml-webgpu.cpp": "7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5",
        "ggml/src/ggml-backend.cpp": "7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e",
        "ggml/include/ggml.h": "1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101",
        "ggml/include/ggml-backend.h": "34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a",
        "ggml/src/ggml-backend-impl.h": "7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3",
        "ggml/src/ggml-webgpu/CMakeLists.txt": "aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a",
        "ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp": "34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d"
    }
}
PATCH_SHA256 = '0a23030dd2504b3c9c7436c77b65ed2000f08c68e17f6e7c2276f2d5737bc8e8'
DAWN_RELEASE = dawn.PINNED_RELEASE
DAWN_HEADERS = dict(dawn.PINNED_HEADERS)
PATCHED_SOURCE_SHA256 = 'af7aab21df671b5a95685ef0d5cdfe79d534844c5eac80bfe965be166809389d'


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked tensor-copy input: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contract_inputs(source: Path) -> None:
    source = directory(source)
    # These remaining historical interface gates are independent of checkout
    # identity. The build driver selects and exports the exact upstream commit.
    for inputs in REVIEWED_REVISIONS.values():
        if all(digest(source / relative) == expected for relative, expected in inputs.items()):
            return
    raise ValueError('Tensor-copy needs semantic review: upstream input changed')


def verify_reviewed_source(source: Path) -> str:
    """Compatibility entry point for callers explicitly inspecting a checkout."""
    source = directory(source)
    if Path(git(source, 'rev-parse', '--show-toplevel').strip()).resolve() != source:
        raise ValueError('Tensor-copy requires a real upstream Git checkout')
    verify_contract_inputs(source)
    return git(source, 'rev-parse', 'HEAD').strip()


def prepare(source: Path, output: Path, patch: Path) -> Path:
    source, output = directory(source), absolute(output)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Tensor-copy overlay must be outside the upstream source tree')
    if digest(patch) != PATCH_SHA256:
        raise ValueError('Tensor-copy patch identity changed; review before building')
    verify_contract_inputs(source)
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (prepared,), \
            tempfile.TemporaryDirectory(prefix='tensor-copy-', dir=prepared.parent) as temporary:
        work = Path(temporary)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        shutil.copyfile(source / SOURCE_PATH, target)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                           cwd=work, env=patch_environment(work), check=True)
        if digest(target) != PATCHED_SOURCE_SHA256:
            raise ValueError('Unexpected patched tensor-copy identity')
        result = output / target.name
        target.replace(prepared / target.name)
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
