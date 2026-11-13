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
    "b1b770d90d94cad9e9ea5b993bdbc383a522fb28": {
        "ggml/src/ggml-webgpu/ggml-webgpu.cpp": "c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee",
        "ggml/src/ggml-backend.cpp": "2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8",
        "ggml/include/ggml.h": "bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b",
        "ggml/include/ggml-backend.h": "34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a",
        "ggml/src/ggml-backend-impl.h": "7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3",
        "ggml/src/ggml-webgpu/CMakeLists.txt": "aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a",
        "ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp": "a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17"
    }
}
PATCH_SHA256 = 'e7a9cf739fba7c75e38590f18d9b04f3520835a70a0d66e92abf117bb783c939'
DAWN_RELEASE = dawn.PINNED_RELEASE
DAWN_HEADERS = dict(dawn.PINNED_HEADERS)
PATCHED_SOURCE_SHA256 = 'aec98ba4fb3996bb944de8554370ce2cc487dcc87be2f2f004a0eb0a78f20742'


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
