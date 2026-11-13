"""Prepare one checked sampler translation unit without changing public getters."""
from __future__ import annotations

from pathlib import Path
from build_tools.source.files import absolute, directory, regular_file
import shutil
import subprocess

from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
SOURCE_PATH = 'src/llama-sampler.cpp'
PATCH_NAME = 'llama-sampler-single-sync.patch'
PATCH_SHA256 = 'c5ec122ea02d13a6bdb70deb8db1c46c1ccdf99fbf9f61a0c8036781bdaf6d8c'
DEPENDENCIES = ('src/llama-sampler.cpp', 'src/llama-sampler.h', 'src/llama-context.cpp', 'src/llama-context.h', 'src/llama-vocab.h', 'src/llama-model.cpp', 'src/llama-vocab.cpp', 'include/llama.h', 'ggml/src/ggml-backend.cpp')


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> Path:
    source, work = directory(source), absolute(work)
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('Sampler overlay must be outside the upstream source tree')
    for relative in DEPENDENCIES:
        regular_file(source / relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Single-sync sampler patch identity changed')
    target = work / SOURCE_PATH
    directory(target.parent, create=True)
    if target.exists():
        if regular_file(target).read_bytes() != (source / SOURCE_PATH).read_bytes():
            raise ValueError('Single-sync sampler composition input changed')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / SOURCE_PATH, target)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    return target
