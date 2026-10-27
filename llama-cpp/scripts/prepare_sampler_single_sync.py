"""Prepare one checked sampler translation unit without changing public getters."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

SOURCE_PATH = 'src/llama-sampler.cpp'
PATCH_NAME = 'llama-sampler-single-sync.patch'
PATCH_SHA256 = 'c5ec122ea02d13a6bdb70deb8db1c46c1ccdf99fbf9f61a0c8036781bdaf6d8c'
INPUT_SHA256 = 'd0c147129ded606691b0b3e78562bec36f53459446aa7a5bac23bb93a01c86f6'
OUTPUT_SHA256 = '216668acd899e58b69eeb83601678c261deebef47107947eaf419dcdc076f193'
CONTRACT_INPUTS = {
    SOURCE_PATH: INPUT_SHA256,
    'src/llama-sampler.h': 'd39c6bfd6c966acde7bfaeb55f08c176b3d9078bc5e8e8de2eee504fc66c258b',
    'src/llama-context.cpp': '2a25e556cb0d16725e01ce853fb1b76da91dcb41b803ce1a6bacec89bfb94aa5',
    'src/llama-context.h': 'c5123eb0b1b4426737a70a3a616d8ba48b646143b596fa3f42d20ab57b2fb9e5',
    'src/llama-vocab.h': 'b3352ac3ca3e4429c6603e2bbc6e1db7a906d8d3098468dc827711e40d7babb5',
    'src/llama-model.cpp': 'a83d3367793f0aa40a869e0ea067a365a6a6bf4b90ea59460ff827efa1dcdd9a',
    'src/llama-vocab.cpp': 'ff2ec59299af84c51814020169177302003d8c14cdd30f2c44f333359bc17b41',
    'include/llama.h': '961430f89e2553dbcd99e3e6c59826af9ec143f05911520c74f804d986ff3cd2',
    'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> Path:
    source, work = source.resolve(), work.resolve()
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('Sampler overlay must be outside the upstream source tree')
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Single-sync sampler contract changed: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Single-sync sampler patch identity changed')
    target = work / SOURCE_PATH
    if not target.resolve().is_relative_to(work):
        raise ValueError('Sampler output must remain inside the isolated work directory')
    if target.exists():
        if tensor_copy.digest(target) != INPUT_SHA256:
            raise ValueError('Single-sync sampler composition input changed')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / SOURCE_PATH, target)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected single-sync sampler output identity')
    return target
