"""Prepare one checked sampler translation unit without changing public getters."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

SOURCE_PATH = 'src/llama-sampler.cpp'
PATCH_NAME = 'llama-sampler-single-sync.patch'
PATCH_SHA256 = 'c5ec122ea02d13a6bdb70deb8db1c46c1ccdf99fbf9f61a0c8036781bdaf6d8c'
INPUT_SHA256 = '6313214bf0a4146036f3d62846299cba4fa9804757c353140a04de52130cf496'
OUTPUT_SHA256 = '3fcfceade964ab4cad60621acca0f69f3eb54272cb2ca6d17f7d8b92c51ee410'
CONTRACT_INPUTS = {
    SOURCE_PATH: INPUT_SHA256,
    'src/llama-sampler.h': 'd39c6bfd6c966acde7bfaeb55f08c176b3d9078bc5e8e8de2eee504fc66c258b',
    'src/llama-context.cpp': '5c7b8268b5d63d329f5b27a05bf3e8a685726b40fad45d9b94e306cf5a1336f6',
    'src/llama-context.h': 'c0ed1f0e8b47d2f001a4a8ba61d01f05b16c2972d693780ac1cc09a6d9a05d88',
    'src/llama-vocab.h': 'b201040378716d004b2ded39165f92c1cffc129a361d8c6a42d8b4a1a2fd65f4',
    'src/llama-model.cpp': '4a05742afc09dc89873e40c8b47a97826f95c4cb90fb32ba48271b4d8f4a14de',
    'src/llama-vocab.cpp': 'ac6fd43eca27c67a213a98e27b87e8ca1f26d8d82a63e79543a06ca64ef10a95',
    'include/llama.h': 'f07ffbabd3fbe9ef21d97b4d4ce8328d9c2a123ccbf5266f04a3146d70ce4f91',
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
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
