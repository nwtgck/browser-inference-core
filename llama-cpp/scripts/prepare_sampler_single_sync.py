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
    'src/llama-context.cpp': '4e61e010041a60f2ed0b17d9b88d9723dc43f6b72e6c81541c55cccea89d93a2',
    'src/llama-context.h': 'c5123eb0b1b4426737a70a3a616d8ba48b646143b596fa3f42d20ab57b2fb9e5',
    'src/llama-vocab.h': '7123e342371b46887d12dc01a7261ff064b651a3c0ce1d99f61c224fd8a55d79',
    'src/llama-model.cpp': '56d2f2d012a4cffa2ac39f8b4b7bc09ad456b8963ed2376a9c42c10757ade9bd',
    'src/llama-vocab.cpp': '9cd7874651e10e0c154eeac9d31e947c5b6cc821bad332ae48cc0bd1db5fdefa',
    'include/llama.h': 'c0854771d3760a41642c4e59429c5b3b9fc774887c59d37eeadc4527681732cd',
    'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
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
