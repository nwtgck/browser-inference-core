"""Prepare one checked sampler translation unit without changing public getters."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from tests.support.llama_cpp.webgpu_stages import tensor_copy as tensor_copy

SOURCE_PATH = 'src/llama-sampler.cpp'
PATCH_NAME = 'llama-sampler-single-sync.patch'
PATCH_SHA256 = 'c5ec122ea02d13a6bdb70deb8db1c46c1ccdf99fbf9f61a0c8036781bdaf6d8c'
INPUT_SHA256 = '4ebee0f29e27d356433811f6724959d4d96b89d968f6e38faa3747972bab00d2'
OUTPUT_SHA256 = 'ce96587f6fdb4161a1f55a5ade03c639f3d4a90c994deff54d8f31ae3b61ce3c'
CONTRACT_INPUTS = {
    SOURCE_PATH: INPUT_SHA256,
    'src/llama-sampler.h': 'd39c6bfd6c966acde7bfaeb55f08c176b3d9078bc5e8e8de2eee504fc66c258b',
    'src/llama-context.cpp': '6b2a143eee8d53cdb559eb18b5612028bdf8545ab832cfdca2c2fd87c9c00309',
    'src/llama-graph.cpp': '77becc9efb8b8eca7ae706dd499d137f694523725c4b849f50cd964bc32b25f8',
    'src/llama-graph.h': '607569ebfcd57dc1cf2e107d72e26b837b013c8167d1d912652f427fecb32b55',
    'src/llama-context.h': '287d337d23520c56d3d8bcc31087b34d55b066d3a5fec574b7f44f9f082d7a5e',
    'src/llama-vocab.h': 'b201040378716d004b2ded39165f92c1cffc129a361d8c6a42d8b4a1a2fd65f4',
    'src/llama-model.cpp': 'c8557bc258fc6c23dfd7e70f9ca07e68ebcd41ca34782af4af4aa484c8900ae7',
    'src/llama-vocab.cpp': '0e725f76067a7d4b4da8a5ba975f97490080294f4721f418bf0787d7c5834998',
    'include/llama.h': '19ad09e336dd61317ff816bbf448e9b76c1d925d430191429c17df4679f9a1a4',
    'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
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
