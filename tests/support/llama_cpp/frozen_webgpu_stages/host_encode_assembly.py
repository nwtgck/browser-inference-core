"""Reduce synchronous host dispatch assembly allocations; preserve queue ownership."""
from __future__ import annotations
from pathlib import Path
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-host-encode-assembly.patch'
PATCH_SHA256 = '36e80570dce3c369739d406e73955d314c7fb435944e776019ea1666fa8c2252'
INPUT_SHA256 = '823653ae0049fcce555213fa95ab2efa311af5d23771391fc712feb8ed45366d'
OUTPUT_SHA256 = '67940a20bba8d06205ad2f6cf1e909450c4fbb6805b5eba59f574c3576d69779'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Host encode assembly contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Host encode assembly composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Host encode assembly patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected host encode assembly output identity')
