"""Compose the bounded dispatch-position bind-group cache into WebGPU source."""
from __future__ import annotations

from pathlib import Path
import subprocess

import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-bind-group-cache.patch'
PATCH_SHA256 = 'b1a3a7b61d186446232e7a88ac5d417af239b04891c87b5b72e482957fbba11b'
INPUT_SHA256 = '8d4d6ae43ad7b946fe182fdb22014664158e92d1ee68e4c357c08770f6a35dd2'
OUTPUT_SHA256 = 'ac00909a1706e65dcc11e18e6074dc05c7a9263fd17cdf48884732035de41fe1'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Bind-group cache contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Bind-group cache composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Bind-group cache patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected bind-group cache output identity')
