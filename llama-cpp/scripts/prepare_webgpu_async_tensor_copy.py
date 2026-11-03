"""Checked ordered CPU/WebGPU copies through the existing optional copy hook."""
from pathlib import Path
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-async-tensor-copy.patch'
PATCH_SHA256 = '7643f09e513bfaa9aca614644466b5d05ae4419f9edcb3f920c480644d247022'
INPUT_SHA256 = '9d4295d16c8f988e109d13f6d8bdbea9b8a4fb7aa644dc5ac4cc20b86d9bf78f'
OUTPUT_SHA256 = 'c549daa613064cc778060ba8ea899b3dcae1d2ddee2c006c2137d13ae753ec67'
CONTRACT_INPUTS = {
    'ggml/src/ggml.c': '4b141cb8e781090b9a4c453781a31c9436052cced6018b080774650b7b57ed4c',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> None:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Async tensor-copy contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Async tensor-copy composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Async tensor-copy patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected async tensor-copy output identity')
