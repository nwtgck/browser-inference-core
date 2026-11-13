"""Checked ordered CPU/WebGPU copies through the existing optional copy hook."""
from pathlib import Path
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-async-tensor-copy.patch'
PATCH_SHA256 = '7643f09e513bfaa9aca614644466b5d05ae4419f9edcb3f920c480644d247022'
INPUT_SHA256 = 'b8fa0732191d3c34a7da4d4839683e4c1f7fc0c62a9307fc3481d42311b93b54'
OUTPUT_SHA256 = '0df1049e16051a9d3cc64178271d22fed68bcf7f4fe5459ff73f131bc5340832'
CONTRACT_INPUTS = {
    'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
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
                       cwd=work, env=env, check=True)
    if tensor_copy.digest(target) != OUTPUT_SHA256:
        raise ValueError('Unexpected async tensor-copy output identity')
