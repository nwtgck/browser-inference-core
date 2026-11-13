"""Checked optional host snapshot contract for single-copy WebGPU scheduling."""
from pathlib import Path
import shutil
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-host-input-snapshot.patch'
PATCH_SHA256 = '8be9ad835fbd1e2f0f745f5abb44d8b8a6360e94bafd4593b007d73041fce776'
INPUT_SHA256 = '7fb3a7f254a3160ddf238a9c67fbbabfd2b5d8c4d0473da0c4a3a9b31af59b39'
OUTPUT_SHA256 = '9ec1aac6c9272cb626d74d84de6870f97fbec53c9dd08036e938295babb8ee92'
BACKEND_SOURCE_PATH = 'ggml/src/ggml-backend.cpp'
BACKEND_OUTPUT_SHA256 = 'e464b9bd2fc054efcba4d450070386fbded2905703734664d75470d1741386f5'
HEADER_PATH = 'ggml/src/ggml-backend-host-snapshot.h'
HEADER_OUTPUT_SHA256 = 'f1133ed5459fb8cfa8f714cb855109f199a33df50ad5fafb4db04839bfc352a9'
CONTRACT_INPUTS = {
    'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
    'ggml/src/ggml-backend-reg.cpp': '0a23a627cec97926fb1fbb2b1bf4fa7fed4900aa21b0215302eced797231578b',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
    'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
    'ggml/src/CMakeLists.txt': '232b65acee5fd0f6f0b92a1ea4ff35e31235035d3d4f6b925e4cd200c08b6a48',
    'src/llama-context.cpp': '6b2a143eee8d53cdb559eb18b5612028bdf8545ab832cfdca2c2fd87c9c00309',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Host input snapshot contract changed: ' + relative)
    target = work / tensor_copy.SOURCE_PATH
    if tensor_copy.digest(target) != INPUT_SHA256:
        raise ValueError('Host input snapshot composition input changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Host input snapshot patch identity changed')
    backend = work / BACKEND_SOURCE_PATH
    header = work / HEADER_PATH
    if backend.exists() or header.exists() or (source / HEADER_PATH).exists():
        raise ValueError('Host input snapshot would replace an unreviewed source')
    backend.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / BACKEND_SOURCE_PATH, backend)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for prepared, expected in ((target, OUTPUT_SHA256), (backend, BACKEND_OUTPUT_SHA256),
                               (header, HEADER_OUTPUT_SHA256)):
        if tensor_copy.digest(prepared) != expected:
            raise ValueError('Unexpected host input snapshot output identity: ' + prepared.name)
    return {backend.name: backend, header.name: header}
