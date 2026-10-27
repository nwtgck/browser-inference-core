"""Checked optional host snapshot contract for single-copy WebGPU scheduling."""
from pathlib import Path
import shutil
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-host-input-snapshot.patch'
PATCH_SHA256 = 'c652f72f19affbe2b274e6715c248e757ce40eabc93452f407777dd2bbba806a'
INPUT_SHA256 = '3de621e1507ba7b7dde3952742ffb9542f536f54e992d75bc257371481daa1bc'
OUTPUT_SHA256 = '0837380d874c570a0a787edc19b21816638b03527075ea8eb9c37c95e3ca5526'
BACKEND_SOURCE_PATH = 'ggml/src/ggml-backend.cpp'
BACKEND_OUTPUT_SHA256 = 'fee60a36d989510e3cdcb4170c2ba62e82df5b150c19b9bd06ebf7a7cd9494b4'
HEADER_PATH = 'ggml/src/ggml-backend-host-snapshot.h'
HEADER_OUTPUT_SHA256 = 'f1133ed5459fb8cfa8f714cb855109f199a33df50ad5fafb4db04839bfc352a9'
CONTRACT_INPUTS = {
    'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
    'ggml/src/ggml-backend-reg.cpp': '1436b46bd1e8355345d454baa4b40069be73e9cb09a3a188d32426ea1e1037fe',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml.c': '42b238412cda218552812a37329fab95a2f94be6d8bff5139083450d432f796d',
    'ggml/src/CMakeLists.txt': 'bae77e31bc3498973b435e29681c74e8a28857e84383698ef3b88ca2fb6b900a',
    'src/llama-context.cpp': '2a25e556cb0d16725e01ce853fb1b76da91dcb41b803ce1a6bacec89bfb94aa5',
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
