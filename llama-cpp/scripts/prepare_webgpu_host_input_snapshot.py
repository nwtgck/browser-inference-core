"""Checked optional host snapshot contract for single-copy WebGPU scheduling."""
from pathlib import Path
import shutil
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-host-input-snapshot.patch'
PATCH_SHA256 = 'c652f72f19affbe2b274e6715c248e757ce40eabc93452f407777dd2bbba806a'
INPUT_SHA256 = '363ce73b31868f5676303b5cd6c6e02155802bde61f2918946d3879feeb99fbd'
OUTPUT_SHA256 = 'e294d19c5b912b5d289b9e43c64dcbfe7864ff2fe7b458326a3a526914139c86'
BACKEND_SOURCE_PATH = 'ggml/src/ggml-backend.cpp'
BACKEND_OUTPUT_SHA256 = '0e6909d28384347222a0fa0f566457e4431651ef667e2ded779140143346f821'
HEADER_PATH = 'ggml/src/ggml-backend-host-snapshot.h'
HEADER_OUTPUT_SHA256 = 'f1133ed5459fb8cfa8f714cb855109f199a33df50ad5fafb4db04839bfc352a9'
CONTRACT_INPUTS = {
    'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
    'ggml/src/ggml-backend-reg.cpp': '1436b46bd1e8355345d454baa4b40069be73e9cb09a3a188d32426ea1e1037fe',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml.c': '45e90696c4a766d5747f3443f817cd3be11f0110cf93cd16926698c0f49ed083',
    'ggml/src/CMakeLists.txt': 'bae77e31bc3498973b435e29681c74e8a28857e84383698ef3b88ca2fb6b900a',
    'src/llama-context.cpp': 'f76532e7d7c510426bb4f96f1c1bcf4ca5321790b07b56ce9446d793feef8d35',
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
