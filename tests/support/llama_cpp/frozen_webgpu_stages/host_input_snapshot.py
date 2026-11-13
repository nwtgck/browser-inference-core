"""Checked optional host snapshot contract for single-copy WebGPU scheduling."""
from pathlib import Path
import shutil
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-host-input-snapshot.patch'
PATCH_SHA256 = 'c652f72f19affbe2b274e6715c248e757ce40eabc93452f407777dd2bbba806a'
INPUT_SHA256 = '328d722b4d7612d2ff1dc26f31b343fa3ec3290ba33cb2dc5d960a035d0dc7b8'
OUTPUT_SHA256 = 'be9fffb8fb8eb1c85aef121a07ffad0a1182eaa7bed2829ba6c9cb7242b1b506'
BACKEND_SOURCE_PATH = 'ggml/src/ggml-backend.cpp'
BACKEND_OUTPUT_SHA256 = 'b9d0104d658d4e79642b9594665663dcbb39d97c1cc82c49f086ad71d60ace8e'
HEADER_PATH = 'ggml/src/ggml-backend-host-snapshot.h'
HEADER_OUTPUT_SHA256 = 'f1133ed5459fb8cfa8f714cb855109f199a33df50ad5fafb4db04839bfc352a9'
CONTRACT_INPUTS = {
    'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
    'ggml/src/ggml-backend-reg.cpp': '1436b46bd1e8355345d454baa4b40069be73e9cb09a3a188d32426ea1e1037fe',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
    'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
    'ggml/src/ggml.c': 'c0e81f0ba55ad76020ada29b01385a7a742c800971e7c974c425a08839750b49',
    'ggml/src/CMakeLists.txt': 'bae77e31bc3498973b435e29681c74e8a28857e84383698ef3b88ca2fb6b900a',
    'src/llama-context.cpp': '4d137ded28efc44c8b471f6c056bee9444a24deb7b42d1e04c64728c25e5c822',
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
                       cwd=work, env=env, check=True)
    for prepared, expected in ((target, OUTPUT_SHA256), (backend, BACKEND_OUTPUT_SHA256),
                               (header, HEADER_OUTPUT_SHA256)):
        if tensor_copy.digest(prepared) != expected:
            raise ValueError('Unexpected host input snapshot output identity: ' + prepared.name)
    return {backend.name: backend, header.name: header}
