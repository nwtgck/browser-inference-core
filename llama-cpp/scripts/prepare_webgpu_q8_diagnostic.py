"""Prepare the optional GPU-profile Q8 diagnostic, leaving normal outputs exact."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-q8-diagnostic.patch'
PATCH_SHA256 = 'a27bdcdbd74dd7c3b39b9bd94075adfda59e2803729dae9ee9d72383b4691ab7'
BASE = Path('ggml/src/ggml-webgpu')
INPUTS = {
    'ggml-webgpu.cpp': '7dcb5ae7d2f60c0010b6e6b2e31232b1efc563cdce3f16fab2ebbfc440c88054',
    'ggml-webgpu-shader-lib.hpp': 'a302e654b7afd4ed1ba39a2031596c1d8cb1ca3ba19533be4f5b40f28e1281e2',
}
OUTPUTS = {
    'ggml-webgpu.cpp': 'c604ef0c3a962bcca567c91a0911f77ae456aad3d8e7d939524a4b73764a98ec',
    'ggml-webgpu-shader-lib.hpp': '629f60c48495fa0c3b1b0bb123f9938100961adac78cc5df232393c220eae171',
}
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-opencl/cl-program-cache.cpp': 'f6a15a96350163de97b2c2feb00e5d675f2a9b69fff26b35e516d695b5877efe',
    'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
}
BRIDGE_HEADER_SHA256 = 'c676709290b2fc562ff19fc91607b9c008560501975372c6f37b2f41e1d143ea'
SHA_HEADER_SHA256 = '2a3a6190dff6717b4f05460a4c2e2f857a88d1224481286461fb43284e0b2ff6'
SHA_FRAGMENT_SHA256 = 'fb6d2ee35d33cac819a5066d32dd3685cd72b0afc136b3b1b6adabf9c6d157a2'
SHA_SOURCE = 'ggml/src/ggml-opencl/cl-program-cache.cpp'
SHA_BEGIN = 'struct sha256_ctx {'
SHA_END = '\nstd::string sha256_hex('


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Q8 diagnostic contract changed: ' + name)
    for name, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q8 diagnostic composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Q8 diagnostic patch changed')
    bridge = patch_root.parent / 'bridge/webgpu-q8-diagnostic.h'
    if tensor_copy.digest(bridge) != BRIDGE_HEADER_SHA256:
        raise ValueError('Q8 diagnostic typed contract changed')
    text = (source / SHA_SOURCE).read_text()
    fragment = text[text.index(SHA_BEGIN):text.index(SHA_END)]
    if hashlib.sha256(fragment.encode()).hexdigest() != SHA_FRAGMENT_SHA256:
        raise ValueError('Q8 diagnostic SHA extraction changed')
    sha = work / 'ggml-webgpu-q8-diagnostic-sha256.hpp'
    sha.write_text('#pragma once\n// SHA-256 (FIPS 180-4), public-domain reference from the pinned upstream\n// ggml/src/ggml-opencl/cl-program-cache.cpp. Mechanically extracted unchanged.\n#include <cstdint>\n#include <cstddef>\n#include <cstring>\nnamespace ggml_webgpu_q8_digest { namespace {\n' + fragment + '\n} }\n')
    if tensor_copy.digest(sha) != SHA_HEADER_SHA256:
        raise ValueError('Q8 diagnostic SHA header changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q8 diagnostic output changed: ' + name)
    header = work / 'ggml-webgpu-q8-diagnostic.h'
    shutil.copyfile(bridge, header)
    return {header.name: header, sha.name: sha}
