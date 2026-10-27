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
    'ggml-webgpu.cpp': '17a4296ce72ad1acf281f87435dd53f3cb6296578be35675cb374213d41139ac',
    'ggml-webgpu-shader-lib.hpp': '84667b9b4afaebbee6c4ebb287a26256d909895eeefc2814eca01dabfdebc8fa',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '70d5da2e45ef99c0980bf3bb797450ce5c48ce252d0b10ad328468e1a84e9d4c',
    'ggml-webgpu-shader-lib.hpp': '9a5c601cf1b54faa17d63190ab3148c3d78b0d58049662c99555d9957ad884bc',
}
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'ab446e25ba92b0733bbe957b32adab70be2c3973cd8f412fbbcf2dd86818a289',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': 'c938be30a28ea800dda6297dae2a9aee71ddef83f189c077b22bf8c8317d98dc',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-opencl/cl-program-cache.cpp': 'f6a15a96350163de97b2c2feb00e5d675f2a9b69fff26b35e516d695b5877efe',
    'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
    'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
    'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
}
BRIDGE_HEADER_SHA256 = '55923a7e5e1f5b96abbe8185e1499280d9a99c36c5c45887d42e54005d9c0bbc'
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
