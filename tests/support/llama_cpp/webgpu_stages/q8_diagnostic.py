"""Prepare the optional GPU-profile Q8 diagnostic, leaving normal outputs exact."""
from pathlib import Path
import hashlib
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-q8-diagnostic.patch'
PATCH_SHA256 = '2d79f1d00ba106bdff3d943b3e5c3dbceef2689c4bf4daa0344a46ab655ad1e8'
BASE = Path('ggml/src/ggml-webgpu')
INPUTS = {
    'ggml-webgpu.cpp': '8ca2a0773d6467acc35e5e56b5f74ef0e6a58379336cfe9d1b0fe8aec80314d0',
    'ggml-webgpu-shader-lib.hpp': '392fb81ddddb2ea569a7d6220704831ee975f96ee2a2131f4d17f2a629ebef56',
}
OUTPUTS = {
    'ggml-webgpu.cpp': '5c96cf065bd1a66eacc7f3b4bb020ccc72a15fd051ac329479d558e9e6a872f5',
    'ggml-webgpu-shader-lib.hpp': '07b2d15e128cc5e0c8afdb5569577900f0afe6a84123b7e4c2e0acc36e816967',
}
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-opencl/cl-program-cache.cpp': 'f6a15a96350163de97b2c2feb00e5d675f2a9b69fff26b35e516d695b5877efe',
    'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
    'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
    'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
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
    bridge = patch_root.parent / 'native/diagnostics/webgpu-q8-diagnostic.h'
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
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Q8 diagnostic output changed: ' + name)
    header = work / 'ggml-webgpu-q8-diagnostic.h'
    shutil.copyfile(bridge, header)
    return {header.name: header, sha.name: sha}
