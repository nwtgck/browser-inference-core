"""Checked generic F32 GET_ROWS/SUM_ROWS/CLAMP/DIV router-tail fusion."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-router-tail.patch'
PATCH_SHA256 = '55da9404bb6812184ddcbb42447610ca2d78a77d73f5083ef4a6f58700d8e9de'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-router-tail-shaders.hpp'
HEADER_SHA256 = 'a2b8efd2a9617c5685ae5e91e4f61570c519b3f787f51395e4d38e809eb62c52'
INPUTS = {'ggml-webgpu.cpp': 'a53ea652fa4bbaec5c8ba503dcf852c672e433086909846fb55f0dc3ede81bfd',
 'ggml-webgpu-shader-lib.hpp': 'bd8f3e7a39cec314c47fc6753eb044dbc5b4411ad20c195345879e003a6040da'}
OUTPUTS = {'ggml-webgpu.cpp': '17a4296ce72ad1acf281f87435dd53f3cb6296578be35675cb374213d41139ac',
 'ggml-webgpu-shader-lib.hpp': '84667b9b4afaebbee6c4ebb287a26256d909895eeefc2814eca01dabfdebc8fa',
 'ggml-webgpu-router-tail.hpp': '9fc65ad3bf41dd4c44ad77ffcdbd3cf242144fbb4588912351c855e360aab8ed',
 'wgsl-shaders/router_tail.wgsl': '020e4829bed8e5867739a74f891ff82cd110e55fb216e6d13445d296ae8b268e',
 'wgsl-shaders/router_tail_merged.wgsl': '878c1b300af91631bfe77ae3a9c3c2b206080c308e55d39c990c7aa4d2c63ce3'}
CONTRACT_INPUTS = {'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3',
 'ggml/src/ggml.c': '42b238412cda218552812a37329fab95a2f94be6d8bff5139083450d432f796d',
 'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff',
 'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
 'src/llama-graph.cpp': 'a046c32b28de82ce5cbefeb1868cab4b021a6318f226ca8346d35822817188ad',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
 'ggml/src/ggml-webgpu/wgsl-shaders/get_rows.wgsl': '0ad2af306e33260e6e6f3a0082aefd1811037ea3550161f17ab3f17dbc0a60a7',
 'ggml/src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl': 'e2e478cdddec955c3abab6de99afc4b012f47c0b5ae8f828e41bc8a606c88a35',
 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1',
 'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl': 'ec26216dfd3881e6ee118103cf9bc8b2f16c22310dbd8d451442f93e7d70d726'}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('Router tail contract changed: ' + name)
    for name, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Router tail composition input changed: ' + name)
    for name in ('ggml-webgpu-router-tail.hpp', HEADER_NAME,
                 'wgsl-shaders/router_tail.wgsl', 'wgsl-shaders/router_tail_merged.wgsl'):
        if (source / BASE / name).exists() or (work / BASE / name).exists():
            raise ValueError('Router tail would replace unreviewed source: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Router tail patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected router tail output: ' + name)
    shaders = work / 'router-tail-only-wgsl'
    shaders.mkdir()
    for name in ('router_tail.wgsl', 'router_tail_merged.wgsl'):
        shutil.copyfile(work / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected router tail embedded identity')
    return {HEADER_NAME: header, 'ggml-webgpu-router-tail.hpp': work / BASE / 'ggml-webgpu-router-tail.hpp'}
