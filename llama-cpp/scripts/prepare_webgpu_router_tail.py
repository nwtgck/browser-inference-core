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
INPUTS = {'ggml-webgpu.cpp': 'd9c3f73f0a430f3ee2fa0e88749cbe56f755eae4f9f6dfa102fc19c6570178ab',
 'ggml-webgpu-shader-lib.hpp': 'a6080cd1f69b12c4514436b7114f81750327e472c52ba502d691f406307f4856'}
OUTPUTS = {'ggml-webgpu.cpp': '4b66eeacddb5927aaed6c67e351b190eb5de3662937a9e8fed8760fdd4a8a27d',
 'ggml-webgpu-shader-lib.hpp': '4ca71dcd2df778ffbe3f1678c525e03a646f68fa8345d14168ac91c16a452656',
 'ggml-webgpu-router-tail.hpp': '9fc65ad3bf41dd4c44ad77ffcdbd3cf242144fbb4588912351c855e360aab8ed',
 'wgsl-shaders/router_tail.wgsl': '020e4829bed8e5867739a74f891ff82cd110e55fb216e6d13445d296ae8b268e',
 'wgsl-shaders/router_tail_merged.wgsl': '878c1b300af91631bfe77ae3a9c3c2b206080c308e55d39c990c7aa4d2c63ce3'}
CONTRACT_INPUTS = {'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
 'ggml/src/ggml.c': '45e90696c4a766d5747f3443f817cd3be11f0110cf93cd16926698c0f49ed083',
 'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
 'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
 'src/llama-graph.cpp': 'b3ddd3f7d99acdebf1be4b547def727907ac122d79bd13aeededdbea2e912df6',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
 'ggml/src/ggml-webgpu/wgsl-shaders/get_rows.wgsl': '89d5cef5182e5ce308d4a0bc77c232301836e7d7bfb0d61e46f0ca775b43640c',
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
