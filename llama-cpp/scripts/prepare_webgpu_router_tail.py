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
INPUTS = {'ggml-webgpu.cpp': '972928fc0755daad4e24b2da548d2e62329d5ef474ab5ab7e5214365e88d48cf',
 'ggml-webgpu-shader-lib.hpp': '7c5b913454691966279c65cab8953ecb19348cfd4653df94202ca60349d3e48d'}
OUTPUTS = {'ggml-webgpu.cpp': '823653ae0049fcce555213fa95ab2efa311af5d23771391fc712feb8ed45366d',
 'ggml-webgpu-shader-lib.hpp': 'd71825ab9ee4f7d37d1aa492c6ed2f8c01c3f35e0aa0bb8abe740c7824d3bc3d',
 'ggml-webgpu-router-tail.hpp': '9fc65ad3bf41dd4c44ad77ffcdbd3cf242144fbb4588912351c855e360aab8ed',
 'wgsl-shaders/router_tail.wgsl': '020e4829bed8e5867739a74f891ff82cd110e55fb216e6d13445d296ae8b268e',
 'wgsl-shaders/router_tail_merged.wgsl': '878c1b300af91631bfe77ae3a9c3c2b206080c308e55d39c990c7aa4d2c63ce3'}
CONTRACT_INPUTS = {'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
 'ggml/src/ggml.c': 'c0e81f0ba55ad76020ada29b01385a7a742c800971e7c974c425a08839750b49',
 'ggml/src/ggml-impl.h': 'c84c8f9e3c246504602f5cd388ba54e36691e46bf150e4d70ae857bb823c985f',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
 'src/llama-graph.cpp': 'de456d5ff7b60d6404205e3451b4c804229f7fb46023ff20e996f0ad71b752d1',
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
