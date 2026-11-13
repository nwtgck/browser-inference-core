"""Checked generic F32 GET_ROWS/SUM_ROWS/CLAMP/DIV router-tail fusion."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-router-tail.patch'
PATCH_SHA256 = '55da9404bb6812184ddcbb42447610ca2d78a77d73f5083ef4a6f58700d8e9de'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-router-tail-shaders.hpp'
HEADER_SHA256 = 'a2b8efd2a9617c5685ae5e91e4f61570c519b3f787f51395e4d38e809eb62c52'
INPUTS = {'ggml-webgpu.cpp': '5feda13cfe57aedfecf3a25a4b2f61757860fa22692de214e3c8f9ec3ef54800',
 'ggml-webgpu-shader-lib.hpp': '3ba390f7a883fa847c66a32c62fdb30d76887f222fd8bb94f5359c8e9c320404'}
OUTPUTS = {'ggml-webgpu.cpp': '36a2bf151a18cc4b8d56547811d7aae6945462d01476ca9be3902b1a3e7078e3',
 'ggml-webgpu-shader-lib.hpp': 'df28317ed56afbdf2a1d0e57a020d9602dd90c5aaed2d463037d77454ef9a57c',
 'ggml-webgpu-router-tail.hpp': '9fc65ad3bf41dd4c44ad77ffcdbd3cf242144fbb4588912351c855e360aab8ed',
 'wgsl-shaders/router_tail.wgsl': '020e4829bed8e5867739a74f891ff82cd110e55fb216e6d13445d296ae8b268e',
 'wgsl-shaders/router_tail_merged.wgsl': '878c1b300af91631bfe77ae3a9c3c2b206080c308e55d39c990c7aa4d2c63ce3'}
CONTRACT_INPUTS = {'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
 'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
 'ggml/src/ggml-impl.h': '75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd',
 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
 'src/llama-graph.cpp': '77becc9efb8b8eca7ae706dd499d137f694523725c4b849f50cd964bc32b25f8',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
 'ggml/src/ggml-webgpu/wgsl-shaders/get_rows.wgsl': '89d5cef5182e5ce308d4a0bc77c232301836e7d7bfb0d61e46f0ca775b43640c',
 'ggml/src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl': '1755bf4cc3e763df644e4c5327a752c7af606960f4ac509c5919cfef40e86625',
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
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected router tail output: ' + name)
    shaders = work / 'router-tail-only-wgsl'
    shaders.mkdir()
    for name in ('router_tail.wgsl', 'router_tail_merged.wgsl'):
        shutil.copyfile(work / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected router tail embedded identity')
    return {HEADER_NAME: header, 'ggml-webgpu-router-tail.hpp': work / BASE / 'ggml-webgpu-router-tail.hpp'}
