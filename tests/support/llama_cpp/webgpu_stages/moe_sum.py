"""Source-bound ordered expert-sum overlay; the upstream checkout is read-only."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-moe-expert-sum.patch'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-moe-sum-shaders.hpp'
PATCH_SHA256 = '5ef323cc36e2e89fc940091cf3492f976d962df5c93d4b5f40107d555d8c29ea'
INPUTS = {'ggml-webgpu.cpp': 'ca3f9148143e9ed1629678dc6310fe44f86227d54767bb2e892113fd8cf539b6', 'ggml-webgpu-shader-lib.hpp': '13f49b33e11186acc451d027fcc94601bf976414a2a6cc35d1581fa7cf6c6726'}
OUTPUTS = {'ggml-webgpu-moe-sum.hpp': '2e5d9bc5261bbccbdbcd5ff74a8f28b6997c4493fd8170eb0719179773ff5f6a', 'ggml-webgpu-shader-lib.hpp': '5998639e6a402d7d97f925b1d7aefa5b6230a13a0534c35bf7a9026fd9848a1b', 'ggml-webgpu.cpp': '7384d2885b7a55fbf638e73be60103b72848ccc7d36208e89691e63ed63523d2', 'wgsl-shaders/moe_expert_sum.wgsl': '2ef25c82dfd203c1edace891fd3d2351574a3b85381f6faecd2e63e945477df6'}
CONTRACT_INPUTS = {'ggml/src/ggml-impl.h': '75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd', 'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc', 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa', 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8', 'src/llama-graph.cpp': '77becc9efb8b8eca7ae706dd499d137f694523725c4b849f50cd964bc32b25f8', 'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl': 'ec26216dfd3881e6ee118103cf9bc8b2f16c22310dbd8d451442f93e7d70d726', 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5', 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}
HEADER_SHA256 = 'ebd5796f443f3840784d7d6f4e3a908100885e1c156384970b4d18d2a5209c80'

def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('MoE sum contract changed: ' + name)
    for name, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('MoE sum composition input changed: ' + name)
    for name in ('ggml-webgpu-moe-sum.hpp', 'wgsl-shaders/moe_expert_sum.wgsl'):
        if (source / BASE / name).exists():
            raise ValueError('Upstream MoE sum now exists: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('MoE sum patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected MoE sum output: ' + name)
    shaders = work / 'moe-sum-only-wgsl'
    shaders.mkdir()
    (shaders / 'moe_expert_sum.wgsl').write_bytes((work / BASE / 'wgsl-shaders/moe_expert_sum.wgsl').read_bytes())
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('MoE sum embedding identity changed')
    return {HEADER_NAME: header, 'ggml-webgpu-moe-sum.hpp': work / BASE / 'ggml-webgpu-moe-sum.hpp'}
