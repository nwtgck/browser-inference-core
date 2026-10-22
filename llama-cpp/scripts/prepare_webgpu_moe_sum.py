"""Source-bound ordered expert-sum overlay; the upstream checkout is read-only."""
from pathlib import Path
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-moe-expert-sum.patch'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-moe-sum-shaders.hpp'
PATCH_SHA256 = '82d9150fd02e84bda7848ab83d7ea2fcb3a32fa1ff37e3cea99fe1cb8c93ccd3'
INPUTS = {'ggml-webgpu.cpp': 'c0ecb208561874a1c0e1f584604e289cd0a2c6673563fa4357982912043b0d75', 'ggml-webgpu-shader-lib.hpp': '0e7176e1b2fc7576ce238516fb2dff437d8376a8ee9694ccda60312492fbdbfa'}
OUTPUTS = {'ggml-webgpu-moe-sum.hpp': '2e5d9bc5261bbccbdbcd5ff74a8f28b6997c4493fd8170eb0719179773ff5f6a', 'ggml-webgpu-shader-lib.hpp': 'f934c06f188621fd5ffbc022b4241dc5c7654d8755a8d6e156e27bda346153f5', 'ggml-webgpu.cpp': 'fc2f93acf5923e823a0d986a1634918248cfa32901d64e4c0aae13f3524737f8', 'wgsl-shaders/moe_expert_sum.wgsl': '2ef25c82dfd203c1edace891fd3d2351574a3b85381f6faecd2e63e945477df6'}
CONTRACT_INPUTS = {'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3', 'ggml/src/ggml.c': '6f839423326ca75319ab0301b4b242e94974b48808888042813f199df98e7482', 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39', 'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0', 'src/llama-graph.cpp': '0f65e40d98be84c1e5e580532b931bfa1864cc3b1f7397e6a7ed21e9afbe3842', 'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl': 'ec26216dfd3881e6ee118103cf9bc8b2f16c22310dbd8d451442f93e7d70d726', 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5', 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}
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
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected MoE sum output: ' + name)
    shaders = work / 'moe-sum-only-wgsl'
    shaders.mkdir()
    (shaders / 'moe_expert_sum.wgsl').write_bytes((work / BASE / 'wgsl-shaders/moe_expert_sum.wgsl').read_bytes())
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('MoE sum embedding identity changed')
    return {HEADER_NAME: header, 'ggml-webgpu-moe-sum.hpp': work / BASE / 'ggml-webgpu-moe-sum.hpp'}
