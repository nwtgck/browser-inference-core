"""Keep browser batched writes on the checked generic scalar fallback."""
from pathlib import Path
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-emscripten-scalar-tensor-writes.patch'
PATCH_SHA256 = '04d43aa1d23d9136c6393e4ea16cb2b90876dc4d65722fd4926016c0f615195e'
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee', 'ggml/src/ggml-webgpu/wgsl-shaders/set_tensor_n.wgsl': '3800dd6c30ea2810b86f3bd4d7c17dd2c828ee727bce817758516b4d1180c0bf', 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8', 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3', 'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a', 'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b', 'src/llama-context.cpp': '6b2a143eee8d53cdb559eb18b5612028bdf8545ab832cfdca2c2fd87c9c00309'}
INPUTS = {False: {'ggml-webgpu.cpp': '2dc724684bdc246e2583e9ed2a7547ffae48bb15e4c400211ae1e90a2fe2dc0a'}, True: {'ggml-webgpu.cpp': 'e6c279f867b6d00e95e17ab5734cbeac50b052cf2ef4a0828e6d19c0941807d1'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '5b89d1700c7fd6e7e1d92adc33a984dbfe5997ca40c1d2a7781e6dfbd8849f7a'}, True: {'ggml-webgpu.cpp': '56d8e3e8eb4eb82323e219edbcfec2d4a9ec91d1a8e490d6b2f880fe3d9c7131'}}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> None:
    for name, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / name) != expected:
            raise ValueError('WebGPU batched-write fallback contract changed: ' + name)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('WebGPU batched-write fallback composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('WebGPU batched-write fallback patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected WebGPU batched-write fallback output: ' + name)
