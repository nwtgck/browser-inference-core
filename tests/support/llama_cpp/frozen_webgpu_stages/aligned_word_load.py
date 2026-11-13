"""Avoid the unused following-word read in aligned shared byte loaders."""
from dataclasses import dataclass
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-aligned-word-load.patch'
INTEGRATION_PATCH_NAME = 'ggml-webgpu-aligned-word-load-integration.patch'
PATCH_SHA256 = '33283e68a5bdffeb616597a5c19d973c03528bea8ac85d36f8e6259c705686bb'
SOURCE_PATH = 'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl'
TEMPLATE_INPUT_SHA256 = '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555'
TEMPLATE_OUTPUT_SHA256 = 'cf3d4d411353796bfb1b100ccb9822839f070c705e0f4430ebd78a9c7db6d85e'
HEADER_NAME = 'ggml-webgpu-aligned-word-load-shaders.hpp'
INPUTS = {False: {'ggml-webgpu-shader-lib.hpp': '7c10f05f5e67be55f5435026aeaf6ed19b89762600318ceae223f846657e9bdb',
         'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76'},
 True: {'ggml-webgpu-shader-lib.hpp': 'ba830e5dda312af69d63320d1297c652a9c01b55a3ddc95cecbf02204d40d6c6',
        'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'e90bac8b868cd3674c69903cc9bf5bac1327a236fb8984d50c340f2c6d73a67d',
         'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76'},
 True: {'ggml-webgpu-shader-lib.hpp': '9959ab22cb286927cbc3c289914cbfc0c81047324d5a7d17f37cb3c9114e9371',
        'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268'}}
@dataclass(frozen=True)
class DedicatedShader:
    directory: Path
    symbol: str


DEDICATED_SHADERS = {
    'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': DedicatedShader(Path('f32-matvec-epilogue-only-wgsl'), 'mul_mat_vec_f32_epilogue'),
    'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': DedicatedShader(Path('iq3s-row-split-source') / BASE / 'wgsl-shaders', 'iq3s_id_subgroup'),
    'ggml-webgpu-mul-mat-id-group-shaders.hpp': DedicatedShader(Path('moe-group-only-wgsl'), 'mul_mat_id_group'),
    'ggml-webgpu-q8-subgroup-shaders.hpp': DedicatedShader(Path('q8-subgroup-only-wgsl'), 'q8_subgroup_matvec'),
    'ggml-webgpu-quant-half-pairs-shaders.hpp': DedicatedShader(Path('quant-half-pairs-only-wgsl'), 'mul_mat_reg_tile_half_pairs'),
}
# These dedicated shader families do not include the shared byte loaders. The
# explicit inventory makes a newly added shader family a composition decision.
INDEPENDENT_EMBEDDINGS = frozenset({
    'ggml-webgpu-hc-shaders.hpp', 'ggml-webgpu-moe-sum-shaders.hpp',
    'ggml-webgpu-scale-unary-shaders.hpp', 'ggml-webgpu-rms-scale-shaders.hpp',
    'ggml-webgpu-ssm-conv-silu-shaders.hpp', 'ggml-webgpu-router-tail-shaders.hpp',
    'ggml-webgpu-mul-mat-id-gather-logical32-shaders.hpp',
})


HELPERS = ('load_u32_at_src', 'load_u32_at_src0')
TEMPLATE_OUTPUT_BYTES = 74306

def definition(text: str, name: str) -> str:
    start = text.index('fn ' + name + '(')
    at = text.index('{', start) + 1
    depth = 1
    while depth:
        if at >= len(text):
            raise ValueError('Unterminated shared word helper: ' + name)
        depth += (text[at] == '{') - (text[at] == '}')
        at += 1
    return text[start:at]




def generate_canonical(source: Path, patched_template: Path, output: Path, *, inputs: Path | None = None) -> None:
    generate_wgsl(source / BASE / 'wgsl-shaders' if inputs is None else inputs,
                  patched_template, output, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *,
            gpu_profile: bool, headers: dict[str, Path]) -> dict[str, Path]:
    if tensor_copy.digest(source / SOURCE_PATH) != TEMPLATE_INPUT_SHA256:
        raise ValueError('Shared word-loader template identity changed')
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Shared word-loader composition input changed: ' + name)
    if (work / HEADER_NAME).exists() or (source / BASE / HEADER_NAME).exists():
        raise ValueError('Shared word-loader canonical embedding already exists')
    for name in DEDICATED_SHADERS:
        if name not in headers or not headers[name].is_file():
            raise ValueError('Missing shared word-loader dedicated input: ' + name)
    unknown = {name for name in headers if name.endswith('-shaders.hpp')} - set(DEDICATED_SHADERS) - INDEPENDENT_EMBEDDINGS
    if unknown:
        raise ValueError('Unclassified shared word-loader shader consumer: ' + ', '.join(sorted(unknown)))
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Shared word-loader patch identity changed')
    isolated = work / 'aligned-word-load-template'
    target = isolated / SOURCE_PATH
    target.parent.mkdir(parents=True)
    original = (source / SOURCE_PATH).read_text()
    target.write_text(original)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=isolated, env=env, check=True)
    if tensor_copy.digest(target) != TEMPLATE_OUTPUT_SHA256:
        raise ValueError('Unexpected patched shared word-loader template')
    candidate = target.read_text()
    inverse = candidate
    for name in HELPERS:
        inverse = inverse.replace(definition(candidate, name), definition(original, name))
    if inverse != original:
        raise ValueError('Shared template changed outside the two helpers')
    canonical = work / HEADER_NAME
    generate_canonical(source, target, canonical)
    generated = {HEADER_NAME: canonical}
    for name, shader in DEDICATED_SHADERS.items():
        target_header = headers[name]
        generate_canonical(source, target, target_header, inputs=work / shader.directory)
        generated[name] = target_header
    # This is the same accepted include redirection, now expressed as a patch
    # instead of a Python replacement in the upstream C++ implementation.
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags,
                        str((patch_root / INTEGRATION_PATCH_NAME).resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected shared word-loader final source: ' + name)
    return generated
