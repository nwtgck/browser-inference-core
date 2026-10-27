"""Skip inactive four-column groups in the existing MoE matrix register tile."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-moe-group-skip.patch'
PATCH_SHA256 = '37a5008bb88f66d84b48fdf65da1bfa2ae6744b5b989fa08ab85078f3e0eadde'
BASE = Path('ggml/src/ggml-webgpu')
SHADER_NAME = 'mul_mat_id_group.wgsl'
HEADER_NAME = 'ggml-webgpu-mul-mat-id-group-shaders.hpp'
INPUTS = {
    'ggml-webgpu.cpp': 'a53ea652fa4bbaec5c8ba503dcf852c672e433086909846fb55f0dc3ede81bfd',
    'ggml-webgpu-shader-lib.hpp': '86567b1e8d20063c7b8a6a12a8b2a10c379ab1c851a0eb18455ed76582fb7ba7',
}
OUTPUTS = {
    'ggml-webgpu.cpp': INPUTS['ggml-webgpu.cpp'],
    'ggml-webgpu-shader-lib.hpp': 'bd8f3e7a39cec314c47fc6753eb044dbc5b4411ad20c195345879e003a6040da',
    'wgsl-shaders/' + SHADER_NAME: 'daa6fd66e380b4c0277f531c07928b6ae5a35ab2c307bcd44f554498e56e962b',
}
HEADER_SHA256 = '2b3287b36d75c14501114d5c76b6aa9c35f8a19046147b2587ba21b35972a8d2'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '995b30ce233bdc23cf918da8916485b568aeac0371bcab458d0df3b7b1d172cf',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'ab446e25ba92b0733bbe957b32adab70be2c3973cd8f412fbbcf2dd86818a289',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl': '667d3a0f8c0e8a8b033d1b3cf861b69d299dee276e76caddaa0ecccbe885a683',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather.wgsl': '5214c78e6edd32841b40e1b5137a13764997db52e965392d3b914357ed7a9116',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl': 'e29282ffcb16bdeb0aa0c6cd7ae15703ec855c4ff010e8398075d0257b6a49d5',
    'ggml/src/ggml-webgpu/wgsl-shaders/quant_inner_loops.tmpl': '7f023ec774a11265985d92fb630fb94cd8e4599499f941983d1b23b37dacf5ad',
}
TEMPLATES = ('common_decls.tmpl', 'mul_mat_decls.tmpl', 'quant_inner_loops.tmpl')


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('MoE group-skip contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('MoE group-skip composition input changed: ' + relative)
    for relative in (HEADER_NAME, 'wgsl-shaders/' + SHADER_NAME):
        if (source / BASE / relative).exists() or (work / BASE / relative).exists():
            raise ValueError('MoE group skip would replace an unreviewed source: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('MoE group-skip patch identity changed')
    shader = work / BASE / 'wgsl-shaders' / SHADER_NAME
    shader.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / BASE / 'wgsl-shaders/mul_mat_id.wgsl', shader)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected MoE group-skip output: ' + relative)
    # Expand only the matrix shader. The canonical header, gather, matvec and
    # all shared load/dequantization templates remain authoritative upstream.
    shaders = work / 'moe-group-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(shader, shaders / SHADER_NAME)
    for name in TEMPLATES:
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected MoE group-skip embedding identity')
    return {HEADER_NAME: header}
