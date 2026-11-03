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
    'ggml-webgpu.cpp': '972928fc0755daad4e24b2da548d2e62329d5ef474ab5ab7e5214365e88d48cf',
    'ggml-webgpu-shader-lib.hpp': 'a242d104ea3841d3041641fdd33561fbcb9157784a6198d6696a0bb6cee93700',
}
OUTPUTS = {
    'ggml-webgpu.cpp': INPUTS['ggml-webgpu.cpp'],
    'ggml-webgpu-shader-lib.hpp': '7c5b913454691966279c65cab8953ecb19348cfd4653df94202ca60349d3e48d',
    'wgsl-shaders/' + SHADER_NAME: 'daa6fd66e380b4c0277f531c07928b6ae5a35ab2c307bcd44f554498e56e962b',
}
HEADER_SHA256 = '8b3dabdd9356b5a5e020d84c354faa68a70b8634c81793060d53fb58bd55bc10'
CONTRACT_INPUTS = {
    'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
    'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
    'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
    'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl': '667d3a0f8c0e8a8b033d1b3cf861b69d299dee276e76caddaa0ecccbe885a683',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather.wgsl': '433c7b72fb9382e6e452004bdb602d7be8f2654543de680a28d31d8a263298e7',
    'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl': 'c461aa2108b752504f47338e41b14ee6570abdd831cd83652e052147c89d69ef',
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
