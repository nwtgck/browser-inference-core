"""Narrow field loads in the existing dedicated IQ3_S indexed embedding."""
from pathlib import Path
import shutil
import subprocess
import sys

import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp'
SHADER_NAME = 'iq3s_id_subgroup.wgsl'
PATCH_NAME = 'ggml-webgpu-iq3s-field-loads.patch'
PATCH_SHA256 = '341bc30043e42ae49a9c791d825cdacb7cd0d17c7c2b71f6bc51825e1a5a815e'
PRIOR_HEADER_SHA256 = '668f7421877db1ca9675b7e3f251f2ba62964c346ffa47f4c970965a6c36e34d'
HEADER_SHA256 = '24d365982e2126df01e03be9ff46b8f48a409005bca7b7b53c2c95c5d2944154'
SHADER_SHA256 = 'a24f289e8a528e806f7d5a4fca0ef45f0588683a100da314d4a01f5da2489547'
ACCUMULATOR_PATH = str(BASE / 'wgsl-shaders/mul_mat_vec_acc.tmpl')
ACCUMULATOR_OUTPUT_SHA256 = '23ebdf701a4ed2aff8b101acee31496b3bbad75c1ff0f32924cf52468411b952'
CONTRACT_INPUTS = {
    str(BASE / 'wgsl-shaders/common_decls.tmpl'): '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
    ACCUMULATOR_PATH: '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
    str(BASE / 'wgsl-shaders/embed_wgsl.py'): '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, previous_header: Path) -> Path:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('IQ3_S field-load contract changed: ' + relative)
    if previous_header.name != HEADER_NAME or not previous_header.is_file() or tensor_copy.digest(previous_header) != PRIOR_HEADER_SHA256:
        raise ValueError('IQ3_S field-load prior embedding changed')
    main = work / BASE / 'wgsl-shaders' / SHADER_NAME
    if not main.is_file() or tensor_copy.digest(main) != SHADER_SHA256:
        raise ValueError('IQ3_S field-load dedicated main changed')
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('IQ3_S field-load patch identity changed')

    # This root contains only private copies. Never patch the canonical shader
    # directory: ordinary/cooperative matvecs must retain their original input.
    stage = work / 'iq3s-field-loads-source'
    shaders = stage / BASE / 'wgsl-shaders'
    shaders.mkdir(parents=True)
    for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    shutil.copyfile(main, shaders / SHADER_NAME)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=stage, env=env, check=True, capture_output=True)
    if tensor_copy.digest(shaders / 'mul_mat_vec_acc.tmpl') != ACCUMULATOR_OUTPUT_SHA256:
        raise ValueError('Unexpected IQ3_S field-load accumulator output')
    generated = work / 'iq3s-field-loads-generated'
    generated.mkdir()
    header = generated / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected IQ3_S field-load embedding output')
    return header
