"""Compile actual capacity/allocator/caller functions with recorded GPU resources."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
from test_webgpu_bind_group_cache import definition


def build(upstream: Path, baseline: Path, candidate: Path, output: Path, library: Path, *, profile: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    header = (candidate / 'ggml-webgpu-shader-lib.hpp').read_text()
    chunks = [(HERE / 'prefix.cpp').read_text(), '\n'.join(re.findall(r'^#define WEBGPU_MUL_MAT_.*$', header, re.M))]
    # This source constant is used only by compiled-but-unexecuted flash branches.
    chunks.append(re.search(r'^inline constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = .*$', header, re.M)[0])
    common = ['static size_t ggml_webgpu_tensor_offset(',
              'static wgpu::Buffer ggml_webgpu_tensor_buf(',
              'static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor *',
              'static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor *',
              'static size_t ggml_webgpu_tensor_misalignment(webgpu_context &',
              'static size_t ggml_webgpu_tensor_align_offset(webgpu_context &',
              'static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor *',
              'static size_t ggml_webgpu_tensor_binding_size(webgpu_context &',
              'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(',
              'static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(',
              'static bool ggml_webgpu_mul_mat_vec4_aligned(',
              'static inline void compute_2d_workgroups(']
    cpp = (candidate / 'ggml-webgpu.cpp').read_text()
    chunks += [definition(cpp, marker) for marker in common]
    for name, prepared in [('baseline', baseline), ('candidate', candidate)]:
        text = (prepared / 'ggml-webgpu.cpp').read_text()
        chunks.append('namespace ' + name + ' {')
        chunks += [definition(text, marker) for marker in
                   ['static uint32_t ggml_webgpu_mul_mat_id_max_tiles(',
                    'static size_t ggml_webgpu_mul_mat_id_tiles_nbytes(',
                    'static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(',
                    'static webgpu_encoded_op ggml_webgpu_mul_mat_id(']]
        chunks.append('}')
    chunks.append((HERE / 'cases.cpp').read_text())
    path = output / 'host.cpp'
    path.write_text('\n'.join(chunks))
    flags = ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE'] if profile else []
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                    '-isystem', str(upstream / 'ggml/include'), '-isystem', str(upstream / 'ggml/src'),
                    *flags, str(path), str(library), '-pthread', '-ldl', '-lm', '-o', str(output / 'host')], check=True)
    return output / 'host'
