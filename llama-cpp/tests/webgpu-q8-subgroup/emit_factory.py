"""Extract the prepared selector/cache/factory and production dispatch calculation.

Only device pipeline creation is recorded; shader specialization is production
code and the real pinned emitter/preprocessor. No native runtime build is needed.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_q8_subgroup as q8
import prepare_webgpu_tensor_copy as copy
from test_webgpu_q8_short_overlay import function

from matmul_vec4_test_support import explicit_layout_argument


def prepare(upstream: Path, output: Path) -> None:
    source.prepare(upstream, output)
    header = output / 'ggml-webgpu-shader-lib.hpp'
    if copy.digest(header) != source.FINAL_OUTPUTS[False]['ggml-webgpu-shader-lib.hpp'] or copy.digest(output / q8.HEADER_NAME) != q8.HEADER_SHA256:
        raise ValueError('Unexpected integrated Q8 subgroup output')


def build_host(upstream: Path, prepared: Path, output: Path) -> Path:
    text = explicit_layout_argument((prepared / 'ggml-webgpu-shader-lib.hpp').read_text())
    backend = (prepared / 'ggml-webgpu.cpp').read_text()
    dispatch_start = backend.index('    if (use_mat_vec) {', backend.index('    // Calculate workgroup dimensions', backend.index('static webgpu_encoded_op ggml_webgpu_mul_mat(')))
    dispatch_end = backend.index('    } else {', dispatch_start)
    dispatch = backend[dispatch_start + len('    if (use_mat_vec) {\n'):dispatch_end]
    chunks = [
        (ROOT / 'tests/webgpu-q8-short/host-prefix.cpp').read_text(),
        '#include "ggml-webgpu-q8-subgroup-shaders.hpp"\n#include <array>\n#include <limits>\n#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))',
        function(text, 'template <typename T> inline void ggml_webgpu_hash_combine('),
        function(text, 'struct ggml_webgpu_shader_lib_context {') + ';',
        function(text, 'inline uint32_t ggml_webgpu_q8_short_workgroup('),
        function(text, 'inline uint32_t ggml_webgpu_q4_short_workgroup('),
        function(text, 'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size('),
        text[text.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {'):
             text.index('struct ggml_webgpu_quantize_q8_pipeline_key {')],
        function(text, 'inline bool ggml_webgpu_can_use_mmvq('),
        '''class RecordingLibrary {
            int device = 0;
            pre_wgsl::Preprocessor preprocessor;
            std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,
                               ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;
            public:
        ''',
        text[text.index('    webgpu_pipeline get_mul_mat_vec_pipeline('):
             text.index('    webgpu_pipeline get_mul_mat_fast_pipeline(')],
        '};',
        function(backend, 'static inline void compute_2d_workgroups('),
        'static std::array<uint32_t,2> production_dispatch(webgpu_pipeline pipeline, ggml_tensor * dst, uint32_t max_wg_per_dim) {\nuint32_t wg_x=1,wg_y=1;\n'
        + dispatch + '\nreturn {wg_x,wg_y};\n}',
        (Path(__file__).with_name('host-cases.cpp')).read_text(),
    ]
    output.mkdir(parents=True, exist_ok=True)
    shader_dir = upstream / q8.BASE / 'wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True)
    code = output / 'host.cpp'
    code.write_text('\n'.join(chunks))
    exe = output / 'host'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / q8.BASE),
                    '-I' + str(prepared), '-I' + str(output), str(code), '-o', str(exe)], check=True)
    return exe
