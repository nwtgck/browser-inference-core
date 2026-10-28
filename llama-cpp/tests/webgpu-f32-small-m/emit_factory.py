"""Extract actual old/new fast factories, keys and production grid calculation.

Only device pipeline creation is recorded. Canonical WGSL is embedded and
specialized by the pinned upstream emitter/preprocessor, without a GPU build.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
from test_webgpu_bind_group_cache import definition as function


def build_host(upstream: Path, baseline: Path, prepared: Path, output: Path) -> Path:
    old = (baseline / 'ggml-webgpu-shader-lib.hpp').read_text()
    new = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    backend = (prepared / 'ggml-webgpu.cpp').read_text()
    prefix = (ROOT / 'tests/webgpu-q8-short/host-prefix.cpp').read_text()
    prefix = prefix[:prefix.index('extern "C" const ggml_type_traits * ggml_get_type_traits(')]
    traits = (upstream / 'ggml/src/ggml.c').read_text()
    table = traits[traits.index('static const struct ggml_type_traits type_traits['):]
    cases = []
    for name, body in re.findall(r'\[(GGML_TYPE_\w+)\] = \{(.*?)\n    \},', table, re.DOTALL):
        label = re.search(r'\.type_name\s*=\s*"([^"]+)"', body)
        quantized = re.search(r'\.is_quantized\s*=\s*(true|false)', body)
        if label and quantized:
            cases.append(f'case {name}: {{ static const auto t=[] {{ ggml_type_traits r{{}}; r.type_name="{label[1]}"; r.is_quantized={quantized[1]}; return r; }}(); return &t; }}')
    if not cases:
        raise ValueError('Pinned public type traits changed')
    trait_functions = ('extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) { switch(type) {'
                       + ''.join(cases) + 'default: throw std::runtime_error("Unknown recording type"); }}\n'
                       'extern "C" bool ggml_is_quantized(ggml_type type) { return ggml_get_type_traits(type)->is_quantized; }')
    constants = '\n'.join(re.findall(r'^#define WEBGPU_MUL_MAT_(?!VEC_).+$', new, re.M))
    chunks = [prefix, '#include <array>\n#include <cassert>\n#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))',
              trait_functions, constants, function(new, 'struct ggml_webgpu_shader_lib_context {')]
    support = function(backend, 'static bool ggml_backend_webgpu_device_supports_op(')
    support = support[support.index('case GGML_OP_MUL_MAT:'):support.index('case GGML_OP_MUL_MAT_ID:')]
    types = list(dict.fromkeys(re.findall(r'case (GGML_TYPE_\w+):', support)))
    if not {'GGML_TYPE_F32', 'GGML_TYPE_F16', 'GGML_TYPE_BF16', 'GGML_TYPE_IQ3_S'}.issubset(types):
        raise ValueError('Actual matrix type admission changed')
    chunks.append('static const std::vector<ggml_type> source_nonf32_types={' +
                  ','.join(name for name in types if name != 'GGML_TYPE_F32') + '};')
    for namespace, header in [('baseline', old), ('candidate', new)]:
        chunks += [f'namespace {namespace} {{',
                   function(header, 'template <typename T> inline void ggml_webgpu_hash_combine('),
                   header[header.index('struct ggml_webgpu_mul_mat_pipeline_key {'):header.index('/** MUL_MAT_ID **/')],
                   '''class Library {
                       int device=0; pre_wgsl::Preprocessor preprocessor;
                       std::unordered_map<ggml_webgpu_mul_mat_pipeline_key, webgpu_pipeline,
                                          ggml_webgpu_mul_mat_pipeline_key_hash> mul_mat_fast_pipelines;
                       public:
                       using Decisions = ggml_webgpu_mul_mat_shader_decisions;
                   ''', function(header, '    webgpu_pipeline get_mul_mat_fast_pipeline('), '};\n}']
    start = backend.index('            uint32_t tile_m_s = decisions->tile_m * decisions->wg_size_m;')
    end = backend.index('\n        }', start)
    chunks += [function(backend, 'static inline void compute_2d_workgroups('),
               '''template<class Library> static std::array<uint32_t,2> production_grid(const webgpu_pipeline & pipeline, const ggml_tensor * dst) {
                   const auto *decisions=static_cast<const typename Library::Decisions *>(pipeline.context.get());
                   uint32_t wg_m=0,wg_n=0,wg_x=0,wg_y=0;
               ''' + backend[start:end] + '''
                   const uint32_t total_wg=wg_m*wg_n*dst->ne[2]*dst->ne[3];
                   compute_2d_workgroups(total_wg,17,wg_x,wg_y); return {wg_x,wg_y};
               }''', (Path(__file__).with_name('host-cases.cpp')).read_text()]
    output.mkdir(parents=True, exist_ok=True)
    shaders = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
    code = output / 'host.cpp'
    code.write_text('\n'.join(chunks))
    executable = output / 'host'
    compiled = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                               '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / 'ggml/src/ggml-webgpu'),
                               '-I' + str(output), str(code), '-o', str(executable)], capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    return executable
