"""Record actual normal/profile Q4 selection, cache, source and host grid."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests'))
from matvec_subgroup_test_support import q4_subgroup_helpers


def definition(text, marker):
    start = text.index(marker)
    brace = text.index('{', start)
    depth = 0
    for end in range(brace, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[start:end + 1]
    raise ValueError(marker)


def build(upstream, before, after, output, *, profile):
    output.mkdir(parents=True)
    prefix = (ROOT / 'tests/webgpu-q4-short/host-prefix.cpp').read_text()
    prefix = prefix[:prefix.index('extern "C" const ggml_type_traits * ggml_get_type_traits(')]
    if profile:
        prefix = '#include <array>\n#include "ggml-webgpu-q8-diagnostic-sha256.hpp"\n' + prefix.replace(
            '    std::string wgsl;', '    std::string wgsl;\n    std::array<uint8_t, 32> diagnostic_sha256 = {};')
    prefix += '''
#include <array>
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#include <optional>
#include "ggml-webgpu-q8-subgroup-shaders.hpp"
extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const auto q40=[] { ggml_type_traits t{};t.type_name="q4_0";return t;}();
    static const auto q6=[] { ggml_type_traits t{};t.type_name="q6_K";return t;}();
    static const auto q4=[] { ggml_type_traits t{};t.type_name="q4_K";return t;}();
    static const auto q8=[] { ggml_type_traits t{};t.type_name="q8_0";return t;}();
    if(type==GGML_TYPE_Q4_0)return &q40;
    if(type==GGML_TYPE_Q6_K)return &q6;
    if(type==GGML_TYPE_Q4_K)return &q4;
    if(type==GGML_TYPE_Q8_0)return &q8;
    throw std::runtime_error("Unexpected recording type");
}
'''
    chunks = [prefix]
    for namespace, directory in [('baseline', before), ('candidate', after)]:
        text = (directory / 'ggml-webgpu-shader-lib.hpp').read_text()
        chunks += ['namespace ' + namespace + ' {',
                   definition(text, 'template <typename T> inline void ggml_webgpu_hash_combine('),
                   definition(text, 'struct ggml_webgpu_shader_lib_context {') + ';']
        for name in ('q4', 'q8'):
            chunks.append(definition(text, 'inline uint32_t ggml_webgpu_' + name + '_short_workgroup('))
        chunks += [definition(text, 'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size('),
                   q4_subgroup_helpers(text),
                   text[text.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {'):text.index('struct ggml_webgpu_quantize_q8_pipeline_key {')],
                   definition(text, 'inline bool ggml_webgpu_can_use_mmvq('), '''
class Library {
    int device=0;pre_wgsl::Preprocessor preprocessor;
    std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;
public:
    using Context=ggml_webgpu_shader_lib_context;
    using Decisions=ggml_webgpu_mul_mat_vec_shader_decisions;
''']
        marker = '    static ggml_webgpu_mul_mat_vec_pipeline_key make_mul_mat_vec_key(' if profile else '    webgpu_pipeline get_mul_mat_vec_pipeline('
        start = text.index(marker)
        chunks += [text[start:text.index('    webgpu_pipeline get_mul_mat_fast_pipeline(', start)], '};', '}']
    backend = (after / 'ggml-webgpu.cpp').read_text()
    caller = ('static webgpu_encoded_op ggml_webgpu_mul_mat_impl(' if profile else
              'static webgpu_encoded_op ggml_webgpu_mul_mat(')
    start = backend.index('    if (use_mat_vec) {', backend.index('    // Calculate workgroup dimensions', backend.index(caller)))
    end = backend.index('    } else {', start)
    dispatch = backend[start + len('    if (use_mat_vec) {\n'):end]
    chunks += ['using ggml_webgpu_mul_mat_vec_shader_decisions = candidate::ggml_webgpu_mul_mat_vec_shader_decisions;',
               definition(backend, 'static inline void compute_2d_workgroups('),
               'static std::array<uint32_t,2> production_dispatch(webgpu_pipeline pipeline,ggml_tensor *dst,uint32_t max_wg_per_dim) {\nuint32_t wg_x=1,wg_y=1;\n' + dispatch + '\nreturn {wg_x,wg_y};\n}']
    chunks.append((HERE / 'cases.cpp').read_text())
    (output / 'factory.cpp').write_text('\n'.join(chunks))
    shaders = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
    executable = output / 'factory'
    result = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / 'ggml/src/ggml-webgpu'),
                    '-I' + str(after), '-I' + str(output), str(output / 'factory.cpp'), '-o', str(executable)],
                   capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return executable
