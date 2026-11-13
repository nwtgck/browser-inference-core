"""Extract actual normal/profile factories; replace pipeline creation only."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(runtime_test_root('llama-cpp')))
from tests.runtimes.llama_cpp.matvec_subgroup_test_support import q4_subgroup_helpers
from tests.runtimes.llama_cpp.webgpu_embedding_test_support import canonical_prefix, legacy_fixed_sources


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
    prefix = canonical_prefix((runtime_test_root('llama-cpp') / 'webgpu-q4-short/host-prefix.cpp').read_text(), after)
    prefix = prefix[:prefix.index('extern "C" const ggml_type_traits * ggml_get_type_traits(')]
    if profile:
        prefix = '#include <array>\n#include "ggml-webgpu-q8-diagnostic-sha256.hpp"\n' + prefix.replace(
            '    std::string wgsl;', '    std::string wgsl;\n    std::array<uint8_t, 32> diagnostic_sha256 = {};')
    prefix += '''
#include <optional>
#include "ggml-webgpu-q8-subgroup-shaders.hpp"
extern "C" const ggml_type_traits * ggml_get_type_traits(ggml_type type) {
    static const auto q6=[] { ggml_type_traits t{};t.type_name="q6_K";return t;}();
    static const auto q4=[] { ggml_type_traits t{};t.type_name="q4_K";return t;}();
    static const auto q8=[] { ggml_type_traits t{};t.type_name="q8_0";return t;}();
    if(type==GGML_TYPE_Q6_K)return &q6;
    if(type==GGML_TYPE_Q4_K)return &q4;
    if(type==GGML_TYPE_Q8_0)return &q8;
    throw std::runtime_error("Unexpected recording type");
}
'''
    chunks = [prefix]
    for namespace, directory in [('baseline', before), ('candidate', after)]:
        text = legacy_fixed_sources(directory)['ggml-webgpu-shader-lib.hpp']
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
    static uint32_t select(const Context & c,bool indexed=false,bool mmvq=false,bool overlap=false) {
        return ggml_webgpu_q4_short_workgroup(c,indexed,mmvq,overlap);
    }
''']
        marker = '    static ggml_webgpu_mul_mat_vec_pipeline_key make_mul_mat_vec_key(' if profile else '    webgpu_pipeline get_mul_mat_vec_pipeline('
        start = text.index(marker)
        chunks += [text[start:text.index('    webgpu_pipeline get_mul_mat_fast_pipeline(', start)], '};', '}']
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
