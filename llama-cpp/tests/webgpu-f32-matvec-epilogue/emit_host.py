"""Extract actual graph admission, physical guard, MM caller and matvec factory."""
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
from test_webgpu_bind_group_cache import definition


def build(upstream: Path, prepared: Path, output: Path, library: Path, *, profile: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    header = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    cpp = (prepared / 'ggml-webgpu.cpp').read_text()
    chunks = [(HERE / 'prefix.cpp').read_text(), '\n'.join(re.findall(r'^#define WEBGPU_MUL_MAT_.*$', header, re.M))]
    for marker in ('template <typename T> inline void ggml_webgpu_hash_combine(',
                   'struct ggml_webgpu_shader_lib_context {', 'struct ggml_webgpu_generic_shader_decisions {',
                   'inline uint32_t ggml_webgpu_q8_short_workgroup(', 'inline uint32_t ggml_webgpu_q4_short_workgroup(',
                   'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size(', 'inline bool ggml_webgpu_can_use_mmvq('):
        chunks.append(definition(header, marker) + (';' if marker.startswith('struct ') else ''))
    chunks += [header[header.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {'):header.index('struct ggml_webgpu_quantize_q8_pipeline_key {')],
               definition(header, 'struct ggml_webgpu_mul_mat_shader_decisions {') + ';', '''class RecordingLibrary {
int device=0; pre_wgsl::Preprocessor preprocessor;
std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;
public:
''']
    start = ('    static ggml_webgpu_mul_mat_vec_pipeline_key make_mul_mat_vec_key(' if profile else
             '    webgpu_pipeline get_mul_mat_vec_pipeline(')
    chunks += [header[header.index(start):header.index('    webgpu_pipeline get_mul_mat_fast_pipeline(', header.index(start))],
               'webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context&,bool,bool) {throw std::runtime_error("Post must remain matvec");}',
               '};', (HERE / 'context.cpp').read_text()]
    # Only GPU object creation and final encode are recorded. These helpers are
    # the complete prepared source, including source rebasing and actual params.
    markers = [
        'static size_t ggml_webgpu_tensor_offset(', 'static wgpu::Buffer ggml_webgpu_tensor_buf(',
        'static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor *',
        'static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor *',
        'static size_t ggml_webgpu_tensor_misalignment(webgpu_context &',
        'static size_t ggml_webgpu_tensor_align_offset(webgpu_context &',
        'static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor *',
        'static size_t ggml_webgpu_tensor_binding_size(webgpu_context &',
        'static bool ggml_webgpu_tensor_binding_overlap(const webgpu_global_context &',
        'struct ggml_webgpu_merged_binding_range {',
        'static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(',
        'static uint32_t ggml_webgpu_tensor_merged_element_offset(',
        'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(',
        'static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(',
        'static bool ggml_webgpu_mul_mat_vec4_aligned(', 'static inline void compute_2d_workgroups(',
        'static inline uint32_t ggml_webgpu_u32_from_f32(']
    for marker in markers:
        chunks.append(definition(cpp, marker) + (';' if marker.startswith('struct ') else ''))
    if profile:
        chunks += [definition(cpp, 'static ggml_webgpu_shader_lib_context ggml_webgpu_mat_context('),
                   'template<bool Diagnostic>\n' + definition(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat_impl(')]
    chunks += [definition(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat(')]
    start = cpp.rindex('static void * ggml_backend_webgpu_buffer_get_base(')
    chunks.append(definition(cpp[start:], 'static void * ggml_backend_webgpu_buffer_get_base('))
    for marker in ('struct webgpu_f32_matvec_buffer_facts {', 'struct webgpu_f32_matvec_binding {',
                   'static bool ggml_webgpu_f32_matvec_binding(',
                   'static std::optional<webgpu_encoded_op> ggml_webgpu_f32_matvec_epilogue('):
        chunks.append(definition(cpp, marker) + (';' if marker.startswith('struct ') else ''))
    chunks.append((HERE / 'cases.cpp').read_text())
    path = output / 'host.cpp'; path.write_text('\n'.join(chunks))
    shader_dir = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
    command = ['c++', '-std=c++17', '-O1', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
               '-I' + str(prepared), '-I' + str(output), '-isystem', str(upstream / 'ggml/include'),
               '-isystem', str(upstream / 'ggml/src'), '-isystem', str(shader_dir.parent)]
    if profile:
        command += ['-DTEST_PROFILE']
    subprocess.run(command + [str(path), str(library), '-pthread', '-ldl', '-lm', '-o', str(output / 'host')], check=True)
    return output / 'host'
