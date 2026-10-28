"""Record the complete actual matmul callers and all four native factories."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
from test_webgpu_bind_group_cache import definition


def build(upstream: Path, baseline: Path, candidate: Path, output: Path, *, profile: bool) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    header = (candidate / 'ggml-webgpu-shader-lib.hpp').read_text()
    chunks = [(HERE / 'prefix.cpp').read_text()]
    chunks += ['\n'.join(re.findall(r'^#define WEBGPU_MUL_MAT_.*$', header, re.M))]
    for marker in ['template <typename T> inline void ggml_webgpu_hash_combine(',
                   'struct ggml_webgpu_shader_lib_context {',
                   'struct ggml_webgpu_generic_shader_decisions {',
                   'inline uint32_t ggml_webgpu_q8_short_workgroup(',
                   'inline uint32_t ggml_webgpu_expert_short_workgroup(',
                   'inline uint32_t ggml_webgpu_q4_short_workgroup(',
                   'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size(',
                   'inline uint32_t ggml_webgpu_iq3s_id_subgroup_size(',
                   'inline bool ggml_webgpu_can_use_mmvq(']:
        chunks.append(definition(header, marker) + (';' if marker.startswith('struct ') else ''))
    chunks.append(header[header.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {'):header.index('/** Cpy **/')])
    for name, prepared in [('baseline', baseline), ('candidate', candidate)]:
        hpp = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
        cpp = (prepared / 'ggml-webgpu.cpp').read_text()
        chunks += ['namespace ' + name + ' {', '''class RecordingLibrary {
int device=0; pre_wgsl::Preprocessor preprocessor;
std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;
std::unordered_map<ggml_webgpu_mul_mat_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_pipeline_key_hash> mul_mat_fast_pipelines;
std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_id_pipeline_key_hash> mul_mat_id_pipelines,mul_mat_id_vec_pipelines;
std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_id_gather_pipeline_key_hash> mul_mat_id_gather_pipelines;
public:''']
        first = ('    static ggml_webgpu_mul_mat_vec_pipeline_key make_mul_mat_vec_key('
                 if profile else '    webgpu_pipeline get_mul_mat_vec_pipeline(')
        chunks += [hpp[hpp.index(first):hpp.index('    webgpu_pipeline get_unary_pipeline(', hpp.index(first))],
                   '};', (HERE / 'dispatch-prefix.cpp').read_text()]
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
            'static inline void compute_2d_workgroups(',
            'static uint32_t ggml_webgpu_mul_mat_id_max_tiles(',
            'static size_t ggml_webgpu_mul_mat_id_tiles_nbytes(',
        ]
        for marker in markers:
            chunks.append(definition(cpp, marker) + (';' if marker.startswith('struct ') else ''))
        if name == 'candidate':
            chunks.append(definition(cpp, 'static bool ggml_webgpu_mul_mat_vec4_aligned('))
        if profile:
            chunks += [definition(cpp, 'static ggml_webgpu_shader_lib_context ggml_webgpu_mat_context('),
                       'template<bool Diagnostic>\n' + definition(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat_impl(')]
        for marker in ['static webgpu_encoded_op ggml_webgpu_mul_mat(',
                       'static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(',
                       'static webgpu_encoded_op ggml_webgpu_mul_mat_id(']:
            chunks.append(definition(cpp, marker))
        chunks += ['static webgpu_encoded_op run(webgpu_context & ctx,ggml_tensor * a,ggml_tensor * x,ggml_tensor * ids,ggml_tensor * dst,bool indexed) {return indexed?ggml_webgpu_mul_mat_id(ctx,a,x,ids,dst):ggml_webgpu_mul_mat(ctx,a,x,dst);}', '}']
    chunks.append((HERE / 'cases.cpp').read_text())
    (output / 'host.cpp').write_text('\n'.join(chunks))
    shader_dir = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
    command = ['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
               '-I' + str(candidate), '-I' + str(output), '-isystem', str(upstream / 'ggml/include'),
               '-isystem', str(shader_dir.parent)]
    if profile:
        command += ['-DTEST_PROFILE']
    result = subprocess.run(command + [str(output / 'host.cpp'), '-o', str(output / 'host')], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return output / 'host'
