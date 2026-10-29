"""Execute actual canonical/prepared matrix factories with recorded pipelines."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_moe_group_skip as group
import prepare_webgpu_moe_gather_logical32 as gather

from matmul_vec4_test_support import explicit_layout_argument


def function(text: str, marker: str) -> str:
    start = text.index(marker)
    opening = text.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def build(upstream: Path, prepared: Path, output: Path, *, profile: bool = False) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    original = explicit_layout_argument((upstream / group.BASE / 'ggml-webgpu-shader-lib.hpp').read_text())
    final = explicit_layout_argument((prepared / 'ggml-webgpu-shader-lib.hpp').read_text())
    backend = (prepared / 'ggml-webgpu.cpp').read_text()
    constants = '\n'.join(line for line in original.splitlines()
                          if line.startswith(('#define WEBGPU_MUL_MAT_TILE_', '#define WEBGPU_MUL_MAT_WG_SIZE_M',
                                              '#define WEBGPU_MUL_MAT_WG_SIZE_N', '#define WEBGPU_MUL_MAT_REG_TILE_K_')))
    keys = final[final.index('struct ggml_webgpu_mul_mat_id_gather_pipeline_key {'):final.index('/** Cpy **/')]
    fields = [function(final, marker) + ';' for marker in
              ('struct ggml_webgpu_shader_lib_context {', 'struct ggml_webgpu_generic_shader_decisions {',
               'struct ggml_webgpu_mul_mat_shader_decisions {')]
    prefix = (HERE / 'factory-prefix.cpp').read_text()
    if (prepared / gather.HEADER_NAME).is_file():
        prefix += '#include "' + gather.HEADER_NAME + '"\n#define TEST_LOGICAL32_GATHER\n'
    chunks = [prefix, constants,
              function(final, 'template <typename T> inline void ggml_webgpu_hash_combine('), *fields, keys]
    layout_marker = 'static bool ggml_webgpu_mul_mat_vec4_aligned('
    checked_layout = layout_marker in backend
    if checked_layout:
        chunks.append(function(backend, layout_marker))
    for name, text in (('Canonical', original), ('Candidate', final)):
        main_factory = function(text, '    webgpu_pipeline get_mul_mat_id_pipeline(')
        if checked_layout and name == 'Canonical':
            # Keep the actual current layout/key selection for both shaders.
            # Only restore the upstream main shader for the group-skip oracle;
            # the old unchecked selector is covered by the layout regression.
            main_factory = function(final, '    webgpu_pipeline get_mul_mat_id_pipeline(').replace(
                'preprocessor.preprocess(wgsl_mul_mat_id_group, defines)',
                'preprocessor.preprocess(wgsl_mul_mat_id, defines)')
        chunks += [f'''class {name} {{ int device=0; pre_wgsl::Preprocessor preprocessor;
std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_id_pipeline_key_hash> mul_mat_id_pipelines;
std::unordered_map<ggml_webgpu_mul_mat_id_gather_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_id_gather_pipeline_key_hash> mul_mat_id_gather_pipelines;
public:''', function(text, '    webgpu_pipeline get_mul_mat_id_gather_pipeline('),
                   main_factory, '};']
    chunks += [function(backend, 'static inline void compute_2d_workgroups('),
               function(backend, 'static uint32_t ggml_webgpu_mul_mat_id_max_tiles(')]
    begin = backend.index('static webgpu_encoded_op ggml_webgpu_mul_mat_id(')
    def vector(marker):
        start = backend.index(marker, begin)
        return backend[start:backend.index('\n    };', start) + len('\n    };')]
    parameters = vector('    std::vector<uint32_t> gather_params = {') + '\n' + vector('    std::vector<uint32_t> main_params = {')
    if checked_layout:
        start = backend.index('    const uint32_t offset_src0 =', begin)
        layout = backend[start:backend.index('    gather_pipeline =', start)]
    else:
        layout = 'const bool vec4_aligned=true; // Historical prefix without the layout stage.'
    start = backend.index('    const uint32_t tile_m_s', begin)
    grid = backend[start:backend.index('\n\n    dispatches.push_back', start)]
    # The test deliberately supplies a smaller legal grid-planning limit, and
    # preserves the actual helper arithmetic for flattening/padded workgroups.
    grid = grid.replace('ctx->global_ctx->', 'context_ptr->global_ctx->')
    main = (HERE / 'factory-main.cpp').read_text().replace('// @ACTUAL_PARAMETER_VECTORS@', parameters)
    main = main.replace('// @ACTUAL_LAYOUT@', layout)
    chunks.append(main.replace('// @ACTUAL_GRID@', grid))
    shader_dir = upstream / group.BASE / 'wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True)
    cpp = output / 'factory.cpp'
    cpp.write_text('\n'.join(chunks))
    executable = output / 'factory'
    flags = ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE'] if profile else []
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-Wno-misleading-indentation',
                    '-I' + str(prepared), '-I' + str(output), '-isystem', str(upstream / group.BASE),
                    '-isystem', str(upstream / 'ggml/include'), *flags, str(cpp), '-o', str(executable)], check=True)
    return executable
