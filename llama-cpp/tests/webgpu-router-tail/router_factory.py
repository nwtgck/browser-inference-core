"""Extract real baseline and fused factory methods from the prepared source."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def function(text, marker):
    start = text.index(marker)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += int(text[end] == '{') - int(text[end] == '}')
        end += 1
    return text[start:end]


def build(upstream: Path, prepared: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    fragments = [(HERE / 'factory-prefix.cpp').read_text()]
    for marker in ('template <typename T> inline void ggml_webgpu_hash_combine(',
                   'inline uintptr_t ggml_webgpu_tensor_addr(', 'inline bool ggml_webgpu_tensor_equal('):
        fragments.append(function(text, marker))
    for name in ('ggml_webgpu_shader_lib_context', 'ggml_webgpu_generic_shader_decisions',
                 'ggml_webgpu_get_rows_shader_decisions', 'ggml_webgpu_binary_shader_decisions',
                 'ggml_webgpu_get_rows_pipeline_key', 'ggml_webgpu_get_rows_pipeline_key_hash',
                 'ggml_webgpu_unary_pipeline_key', 'ggml_webgpu_unary_pipeline_key_hash',
                 'ggml_webgpu_binary_pipeline_key', 'ggml_webgpu_binary_pipeline_key_hash'):
        fragments.append(function(text, 'struct ' + name + ' {') + ';')
    fragments.append('''class Library {
    int device = 0;
    pre_wgsl::Preprocessor preprocessor;
    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;
    std::unordered_map<ggml_webgpu_get_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_key_hash> get_rows_pipelines;
    std::unordered_map<ggml_webgpu_unary_pipeline_key, webgpu_pipeline, ggml_webgpu_unary_pipeline_key_hash> unary_pipelines;
    std::unordered_map<ggml_webgpu_binary_pipeline_key, webgpu_pipeline, ggml_webgpu_binary_pipeline_key_hash> binary_pipelines;
    std::unordered_map<uint64_t, webgpu_pipeline> router_tail_pipelines;
public:
''')
    for name in ('get_get_rows_pipeline', 'get_sum_rows_pipeline', 'get_unary_pipeline',
                 'get_binary_pipeline', 'get_router_tail_pipeline'):
        fragments.append(function(text, '    webgpu_pipeline ' + name + '('))
    fragments.append((HERE / 'factory-cases.cpp').read_text())
    cpp = directory / 'factory.cpp'
    cpp.write_text('\n'.join(fragments))
    shaders = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                    '--output_file', str(directory / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
    executable = directory / 'factory'
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-I' + str(prepared),
                    '-I' + str(directory), '-isystem', str(upstream / 'ggml/include'),
                    '-isystem', str(upstream / 'ggml/src/ggml-webgpu'), str(cpp), '-o', str(executable)], check=True)
    return executable
