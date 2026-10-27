"""Actual prepared indexed selector/cache/factory and complete dispatch recorder."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_iq3s_subgroup as iq3
import prepare_webgpu_tensor_copy as copy
from test_webgpu_q8_short_overlay import function


def prepare_base(upstream: Path, output: Path) -> None:
    independent = os.environ.get('LCB_TEST_IQ3S_BASE')
    if independent:
        base = Path(independent).resolve()
        for name, expected in iq3.INPUTS.items():
            if copy.digest(base / name) != expected:
                raise ValueError('Independent IQ3_S base mismatch: ' + name)
        shutil.copytree(base, output, dirs_exist_ok=True)
    else:
        source.prepare(upstream, output)


def prepare(upstream: Path, output: Path) -> None:
    prepare_base(upstream, output)
    header = output / 'ggml-webgpu-shader-lib.hpp'
    if copy.digest(header) in (iq3.OUTPUTS[header.name], source.SSM_HEADER_OUTPUT_SHA256):
        if copy.digest(output / iq3.HEADER_NAME) != iq3.HEADER_SHA256:
            raise ValueError('Unexpected integrated IQ3_S embedding')
        return
    with tempfile.TemporaryDirectory(prefix='iq3s-stage-') as tmp:
        work = Path(tmp)
        (work / iq3.BASE).mkdir(parents=True)
        for name in iq3.INPUTS:
            shutil.copyfile(output / name, work / iq3.BASE / name)
        extra = iq3.prepare(work, upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
        for name in ('ggml-webgpu.cpp', header.name):
            shutil.copyfile(work / iq3.BASE / name, output / name)
        for name, path in extra.items():
            shutil.copyfile(path, output / name)


def build_host(upstream: Path, prepared: Path, output: Path) -> Path:
    header = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    cpp = (prepared / 'ggml-webgpu.cpp').read_text()
    prefix = (ROOT / 'tests/webgpu-expert-short/host-prefix.cpp').read_text()
    prefix = prefix.replace('    throw std::runtime_error("Unexpected recording type");',
        '    static const auto iq3 = [] { ggml_type_traits t{}; t.type_name="iq3_s"; return t; }();\n'
        '    if(type==GGML_TYPE_IQ3_S)return &iq3;\n    throw std::runtime_error("Unexpected recording type");')
    start = header.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(')
    end = header.index('    webgpu_pipeline get_unary_pipeline(', start)
    original = (upstream / iq3.BASE / 'ggml-webgpu-shader-lib.hpp').read_text()
    reference_start = original.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(')
    reference_end = original.index('    webgpu_pipeline get_unary_pipeline(', reference_start)
    chunks = [prefix, '#include "ggml-webgpu-iq3s-id-subgroup-shaders.hpp"\n#include <limits>',
        function(header, 'template <typename T> inline void ggml_webgpu_hash_combine('),
        function(header, 'struct ggml_webgpu_shader_lib_context {') + ';',
        function(header, 'inline uint32_t ggml_webgpu_expert_short_workgroup('),
        function(header, 'inline uint32_t ggml_webgpu_q4_short_workgroup('),
        function(header, 'inline uint32_t ggml_webgpu_iq3s_id_subgroup_size('),
        function(header, 'struct ggml_webgpu_mul_mat_vec_shader_decisions {') + ';',
        header[header.index('struct ggml_webgpu_mul_mat_id_pipeline_key {'):header.index('/** Cpy **/')],
        '''class RecordingLibrary {
            int device=0;
            pre_wgsl::Preprocessor preprocessor;
            std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key,webgpu_pipeline,
                               ggml_webgpu_mul_mat_id_pipeline_key_hash> mul_mat_id_vec_pipelines;
            public:
        ''', header[start:end], '};',
        '''class ReferenceLibrary {
            int device=0;
            pre_wgsl::Preprocessor preprocessor;
            std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key,webgpu_pipeline,
                               ggml_webgpu_mul_mat_id_pipeline_key_hash> mul_mat_id_vec_pipelines;
            public:
        ''', original[reference_start:reference_end], '};',
        (Path(__file__).with_name('dispatch-prefix.cpp')).read_text(),
        function(cpp, 'static inline void compute_2d_workgroups('),
        function(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec('),
        (Path(__file__).with_name('host-cases.cpp')).read_text()]
    output.mkdir(parents=True, exist_ok=True)
    shaders = upstream / iq3.BASE / 'wgsl-shaders'
    subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True)
    code = output / 'host.cpp'
    code.write_text('\n'.join(chunks))
    executable = output / 'host'
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
        '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / iq3.BASE),
        '-I' + str(prepared), '-I' + str(output), str(code), '-o', str(executable)], check=True)
    return executable
