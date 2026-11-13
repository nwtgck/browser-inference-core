"""Historical IQ3 prefix recorder; never a current-compatibility check."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
from tests.support import frozen_webgpu_source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import frozen_webgpu_source as reference_source
from tests.support.llama_cpp.frozen_webgpu_stages import iq3s_subgroup as iq3
from tests.support.llama_cpp.frozen_webgpu_stages import iq3s_field_loads as fields
from tests.support.llama_cpp.frozen_webgpu_stages import iq3s_row_split as rows
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy as copy
from tests.runtimes.llama_cpp.test_webgpu_q8_short_overlay import function

from tests.runtimes.llama_cpp.matmul_vec4_test_support import explicit_layout_argument
from tests.support.llama_cpp.frozen_embedding import canonical_prefix, embedding_sha256


def prepare_base(upstream: Path, output: Path) -> None:
    independent = os.environ.get('LCB_TEST_IQ3S_BASE')
    if independent:
        base = Path(independent).resolve()
        for name, expected in iq3.INPUTS.items():
            if copy.digest(base / name) != expected:
                raise ValueError('Independent IQ3_S base mismatch: ' + name)
        shutil.copytree(base, output, dirs_exist_ok=True)
    else:
        source.prepare(upstream, output, patch_root=upstream / 'patches')


def prepare(upstream: Path, output: Path) -> None:
    if os.environ.get('LCB_TEST_IQ3S_BASE'):
        raise ValueError('Current IQ3_S numerical checks require the final embedding; historical prefixes are native-test-only')
    prepare_base(upstream, output)
    header = output / 'ggml-webgpu-shader-lib.hpp'
    if copy.digest(header) != reference_source.FINAL_OUTPUTS[False][header.name] or copy.digest(output / iq3.HEADER_NAME) != embedding_sha256(output, iq3.HEADER_NAME, rows.HEADER_SHA256):
        raise ValueError('Unexpected final IQ3_S embedding')


def prepare_historical_prefix(upstream: Path, output: Path) -> None:
    """Native prefix fixtures only; never the current numerical-checker path."""
    if not os.environ.get('LCB_TEST_IQ3S_BASE'):
        raise ValueError('Historical IQ3_S preparation requires an explicit prefix')
    prepare_base(upstream, output)
    header = output / 'ggml-webgpu-shader-lib.hpp'
    with tempfile.TemporaryDirectory(prefix='iq3s-stage-') as tmp:
        work = Path(tmp)
        (work / iq3.BASE).mkdir(parents=True)
        for name in iq3.INPUTS:
            shutil.copyfile(output / name, work / iq3.BASE / name)
        extra = iq3.prepare(work, upstream, upstream / 'patches', reference_source.patch_environment(work))
        extra[iq3.HEADER_NAME] = fields.prepare(
            work, upstream, upstream / 'patches', reference_source.patch_environment(work),
            previous_header=extra[iq3.HEADER_NAME])
        for name in ('ggml-webgpu.cpp', header.name):
            shutil.copyfile(work / iq3.BASE / name, output / name)
        for name, path in extra.items():
            shutil.copyfile(path, output / name)


def build_host(upstream: Path, prepared: Path, output: Path) -> Path:
    header = explicit_layout_argument((prepared / 'ggml-webgpu-shader-lib.hpp').read_text())
    cpp = (prepared / 'ggml-webgpu.cpp').read_text()
    prefix = canonical_prefix((runtime_test_root('llama-cpp') / 'webgpu-expert-short/host-prefix.cpp').read_text(), prepared)
    prefix = prefix.replace('    throw std::runtime_error("Unexpected recording type");',
        '    static const auto iq3 = [] { ggml_type_traits t{}; t.type_name="iq3_s"; return t; }();\n'
        '    if(type==GGML_TYPE_IQ3_S)return &iq3;\n    throw std::runtime_error("Unexpected recording type");')
    start = header.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(')
    end = header.index('    webgpu_pipeline get_unary_pipeline(', start)
    original = explicit_layout_argument((upstream / iq3.BASE / 'ggml-webgpu-shader-lib.hpp').read_text())
    reference_start = original.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(')
    reference_end = original.index('    webgpu_pipeline get_unary_pipeline(', reference_start)
    chunks = [prefix,
        'static unsigned pipeline_id(const webgpu_pipeline & p) { return p.id; }\n'
        'static const std::string & pipeline_code(const webgpu_pipeline & p) { return p.wgsl; }', '#include "ggml-webgpu-iq3s-id-subgroup-shaders.hpp"\n#include <limits>',
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
        (runtime_test_root('llama-cpp') / 'webgpu-iq3s-subgroup/dispatch-prefix.cpp').read_text(),
        function(cpp, 'static inline void compute_2d_workgroups('),
               (function(cpp, 'static bool ggml_webgpu_mul_mat_vec4_aligned(') if 'static bool ggml_webgpu_mul_mat_vec4_aligned(' in cpp else ''),
        function(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec('),
        (runtime_test_root('llama-cpp') / 'webgpu-iq3s-subgroup/host-cases.cpp').read_text()]
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
