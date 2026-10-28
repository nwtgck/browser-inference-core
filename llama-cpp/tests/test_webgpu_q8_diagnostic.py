"""Profile-only Q8 diagnostic: actual native factories, ABI and submission source."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/webgpu-q8-diagnostic'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_q8_diagnostic as diagnostic
import generate_webgpu_q8_diagnostic as extension

from matmul_vec4_test_support import explicit_layout_argument


def normalized_base_generation(outputs, upstream):
    schema_text = outputs['schema.json']
    schema = json.loads(schema_text)
    fingerprint = hashlib.sha256(schema_text.encode()).hexdigest()
    module = 'export default ' + json.dumps({**schema, 'schemaSha256': fingerprint}, separators=(',', ':')) + ';\n'
    if outputs['schema.mjs'] != module:
        raise AssertionError('Base module does not match the raw schema and fingerprint')
    marker = 'uint64_t lcb_schema_hash(void) { return (uint64_t)(uintptr_t)"' + fingerprint + '"; }'
    if outputs['bindings.cpp'].count(marker) != 1:
        raise AssertionError('Base bindings do not embed the raw schema fingerprint exactly once')
    for record in schema['records']:
        if record['name'] != 'ggml_opt_optimizer_params':
            continue
        for field in record['fields']:
            if field['name'] not in ('adamw', 'sgd'):
                continue
            location = re.fullmatch(r'struct \(unnamed(?: struct)? at ' + re.escape(str(upstream)) +
                                    r'/(ggml/include/ggml-opt\.h:[0-9]+:[0-9]+)\)', field['cType'])
            if location:
                field['cType'] = 'struct (unnamed at <upstream>/' + location[1] + ')'
    canonical = json.dumps(schema, indent=2) + '\n'
    canonical_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    result = dict(outputs)
    result['schema.json'] = canonical
    result['schema.mjs'] = 'export default ' + json.dumps({**schema, 'schemaSha256': canonical_fingerprint}, separators=(',', ':')) + ';\n'
    result['bindings.cpp'] = outputs['bindings.cpp'].replace(marker, marker.replace(fingerprint, canonical_fingerprint), 1)
    return result


def function(text, marker):
    a = text.index(marker)
    brace = text.index('{', a)
    level, b = 1, brace + 1
    while level:
        level += (text[b] == '{') - (text[b] == '}')
        b += 1
    return text[a:b]


def factory_code(prepared, *, provider=False):
    text = explicit_layout_argument((prepared / 'ggml-webgpu-shader-lib.hpp').read_text())
    prefix = (ROOT / 'tests/webgpu-q8-short/host-prefix.cpp').read_text()
    prefix = prefix.replace('#include <vector>', '#include <vector>\n#include <array>\n#include <optional>\n#include <set>\n#include <functional>\n#include "ggml-webgpu-q8-diagnostic-sha256.hpp"\n#include "ggml-webgpu-q8-subgroup-shaders.hpp"')
    prefix = prefix.replace('    std::string wgsl;', '    std::string wgsl;\n    std::array<uint8_t,32> diagnostic_sha256{};\n    unsigned pipeline=0;')
    prefix = prefix.replace('static unsigned created = 0;', 'static unsigned created = 0;\nstatic std::function<void()> prepare_hook;')
    prefix = prefix.replace('    return {++created, name, nullptr, code};', '    if(prepare_hook){auto hook=std::move(prepare_hook);prepare_hook={};hook();}\n    webgpu_pipeline result{++created,name,nullptr,code};result.pipeline=result.id;return result;')
    chunks = [prefix]
    for marker in ('template <typename T> inline void ggml_webgpu_hash_combine(', 'struct ggml_webgpu_shader_lib_context {',
                   'inline uint32_t ggml_webgpu_q8_short_workgroup(', 'inline uint32_t ggml_webgpu_q4_short_workgroup(',
                   'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size(', 'inline bool ggml_webgpu_can_use_mmvq('):
        chunks.append(function(text, marker) + (';' if marker.startswith('struct ') else ''))
    a = text.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {')
    chunks.append(text[a:text.index('struct ggml_webgpu_quantize_q8_pipeline_key {', a)])
    chunks.append('class RecordingLibrary { int device=0;pre_wgsl::Preprocessor preprocessor;\n'
                  'std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;\npublic:\n')
    a = text.index('    static ggml_webgpu_mul_mat_vec_pipeline_key make_mul_mat_vec_key(')
    chunks.append(text[a:text.index('    webgpu_pipeline get_mul_mat_fast_pipeline(', a)])
    if provider:
        chunks.append('webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context &,bool,bool){throw std::runtime_error("unexpected matrix");}')
    chunks.append('};')
    return '\n'.join(chunks)


class Q8Diagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='q8-diagnostic-tests-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.normal = source.prepare(cls.upstream, cls.directory / 'normal').parent
        cls.profile = source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True).parent
        shader = cls.upstream / diagnostic.BASE / 'wgsl-shaders'
        subprocess.run([sys.executable, str(shader / 'embed_wgsl.py'), '--input_dir', str(shader),
                        '--output_file', str(cls.directory / 'ggml-wgsl-shaders.hpp')], check=True)

    def compile(self, code, name, extra=()):
        cpp = self.directory / (name + '.cpp')
        cpp.write_text(code)
        exe = self.directory / name
        command = ['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-I' + str(self.profile),
                   '-I' + str(self.directory), '-I' + str(ROOT / 'bridge'),
                   '-isystem', str(self.upstream / diagnostic.BASE), '-isystem', str(self.upstream / 'ggml/include'),
                   '-isystem', str(self.upstream / 'ggml/src')]
        subprocess.run(command + [str(cpp)] + list(extra) + ['-o', str(exe)], check=True)
        return exe

    def test_actual_factory_cache_auto_and_processed_wgsl_digest(self):
        normal = explicit_layout_argument((self.normal / 'ggml-webgpu-shader-lib.hpp').read_text())
        reference = ('class ReferenceLibrary { int device=0; pre_wgsl::Preprocessor preprocessor; '
                     'std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines; public: '
                     + function(normal, '    webgpu_pipeline get_mul_mat_vec_pipeline(') + '};')
        code = factory_code(self.profile) + '\n' + reference + '\n' + (HERE / 'factory-cases.cpp').read_text()
        executable = self.compile(code, 'factory')
        shaders = self.directory / 'shaders'; shaders.mkdir()
        subprocess.run([str(executable), str(shaders)], check=True)
        self.assertEqual(len(list(shaders.glob('*.wgsl'))), 25)
        for shader in shaders.glob('*.wgsl'):
            self.assertEqual(hashlib.sha256(shader.read_bytes()).hexdigest(), shader.with_suffix('.sha256').read_text())

    def test_actual_provider_params_submission_and_suspension_lifetime(self):
        backend = (self.profile / 'ggml-webgpu.cpp').read_text()
        library = explicit_layout_argument((self.profile / 'ggml-webgpu-shader-lib.hpp').read_text())
        code = factory_code(self.profile, provider=True).replace(
            'extern "C" const ggml_type_traits * ggml_get_type_traits(',
            'extern "C" __attribute__((weak)) const ggml_type_traits * ggml_get_type_traits(')
        code += '\n' + (HERE / 'provider-prefix.cpp').read_text()
        for marker in ('struct webgpu_encoded_op {', 'struct webgpu_dispatch_desc {',
                       'struct ggml_webgpu_merged_binding_range {', 'struct webgpu_profile_operation_guard {',
                       'struct webgpu_dispatch_view {'):
            code += '\n' + function(backend, marker) + ';'
        code += '\n' + function(library, 'struct ggml_webgpu_mul_mat_shader_decisions {') + ';'
        for marker in ('static inline void compute_2d_workgroups(', 'static size_t ggml_webgpu_tensor_offset(',
                       'static wgpu::Buffer ggml_webgpu_tensor_buf(',
                       'static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor *',
                       'static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor *',
                       'static size_t ggml_webgpu_tensor_misalignment(webgpu_context &',
                       'static size_t ggml_webgpu_tensor_align_offset(webgpu_context &',
                       'static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor *',
                       'static size_t ggml_webgpu_tensor_binding_size(webgpu_context &',
                       'static bool ggml_webgpu_tensor_binding_overlap(const webgpu_global_context &',
                       'static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(',
                       'static uint32_t ggml_webgpu_tensor_merged_element_offset(',
                       'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(',
                       'static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(',
                       'static webgpu_encoded_op ggml_backend_webgpu_build_impl(',
                       'static webgpu_encoded_op ggml_backend_webgpu_build_multi(',
                       'static bool ggml_webgpu_mul_mat_vec4_aligned(',
                       'static ggml_webgpu_shader_lib_context ggml_webgpu_mat_context('):
            code += '\n' + ('template<class Dispatch>\n' if 'ggml_backend_webgpu_build_impl(' in marker else '') + function(backend, marker)
        code += '\nstatic void ggml_webgpu_quantize_q8_dispatch(webgpu_context &,ggml_tensor *,ggml_tensor *,ggml_tensor *,std::vector<webgpu_dispatch_desc> &){throw std::runtime_error("unexpected MMVQ");}'
        code += '\ntemplate<bool Diagnostic>\n' + function(backend, 'static webgpu_encoded_op ggml_webgpu_mul_mat_impl(')
        code += '\nstatic std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context & ctx,ggml_cgraph * g,int i,int & count){count=1;auto * d=g->nodes[i];return ggml_webgpu_mul_mat_impl<false>(ctx,d->src[0],d->src[1],d);}'
        code += '\n' + function(backend, 'static void ggml_backend_webgpu_collect_profile_results(')
        code += '\ntemplate<class Encode>\n' + function(backend, 'static ggml_status ggml_backend_webgpu_graph_compute_impl(')
        for marker in ('static ggml_status ggml_backend_webgpu_graph_compute(', 'static ggml_guid_t ggml_backend_webgpu_guid('):
            code += '\n' + function(backend, marker)
        marker = 'static void * ggml_backend_webgpu_buffer_get_base('
        code += '\n' + function(backend[backend.rindex(marker):], marker)
        for marker in ('struct webgpu_q8_diagnostic_binding {',):
            code += '\n' + function(backend, marker) + ';'
        for marker in ('static bool ggml_webgpu_q8_diagnostic_binding(', 'static bool ggml_webgpu_q8_diagnostic_shape(',
                       'static int32_t ggml_backend_webgpu_q8_diagnostic_v1('):
            code += '\n' + function(backend, marker)
        code += '\n' + (HERE / 'provider-cases.cpp').read_text()
        library_path = os.environ.get('LCB_TEST_GGML_BASE_LIBRARY')
        if library_path is None:
            build = self.directory / 'native-base'
            subprocess.run(['cmake', '-S', str(self.upstream), '-B', str(build), '-DBUILD_SHARED_LIBS=OFF',
                            '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF', '-DGGML_CCACHE=OFF', '-DLLAMA_BUILD_TESTS=OFF',
                            '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_COMMON=OFF'],
                           check=True, capture_output=True)
            subprocess.run(['cmake', '--build', str(build), '--target', 'ggml-base', '-j', '2'], check=True, capture_output=True)
            library_path = str(build / 'ggml/src/libggml-base.a')
        executable = self.compile(code, 'provider', ['-DGGML_WEBGPU_GPU_PROFILE=1', '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS',
                                  '-Wno-unused-function', library_path, '-pthread', '-ldl', '-lm'])
        subprocess.run([str(executable)], check=True)

    def generate_extension(self):
        directory = self.directory / 'extension'
        directory.mkdir(exist_ok=True)
        (directory / 'base-exports.json').write_text('["_lcb_malloc"]')
        (directory / 'base-jspi.json').write_text('["lcb_ggml_init"]')
        subprocess.run([sys.executable, str(ROOT / 'scripts/generate_webgpu_q8_diagnostic.py'),
                        '--source', str(self.upstream), '--contract-header', str(ROOT / 'bridge/webgpu-q8-diagnostic.h'),
                        '--base-exports', str(directory / 'base-exports.json'), '--base-jspi', str(directory / 'base-jspi.json'),
                        '--output', str(directory)], check=True)
        return directory, json.loads((directory / 'q8-diagnostic-v1.schema.json').read_text())

    def test_typed_registry_bridge_capacity_compiled_schema_and_real_layout(self):
        directory, schema = self.generate_extension()
        self.assertEqual(json.loads((directory / 'base-exports.json').read_text()), ['_lcb_malloc'])
        self.assertEqual(json.loads((directory / 'q8-diagnostic-jspi-exports.json').read_text()),
                         ['lcb_ggml_init', 'lcb_webgpu_q8_diagnostic_run_v1'])
        canonical = dict(schema); fingerprint = canonical.pop('schemaSha256')
        self.assertEqual(hashlib.sha256((json.dumps(canonical, indent=2) + '\n').encode()).hexdigest(), fingerprint)
        prefix = '#include "ggml-backend-impl.h"\n#include "ggml-backend-host-snapshot.h"\n#include "webgpu-q8-diagnostic.h"\n#include <cstring>\n#include <iostream>\n#include <stdexcept>\n'
        prefix += '#undef GGML_ABORT\n#define GGML_ABORT(...) throw std::runtime_error("assert")\n'
        generic = (self.upstream / 'ggml/src/ggml-backend.cpp').read_text()
        for name in ('ggml_backend_dev_t ggml_backend_get_device(', 'ggml_backend_reg_t ggml_backend_dev_backend_reg(',
                     'void * ggml_backend_reg_get_proc_address(', 'const char * ggml_backend_reg_name(',
                     'enum ggml_backend_dev_type ggml_backend_dev_type('):
            prefix += function(generic, name) + '\n'
        code = (HERE / 'bridge-cases.cpp').read_text()
        code = code.replace('// @ACTUAL_REGISTRY@', function((self.profile / 'ggml-webgpu.cpp').read_text(),
                                                           'static void * ggml_backend_webgpu_reg_get_proc_address('))
        layouts = []
        for record in schema['records']:
            rid, typ = record['id'], record['cType']
            layouts += [f'check(lcb_webgpu_q8_diagnostic_sizeof_record({rid})==sizeof({typ}));',
                        f'check(lcb_webgpu_q8_diagnostic_alignof_record({rid})==alignof({typ}));']
            for field in record['fields']:
                fid, name = field['id'], field['name']
                layouts += [f'check(lcb_webgpu_q8_diagnostic_offsetof_field({rid},{fid})==offsetof({typ},{name}));',
                            f'check(lcb_webgpu_q8_diagnostic_sizeof_field({rid},{fid})==sizeof((({typ} *)0)->{name}));']
        code = prefix + code.replace('// @GENERATED_LAYOUT_CHECKS@', '\n'.join(layouts))
        executable = self.compile(code, 'bridge', ['-DLCB_WEBGPU_Q8_DIAGNOSTIC=1', str(directory / 'q8-diagnostic-metadata.cpp')])
        subprocess.run([str(executable), fingerprint], check=True)

    def test_upstream_sha_known_vectors_and_block_boundaries(self):
        code = '#include "ggml-webgpu-q8-diagnostic-sha256.hpp"\n#include <iostream>\n#include <string>\n'
        code += 'int main(int n,char ** a){for(int i=1;i<n;i++){std::string s=a[i];uint8_t digest[32];ggml_webgpu_q8_digest::sha256_ctx c;ggml_webgpu_q8_digest::sha256_init(c);for(char ch:s)ggml_webgpu_q8_digest::sha256_update(c,&ch,1);ggml_webgpu_q8_digest::sha256_final(c,digest);for(uint8_t b:digest){static const char hex[]="0123456789abcdef";std::cout<<hex[b>>4]<<hex[b&15];}std::cout<<"\\n";}}'
        executable = self.compile(code, 'sha-vectors')
        messages = ['', 'abc', 'a' * 55, 'a' * 56, 'a' * 64, 'a' * 129, 'abc' * 1000]
        actual = subprocess.check_output([str(executable), *messages], text=True).splitlines()
        self.assertEqual(actual, [hashlib.sha256(value.encode()).hexdigest() for value in messages])

    def test_normal_outputs_and_base_binding_generation_remain_exact(self):
        identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in identity['public_source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(source.profile_query_budget.INPUT_SHA256, source.OUTPUT_SHA256)
        self.assertEqual(diagnostic.INPUTS['ggml-webgpu.cpp'], source.profile_query_budget.OUTPUT_SHA256)
        for name, expected in source.FINAL_OUTPUTS[False].items():
            self.assertEqual(hashlib.sha256((self.normal / name).read_bytes()).hexdigest(), expected)
        self.assertFalse((self.normal / 'ggml-webgpu-q8-diagnostic.h').exists())
        for name, expected in source.FINAL_OUTPUTS[True].items():
            self.assertEqual(hashlib.sha256((self.profile / name).read_bytes()).hexdigest(), expected)
        for prepared in (self.normal, self.profile):
            self.assertEqual(hashlib.sha256((prepared / source.iq3s_row_split.HEADER_NAME).read_bytes()).hexdigest(), source.iq3s_row_split.HEADER_SHA256)
        original = (self.upstream / diagnostic.SHA_SOURCE).read_text()
        fragment = original[original.index(diagnostic.SHA_BEGIN):original.index(diagnostic.SHA_END)]
        self.assertIn(fragment, (self.profile / 'ggml-webgpu-q8-diagnostic-sha256.hpp').read_text())
        self.assertNotIn('q8_diagnostic', (ROOT / 'scripts/generate_bindings.py').read_text())

    def test_real_base_generation_has_no_extension_or_schema_change(self):
        import generate_bindings
        directory = self.directory / 'base-generated'
        generate_bindings.generate(self.upstream, directory, 'clang')
        baseline = ROOT / 'tests/webgpu-q8-diagnostic/inputs.json'
        identity = json.loads(baseline.read_text())
        outputs = {name: (directory / name).read_text() for name in identity['base_generated_normalized_sha256']}
        normalized = normalized_base_generation(outputs, self.upstream)
        for name, expected in identity['base_generated_normalized_sha256'].items():
            self.assertEqual(hashlib.sha256(normalized[name].encode()).hexdigest(), expected, name)
        schema = json.loads((directory / 'schema.json').read_text())
        self.assertFalse(any('diagnostic' in entry['name'] for entry in schema['functions']))

    def test_base_generation_normalizes_only_known_anonymous_locations(self):
        import generate_bindings
        directory = self.directory / 'base-portability'
        generate_bindings.generate(self.upstream, directory, 'clang')
        names = json.loads((HERE / 'inputs.json').read_text())['base_generated_normalized_sha256']
        outputs = {name: (directory / name).read_text() for name in names}
        expected = normalized_base_generation(outputs, self.upstream)
        raw_fingerprint = hashlib.sha256(outputs['schema.json'].encode()).hexdigest()

        def with_schema(schema):
            text = json.dumps(schema, indent=2) + '\n'
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            return {**outputs, 'schema.json': text,
                    'schema.mjs': 'export default ' + json.dumps({**schema, 'schemaSha256': fingerprint}, separators=(',', ':')) + ';\n',
                    'bindings.cpp': outputs['bindings.cpp'].replace('"' + raw_fingerprint + '"', '"' + fingerprint + '"')}

        for root in (Path('/fixture/checkout-one/upstream'), Path('/other/checkout-two/upstream')):
            for spelling in ('unnamed struct', 'unnamed'):
                schema = json.loads(outputs['schema.json'])
                fields = next(record['fields'] for record in schema['records'] if record['name'] == 'ggml_opt_optimizer_params')
                for field in fields:
                    field['cType'] = re.sub(r'struct \(unnamed(?: struct)? at ' + re.escape(str(self.upstream)),
                                           'struct (' + spelling + ' at ' + str(root), field['cType'])
                relocated = with_schema(schema)
                with self.subTest(root=root, spelling=spelling):
                    self.assertNotEqual(relocated['schema.json'], outputs['schema.json'])
                    self.assertEqual(normalized_base_generation(relocated, root), expected)

        for change in ('field-type', 'record-name', 'source-line', 'exports'):
            schema = json.loads(outputs['schema.json'])
            record = next(record for record in schema['records'] if record['name'] == 'ggml_opt_optimizer_params')
            if change == 'field-type': record['fields'][0]['cType'] = 'float'
            if change == 'record-name': record['name'] = 'ggml_unexpected_record'
            if change == 'source-line': record['fields'][0]['cType'] = record['fields'][0]['cType'].replace(':86:9)', ':87:9)')
            mutated = with_schema(schema)
            if change == 'exports': mutated['exports.json'] = json.dumps(json.loads(mutated['exports.json']) + ['_lcb_unapproved_export'], indent=2) + '\n'
            with self.subTest(change=change):
                self.assertNotEqual(normalized_base_generation(mutated, self.upstream), expected)

        for name in ('schema.json', 'schema.mjs', 'bindings.cpp'):
            corrupted = dict(outputs)
            corrupted[name] = (outputs[name] + '\n' if name == 'schema.json' else
                               outputs[name].replace(raw_fingerprint, '0' * 64))
            with self.subTest(corrupted=name), self.assertRaisesRegex(AssertionError, 'raw schema'):
                normalized_base_generation(corrupted, self.upstream)

    def test_full_prepared_normal_and_profile_cpp_and_bridge_syntax(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir(): self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items():
            self.assertEqual(hashlib.sha256((dawn / name).read_bytes()).hexdigest(), expected)
        command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4', '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS',
                   '-I' + str(self.directory), '-I' + str(self.upstream / diagnostic.BASE),
                   '-I' + str(self.upstream / 'ggml/include'), '-I' + str(self.upstream / 'ggml/src'),
                   '-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
        for directory, flags in ((self.normal, []), (self.profile, ['-DGGML_WEBGPU_GPU_PROFILE=1']),
                                 (self.profile, ['-DGGML_WEBGPU_GPU_PROFILE=1', '-DGGML_WEBGPU_CPU_PROFILE'])):
            subprocess.run(command + ['-I' + str(directory)] + flags + [str(directory / 'ggml-webgpu.cpp')], check=True)
        generated, _ = self.generate_extension()
        subprocess.run(['c++', '-std=c++17', '-fsyntax-only', '-DLCB_WEBGPU_Q8_DIAGNOSTIC=1',
                        '-I' + str(ROOT / 'bridge'), '-I' + str(self.upstream / 'ggml/include'),
                        str(ROOT / 'bridge/webgpu-q8-diagnostic.cpp'), str(generated / 'q8-diagnostic-metadata.cpp')], check=True)

    def test_actual_cmake_extension_targets_and_on_off_reconfiguration(self):
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        begin = cmake.index('# Keep the common generated ABI')
        block = cmake[begin:cmake.index('if(LCB_NATIVE_CHECK)', begin)].replace('${CMAKE_CURRENT_SOURCE_DIR}', '${LCB_SOURCE_ROOT}')
        tail = cmake[cmake.index('if(LCB_WEBGPU AND GGML_WEBGPU_GPU_PROFILE)'):].replace('${CMAKE_CURRENT_SOURCE_DIR}', '${LCB_SOURCE_ROOT}').replace('PRIVATE bridge/', 'PRIVATE ${LCB_SOURCE_ROOT}/bridge/')
        project = self.directory / 'cmake-fixture'; project.mkdir()
        (project / 'empty.cpp').write_text('int empty;\n')
        (project / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.24)\nproject(q8_diagnostic_configure CXX)\n'
            'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\nfind_package(Python3 REQUIRED COMPONENTS Interpreter)\nfind_program(LCB_CLANG clang REQUIRED)\n'
            f'set(LCB_SOURCE_ROOT "{ROOT}")\nset(LCB_LLAMA_SOURCE "{self.upstream}")\n'
            'set(LCB_WEBGPU ON)\nset(LCB_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")\nfile(MAKE_DIRECTORY "${LCB_GENERATED}")\n'
            'file(WRITE "${LCB_GENERATED}/exports.json" "[]")\nfile(WRITE "${LCB_GENERATED}/jspi-exports.json" "[]")\n'
            + block + '\nadd_library(core STATIC empty.cpp)\n' + tail)
        build = self.directory / 'cmake-build'
        for enabled in (True, False, True):
            configured = subprocess.run(['cmake', '-S', str(project), '-B', str(build), '-DGGML_WEBGPU_GPU_PROFILE=' + ('ON' if enabled else 'OFF')], capture_output=True, text=True)
            self.assertEqual(configured.returncode, 0, configured.stderr)
            files = [Path(entry['file']).name for entry in json.loads((build / 'compile_commands.json').read_text())]
            self.assertEqual('webgpu-q8-diagnostic.cpp' in files, enabled)
            self.assertEqual('q8-diagnostic-metadata.cpp' in files, enabled)
            self.assertEqual((build / 'runtime/q8-diagnostic-v1.schema.json').exists(), enabled)
            self.assertEqual((build / 'runtime/q8-diagnostic-v1.functions.d.ts').exists(), enabled)
            self.assertEqual((build / 'generated/exports.json').read_text(), '[]')
            self.assertEqual((build / 'generated/jspi-exports.json').read_text(), '[]')
        # The source overlay also removes its own stale profile-only sibling headers.
        source.prepare(self.upstream, self.profile)
        self.assertEqual(hashlib.sha256((self.profile / 'ggml-webgpu.cpp').read_bytes()).hexdigest(), source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'])
        self.assertFalse((self.profile / 'ggml-webgpu-q8-diagnostic.h').exists())
        source.prepare(self.upstream, self.profile, gpu_profile=True)

    def test_extension_generator_rejects_drift_before_writing_files(self):
        import copy
        ast = json.loads(subprocess.check_output(['clang', '-x', 'c', '-std=c11', '-fsyntax-only',
                            '-I' + str(self.upstream / 'ggml/include'), '-Xclang', '-ast-dump=json',
                            str(ROOT / 'bridge/webgpu-q8-diagnostic.h')], text=True))
        for mode in ('signature', 'field', 'variadic', 'collision', 'jspi-collision'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                changed = copy.deepcopy(ast)
                for node in extension.walk(changed):
                    if mode == 'signature' and node.get('kind') == 'FunctionDecl' and node.get('name', '').endswith('diagnostic_run_v1'):
                        next(item for item in node['inner'] if item.get('kind') == 'ParmVarDecl')['type']['qualType'] = 'uint32_t'
                    if mode == 'variadic' and node.get('kind') == 'FunctionDecl' and node.get('name', '').endswith('diagnostic_run_v1'):
                        node['variadic'] = True
                    if mode == 'field' and node.get('kind') == 'FieldDecl' and node.get('name') == 'shader_sha256':
                        node['type']['qualType'] = 'uint8_t[16]'
                output = Path(tmp) / 'missing'
                with self.assertRaises(ValueError):
                    extension.generate(changed, output, ROOT / 'bridge/webgpu-q8-diagnostic.h',
                        ['_lcb_webgpu_q8_diagnostic_version'] if mode == 'collision' else [],
                        ['lcb_webgpu_q8_diagnostic_run_v1'] if mode == 'jspi-collision' else [])
                self.assertFalse(output.exists())

    def test_profile_patch_contract_outputs_and_sha_drift_fail_closed(self):
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            values = getattr(diagnostic, field)
            for name in values:
                with self.subTest(field=field, name=name), patch.object(diagnostic, field, {**values, name: '0' * 64}):
                    with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                        source.prepare(self.upstream, Path(tmp), gpu_profile=True)
        for field in ('PATCH_SHA256', 'BRIDGE_HEADER_SHA256', 'SHA_HEADER_SHA256', 'SHA_FRAGMENT_SHA256'):
            with self.subTest(field=field), patch.object(diagnostic, field, '0' * 64):
                with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                    source.prepare(self.upstream, Path(tmp), gpu_profile=True)


if __name__ == '__main__': unittest.main()
