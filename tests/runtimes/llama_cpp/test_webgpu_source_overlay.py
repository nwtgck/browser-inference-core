"""Check composed source preparation and real target configuration, not GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp import source as source
from build_tools.source.patches import patch_environment
from build_tools.source.wgsl import input_files as canonical_shader_inputs



GENERATED_SHADERS = ('ggml-webgpu-scale-unary-shaders.hpp', 'ggml-webgpu-q8-subgroup-shaders.hpp', 'ggml-webgpu-rms-scale-shaders.hpp', 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp', 'ggml-webgpu-ssm-conv-silu-shaders.hpp', 'ggml-webgpu-mul-mat-id-group-shaders.hpp')


class WebgpuSourceOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / 'ggml/src/ggml-webgpu/ggml-webgpu.cpp').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')
        temporary = tempfile.TemporaryDirectory(prefix='current-source-reference-')
        cls.addClassCleanup(temporary.cleanup)
        expected = Path(temporary.name) / 'prepared'
        source.prepare(cls.upstream, expected)
        cls.expected = {p.name: p.read_bytes() for p in expected.iterdir() if p.is_file()}

    def test_combined_source_and_reuse_preserve_input(self):
        before = (self.upstream / 'ggml/src/ggml-webgpu/ggml-webgpu.cpp').read_bytes()
        loader_before = (self.upstream / 'src/llama-model-loader.cpp').read_bytes()
        sampler_before = (self.upstream / 'src/llama-sampler.cpp').read_bytes()
        backend_before = (self.upstream / 'ggml/src/ggml-backend.cpp').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            result = source.prepare(self.upstream, Path(tmp))
            self.assertEqual(result.read_bytes(), self.expected['ggml-webgpu.cpp'])
            self.assertEqual((result.parent / 'ggml-webgpu-shader-lib.hpp').read_bytes(), self.expected['ggml-webgpu-shader-lib.hpp'])
            self.assertEqual((result.parent / 'llama-model-loader.cpp').read_bytes(), self.expected['llama-model-loader.cpp'])
            self.assertEqual((result.parent / 'llama-sampler.cpp').read_bytes(), self.expected['llama-sampler.cpp'])
            self.assertEqual((result.parent / 'ggml-backend.cpp').read_bytes(), self.expected['ggml-backend.cpp'])
            self.assertEqual((result.parent / 'ggml-backend-host-snapshot.h').read_bytes(), self.expected['ggml-backend-host-snapshot.h'])
            for name in GENERATED_SHADERS:
                self.assertEqual((result.parent / name).read_bytes(), self.expected[name])
            header_stamp = (result.parent / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns
            stamp = result.stat().st_mtime_ns
            loader_stamp = (result.parent / 'llama-model-loader.cpp').stat().st_mtime_ns
            sampler_stamp = (result.parent / 'llama-sampler.cpp').stat().st_mtime_ns
            backend_stamp = (result.parent / 'ggml-backend.cpp').stat().st_mtime_ns
            snapshot_header_stamp = (result.parent / 'ggml-backend-host-snapshot.h').stat().st_mtime_ns
            generated_stamps = {name: (result.parent / name).stat().st_mtime_ns
                                for name in GENERATED_SHADERS}
            self.assertEqual(source.prepare(self.upstream, Path(tmp)).stat().st_mtime_ns, stamp)
            self.assertEqual((result.parent / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns, header_stamp)
            self.assertEqual((result.parent / 'llama-model-loader.cpp').stat().st_mtime_ns, loader_stamp)
            self.assertEqual((result.parent / 'llama-sampler.cpp').stat().st_mtime_ns, sampler_stamp)
            self.assertEqual((result.parent / 'ggml-backend.cpp').stat().st_mtime_ns, backend_stamp)
            self.assertEqual((result.parent / 'ggml-backend-host-snapshot.h').stat().st_mtime_ns, snapshot_header_stamp)
            for name, previous in generated_stamps.items():
                self.assertEqual((result.parent / name).stat().st_mtime_ns, previous)
        self.assertEqual(before, (self.upstream / 'ggml/src/ggml-webgpu/ggml-webgpu.cpp').read_bytes())
        self.assertEqual(loader_before, (self.upstream / 'src/llama-model-loader.cpp').read_bytes())
        self.assertEqual(sampler_before, (self.upstream / 'src/llama-sampler.cpp').read_bytes())
        self.assertEqual(backend_before, (self.upstream / 'ggml/src/ggml-backend.cpp').read_bytes())

    def test_repository_nested_builds_apply_all_patches_without_touching_git_state(self):
        upstream_before = {name: (self.upstream / name).read_bytes()
                           for name in ('ggml/src/ggml-webgpu/ggml-webgpu.cpp', 'src/llama-model-loader.cpp', 'src/llama-sampler.cpp',
                                        'ggml/src/ggml-backend.cpp')}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkout = root / 'checkout'
            subprocess.run(['git', 'init', '--quiet', str(checkout)], check=True)
            (checkout / '.gitignore').write_text('llama-cpp/build/\n')
            subprocess.run(['git', 'add', '.gitignore'], cwd=checkout, check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            'commit', '--quiet', '-m', 'fixture'], cwd=checkout, check=True)
            worktree = root / 'worktree'
            subprocess.run(['git', 'worktree', 'add', '--quiet', '--detach', str(worktree)],
                           cwd=checkout, check=True)
            self.assertTrue((worktree / '.git').is_file())
            for parent in (checkout, worktree):
                (parent / 'keep-untracked.txt').write_text('must remain unchanged')
                status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=parent)
                index = subprocess.check_output(['git', 'ls-files', '--stage'], cwd=parent)
                with self.subTest(worktree=parent == worktree):
                    build = parent / 'llama-cpp/build/webgpu-wasm32-jspi/test/webgpu-source-overlay'
                    result = source.prepare(self.upstream, build)
                    self.assertEqual(result.read_bytes(), self.expected['ggml-webgpu.cpp'])
                    self.assertEqual((build / 'llama-model-loader.cpp').read_bytes(), self.expected['llama-model-loader.cpp'])
                    self.assertEqual((build / 'llama-sampler.cpp').read_bytes(), self.expected['llama-sampler.cpp'])
                    self.assertEqual((build / 'ggml-backend.cpp').read_bytes(), self.expected['ggml-backend.cpp'])
                    self.assertEqual((build / 'ggml-backend-host-snapshot.h').read_bytes(), self.expected['ggml-backend-host-snapshot.h'])
                    stamp = result.stat().st_mtime_ns
                    self.assertEqual(source.prepare(self.upstream, build).stat().st_mtime_ns, stamp)
                    self.assertEqual(subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=parent), status)
                    self.assertEqual(subprocess.check_output(['git', 'ls-files', '--stage'], cwd=parent), index)
                    self.assertEqual((parent / 'keep-untracked.txt').read_text(), 'must remain unchanged')
        for name, before in upstream_before.items():
            self.assertEqual((self.upstream / name).read_bytes(), before)

    def test_patch_environment_does_not_inherit_repository_selectors(self):
        context = {key: '/enclosing/repository' for key in
                   ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE')}
        with patch.dict(os.environ, {**context, 'GIT_CEILING_DIRECTORIES': '/wrong', 'KEEP_ME': 'value'}):
            env = patch_environment(Path('/build/overlay/scratch'))
            self.assertEqual(env['GIT_CEILING_DIRECTORIES'], '/build/overlay')
            self.assertEqual(env['KEEP_ME'], 'value')
            for key in context:
                self.assertNotIn(key, env)
                self.assertEqual(os.environ[key], context[key])

    def test_new_upstream_kernels_and_scratch_contract_survive_composition(self):
        for profile in (False, True):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                result = source.prepare(self.upstream, Path(tmp), gpu_profile=profile)
                text = result.read_text()
                library = (result.parent / 'ggml-webgpu-shader-lib.hpp').read_text()
                allocation = text[text.index('static size_t ggml_backend_webgpu_buffer_type_get_alloc_size('):]
                self.assertIn('const int64_t n_rows   = std::min<int64_t>(src1->ne[1], GGML_WEBGPU_MUL_MAT_VEC_MAX_ROWS);', allocation)
                self.assertIn('src0, src1, n_rows, ctx->webgpu_global_ctx->capabilities.supports_dot_product,', allocation)
                self.assertNotIn('src0, src1, src1->ne[1], n_rows,', text)
                self.assertIn('src1->ne[1]', library[library.index('ggml_webgpu_can_use_mmvq(context.src0'):])
                self.assertIn('wgsl_arange', library)
                self.assertIn('wgsl_arange', (result.parent / 'ggml-webgpu-aligned-word-load-shaders.hpp').read_text())
                self.assertIn(self.upstream / 'ggml/src/ggml-webgpu/wgsl-shaders/arange.wgsl',
                              canonical_shader_inputs(self.upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'))
                for hook in ('set_tensor_n', 'get_tensor_n', 'cpy_tensor_n'):
                    self.assertIn('/* .' + hook, text)


    @unittest.skipUnless(shutil.which('cmake'), 'CMake required')
    def test_actual_target_preserves_upstream_shaders_on_reconfiguration(self):
        dawn = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        if not dawn:
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        before = (self.upstream / 'ggml/src/ggml-webgpu/ggml-webgpu.cpp').read_bytes()
        loader_before = (self.upstream / 'src/llama-model-loader.cpp').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['git', 'init', '--quiet', tmp], check=True)
            build = Path(tmp) / 'llama-cpp/build/overlay-check'
            stale = build / 'moe-direct-slot-overlay'
            stale.mkdir(parents=True)
            (stale / 'ggml-wgsl-shaders.hpp').write_text('#error retired MoE overlay used\n')
            for _ in range(2):
                command = ['cmake', '-S', str(runtime_test_root('llama-cpp') / 'webgpu-tensor-copy'), '-B', str(build),
                           '-DLCB_LLAMA_SOURCE=' + str(self.upstream), '-DEMDAWNWEBGPU_DIR=' + dawn]
                subprocess.run(command, check=True, capture_output=True, text=True)
                entries = json.loads((build / 'compile_commands.json').read_text())
                compiled = [entry for entry in entries if Path(entry['file']).name == 'ggml-webgpu.cpp']
                self.assertEqual(len(compiled), 1)
                self.assertEqual(Path(compiled[0]['file']), build / 'webgpu-source-overlay/ggml-webgpu.cpp')
                self.assertEqual((Path(compiled[0]['file'])).read_bytes(), self.expected['ggml-webgpu.cpp'])
                self.assertIn('GGML_WEBGPU_BATCH_PARAM_UPLOADS', compiled[0]['command'])
                self.assertEqual((build / 'webgpu-source-overlay/ggml-webgpu-shader-lib.hpp').read_bytes(), self.expected['ggml-webgpu-shader-lib.hpp'])
                self.assertFalse((build / 'webgpu-source-overlay/ggml-wgsl-shaders.hpp').exists())
                canonical = build / 'webgpu-source-overlay' / 'ggml-webgpu-aligned-word-load-shaders.hpp'
                self.assertIn('wgsl_arange', canonical.read_text())
                before_sources = (build / 'before.txt').read_text().split(';')
                after_sources = (build / 'after.txt').read_text().split(';')
                self.assertEqual(after_sources, [name for name in before_sources if name != 'ggml-webgpu.cpp'] +
                                 [str(build / 'webgpu-source-overlay/ggml-webgpu.cpp'), str(canonical)])
                loaders = [entry for entry in entries if Path(entry['file']).name == 'llama-model-loader.cpp']
                self.assertEqual(len(loaders), 1)
                self.assertEqual(Path(loaders[0]['file']), build / 'webgpu-source-overlay/llama-model-loader.cpp')
                self.assertEqual((Path(loaders[0]['file'])).read_bytes(), self.expected['llama-model-loader.cpp'])
                samplers = [entry for entry in entries if Path(entry['file']).name == 'llama-sampler.cpp']
                self.assertEqual(len(samplers), 1)
                self.assertEqual(Path(samplers[0]['file']), build / 'webgpu-source-overlay/llama-sampler.cpp')
                self.assertEqual((Path(samplers[0]['file'])).read_bytes(), self.expected['llama-sampler.cpp'])
                self.assertEqual((build / 'sampler-unity.txt').read_text(), 'ON')
                backends = [entry for entry in entries if Path(entry['file']).name == 'ggml-backend.cpp']
                self.assertEqual(len(backends), 1)
                self.assertEqual(Path(backends[0]['file']), build / 'webgpu-source-overlay/ggml-backend.cpp')
                self.assertEqual((Path(backends[0]['file'])).read_bytes(), self.expected['ggml-backend.cpp'])
                self.assertEqual((build / 'backend-unity.txt').read_text(), 'ON')
                self.assertIn(str(self.upstream / 'ggml/src'), backends[0]['command'])
                self.assertIn(str(build / 'webgpu-source-overlay'), backends[0]['command'])
                backend_before = (build / 'backend-before.txt').read_text().split(';')
                backend_after = (build / 'backend-sources.txt').read_text().split(';')
                self.assertEqual(backend_after, [str(build / 'webgpu-source-overlay/ggml-backend.cpp')
                                                 if name == 'ggml-backend.cpp' else name
                                                 for name in backend_before])
                before_names = [Path(path).name for path in (build / 'llama-before.txt').read_text().split(';')]
                after_names = [Path(path).name for path in (build / 'llama-sources.txt').read_text().split(';')]
                self.assertEqual([name for name in before_names if name != 'llama-model-loader.cpp'],
                                 [name for name in after_names if name != 'llama-model-loader.cpp'])
                self.assertNotIn('moe-direct-slot-overlay', (build / 'includes.txt').read_text())
                self.assertNotIn('moe-direct-slot-overlay', (build / 'compile_commands.json').read_text())
                mtmd_overlay = build / 'mtmd-source-overlay'
                manifest = json.loads((mtmd_overlay / 'mtmd-source-manifest.json').read_text())
                self.assertEqual(manifest['reviewedCommit'], next(iter(copy.REVIEWED_REVISIONS)))
                for name in ('mtmd-model.cpp', 'mtmd-subcontext.cpp', 'mtmd-audio.cpp', 'mtmd.cpp'):
                    self.assertTrue((mtmd_overlay / name).is_file())
                    self.assertEqual(copy.digest(mtmd_overlay / name), manifest['preparedFiles'][name]['sha256'])
        self.assertEqual(before, (self.upstream / 'ggml/src/ggml-webgpu/ggml-webgpu.cpp').read_bytes())
        self.assertEqual(loader_before, (self.upstream / 'src/llama-model-loader.cpp').read_bytes())

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required')
    def test_sampler_remains_selected_with_independent_model_overlays(self):
        dawn = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        if not dawn:
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        with tempfile.TemporaryDirectory() as tmp:
            for rows, qsa in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(rows=rows, qsa=qsa):
                    build = Path(tmp) / (str(rows) + '-' + str(qsa))
                    command = ['cmake', '-S', str(runtime_test_root('llama-cpp') / 'webgpu-tensor-copy'), '-B', str(build),
                               '-DLCB_LLAMA_SOURCE=' + str(self.upstream), '-DEMDAWNWEBGPU_DIR=' + dawn,
                               '-DLCB_TEST_FILE_ROWS=' + ('ON' if rows else 'OFF'),
                               '-DLCB_TEST_QSA=' + ('ON' if qsa else 'OFF')]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    entries = json.loads((build / 'compile_commands.json').read_text())
                    samplers = [entry for entry in entries if Path(entry['file']).name == 'llama-sampler.cpp']
                    self.assertEqual(len(samplers), 1)
                    self.assertEqual(Path(samplers[0]['file']), build / 'webgpu-source-overlay/llama-sampler.cpp')
                    self.assertEqual((Path(samplers[0]['file'])).read_bytes(), self.expected['llama-sampler.cpp'])
                    self.assertEqual((build / 'sampler-unity.txt').read_text(), 'ON')
                    backends = [entry for entry in entries if Path(entry['file']).name == 'ggml-backend.cpp']
                    self.assertEqual(len(backends), 1)
                    self.assertEqual(Path(backends[0]['file']), build / 'webgpu-source-overlay/ggml-backend.cpp')
                    paths = (build / 'llama-sources.txt').read_text().split(';')
                    self.assertEqual(sum(Path(path).name == 'llama-sampler.cpp' for path in paths), 1)
                    self.assertEqual(sum(Path(path).name == 'llama-model-loader.cpp' for path in paths), 1)

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required')
    def test_backend_source_layout_drift_fails_closed(self):
        dawn = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        if not dawn:
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        with tempfile.TemporaryDirectory() as tmp:
            for layout in ('missing', 'duplicate'):
                with self.subTest(layout=layout):
                    result = subprocess.run(
                        ['cmake', '-S', str(runtime_test_root('llama-cpp') / 'webgpu-tensor-copy'), '-B', str(Path(tmp) / layout),
                         '-DLCB_LLAMA_SOURCE=' + str(self.upstream), '-DEMDAWNWEBGPU_DIR=' + dawn,
                         '-DLCB_TEST_BACKEND_LAYOUT=' + layout], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Upstream ggml backend source layout changed', result.stderr)








    def test_webgpu_is_the_only_source_overlay_build_switch(self):
        text = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('if(LCB_WEBGPU)\n    include(cmake/WebgpuSourceOverlay.cmake)', text)
        self.assertNotIn('LCB_WEBGPU_TENSOR_COPY', text)
        self.assertNotIn('LCB_WEBGPU_PARAM_UPLOAD_BATCHING', text)

    def test_rope_truncate_and_final_matmul_precision_rejection_survive_composition(self):
            from tests.runtimes.llama_cpp.test_webgpu_buffer_limit import definition
            original = (self.upstream / Path('ggml/src/ggml-webgpu/ggml-webgpu.cpp')).read_text()
            for profile in (False, True):
                with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    text = source.prepare(self.upstream, directory / 'prepared', gpu_profile=profile).read_text()
                    self.assertEqual(definition(text, 'static webgpu_encoded_op ggml_webgpu_rope('),
                                     definition(original, 'static webgpu_encoded_op ggml_webgpu_rope('))
                    self.assertIn('ggml_rope_get_truncate(dst)', definition(text, 'static webgpu_encoded_op ggml_webgpu_rope('))
                    support = definition(text, 'static bool ggml_backend_webgpu_device_supports_op(')
                    raw_support = definition(original, 'static bool ggml_backend_webgpu_device_supports_op(')
                    functions = []
                    for operation in ('MUL_MAT', 'MUL_MAT_ID'):
                        marker = 'case GGML_OP_' + operation + ':'
                        block = support.split(marker, 1)[1].split('\n        case ', 1)[0]
                        raw_block = raw_support.split(marker, 1)[1].split('\n        case ', 1)[0]
                        guard_marker = 'if (ggml_prec_get_src(op, 1)'
                        guard = definition(block, guard_marker)
                        self.assertEqual(guard, definition(raw_block, guard_marker))
                        self.assertNotIn('return true;', block)
                        self.assertRegex(block.split(guard, 1)[1], r'^\s*break;\s*\}?\s*$')
                        functions.append('bool supports_' + operation + '(bool initial,int precision){\n'
                            'ggml_tensor value{precision,0};const auto* op=&value;bool supports_op=initial;\n' +
                            guard + '\nreturn supports_op;}\n')
                    rope = support.split('case GGML_OP_ROPE:', 1)[1].split('break;', 1)[0]
                    raw_rope = raw_support.split('case GGML_OP_ROPE:', 1)[1].split('break;', 1)[0]
                    self.assertEqual(rope, raw_rope)
                    functions.append('bool supports_rope(int type,bool truncate){\n'
                        'ggml_tensor value{0,type};const auto* op=&value;bool supports_op=false;(void)truncate;\n' +
                        rope + '\nreturn supports_op;}\n')
                    # Execute the actual terminal guards after either selector result.
                    # Tensor bookkeeping is modeled; no backend or GPU is created.
                    precision_enum = definition((self.upstream / 'ggml/include/ggml.h').read_text(),
                                                'enum ggml_prec {') + ';\n'
                    code = '''#include <cassert>
    struct ggml_tensor {int precision,type;};
    ''' + precision_enum + '''
    enum {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0};
    int ggml_prec_get_src(const ggml_tensor* op,int index){assert(index==1);return op->precision;}
    ''' + ''.join(functions) + '''
    int main(){
    for(bool initial:{false,true})for(int precision:{GGML_PREC_UNDEFINED,GGML_PREC_F16,GGML_PREC_BF16,GGML_PREC_F32}){
    bool expected=initial && precision!=GGML_PREC_BF16 && precision!=GGML_PREC_F32;
    assert(supports_MUL_MAT(initial,precision)==expected);
    assert(supports_MUL_MAT_ID(initial,precision)==expected);
    }
    for(bool truncate:{false,true})for(int type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
    assert(supports_rope(type,truncate)==(type==GGML_TYPE_F32 || type==GGML_TYPE_F16));
    }
    '''
                    code = '#include <initializer_list>\n' + code
                    native = directory / 'admission.cpp'
                    native.write_text(code)
                    binary = directory / 'admission'
                    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', str(native), '-o', str(binary)], check=True)
                    subprocess.run([str(binary)], check=True)



if __name__ == '__main__':
    unittest.main()
