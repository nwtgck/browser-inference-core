"""Expert-sum source and host contracts; software GPU checks are separately opt-in."""
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy
import prepare_webgpu_moe_sum as moe

class MoeSumOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_reuse_and_unchanged_upstream(self):
        before = {name: copy.digest(self.upstream / name) for name in moe.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out)
            names = ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-moe-sum.hpp', moe.HEADER_NAME]
            stamps = {name: (out / name).stat().st_mtime_ns for name in names}
            for name in names[:-1]:
                self.assertEqual(copy.digest(out / name), moe.OUTPUTS[name])
            self.assertEqual(copy.digest(out / moe.HEADER_NAME), moe.HEADER_SHA256)
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            source.prepare(self.upstream, out)
            self.assertEqual(stamps, {name: (out / name).stat().st_mtime_ns for name in names})
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_fail_closed_contract_patch_composition_and_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
                for name in getattr(moe, field):
                    with self.subTest(field=field, path=name), patch.object(moe, field, {**getattr(moe, field), name: '0'*64}):
                        with self.assertRaises(ValueError): source.prepare(self.upstream, out)
                    self.assertFalse((out / 'ggml-webgpu.cpp').exists())
                    self.assertFalse((out / moe.HEADER_NAME).exists())
            for field in ('PATCH_SHA256', 'HEADER_SHA256'):
                with self.subTest(field=field), patch.object(moe, field, '0'*64):
                    with self.assertRaises(ValueError): source.prepare(self.upstream, out)
                self.assertFalse((out / 'ggml-webgpu.cpp').exists())

    def test_cmake_tracks_every_contract(self):
        text = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        for path in moe.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + path, text)
        self.assertIn(moe.PATCH_NAME, text)
        self.assertIn('scripts/prepare_webgpu_moe_sum.py', text)

    @unittest.skipUnless(shutil.which('g++') and shutil.which('cmake'), 'native build tools needed')
    def test_actual_graph_allocator_and_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); source.prepare(self.upstream, out / 'prepared')
            build = out / 'build'
            subprocess.run(['cmake', '-S', str(self.upstream), '-B', str(build),
                            '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
                            '-DGGML_CCACHE=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                            '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_CURL=OFF'],
                           check=True, capture_output=True)
            subprocess.run(['cmake', '--build', str(build), '--target', 'ggml-base', '-j', '2'],
                           check=True, capture_output=True)
            exe = out / 'admission'
            subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-isystem', str(self.upstream / 'ggml/include'), '-isystem', str(self.upstream / 'ggml/src'),
                            '-I' + str(out / 'prepared'), str(ROOT / 'tests/webgpu-moe-sum/admission.cpp'),
                            str(build / 'ggml/src/libggml-base.a'), '-pthread', '-ldl', '-lm', '-o', str(exe)],
                           check=True, capture_output=True)
            result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
            self.assertIn('actual ggml graph admission checks passed', result.stdout)
            print(result.stdout.strip())
            spec = importlib.util.spec_from_file_location('moe_dispatch', ROOT / 'tests/webgpu-moe-sum/check_dispatch.py')
            checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
            result = checker.check(self.upstream, out / 'prepared', build / 'ggml/src/libggml-base.a', out / 'dispatch')
            self.assertIn('actual production dispatch checks passed', result)
            print(result.strip())

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler needed')
    def test_actual_factory_and_cache(self):
        path = ROOT / 'tests/webgpu-moe-sum/emit_factory.py'
        spec = importlib.util.spec_from_file_location('moe_factory', path)
        factory = importlib.util.module_from_spec(spec); spec.loader.exec_module(factory)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); source.prepare(self.upstream, out / 'prepared')
            exe = factory.emit(self.upstream, out / 'prepared', out / 'factory')
            shaders = out / 'shaders'; shaders.mkdir()
            subprocess.run([str(exe), str(shaders)], check=True, capture_output=True)
            self.assertEqual(len(list(shaders.glob('moe-*.wgsl'))), 6)
            for shader in shaders.glob('*.wgsl'):
                self.assertIn('@workgroup_size(128)', shader.read_text())
                self.assertNotIn('N_EXPERTS', shader.read_text())

if __name__ == '__main__': unittest.main()
