"""HC source/host checks. These do not execute a GPU or a language model."""
import os
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_hc as hc
import prepare_webgpu_tensor_copy as copy

class HcOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_dedicated_headers_reuse_and_unchanged_upstream(self):
        before={p:copy.digest(self.upstream/p) for p in hc.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);source.prepare(self.upstream,output)
            names=[hc.HEADER_NAME,'ggml-webgpu-hc.hpp']
            stamps={p:(output/p).stat().st_mtime_ns for p in names}
            self.assertEqual(copy.digest(output/hc.HEADER_NAME),hc.HEADER_SHA256)
            self.assertEqual(copy.digest(output/names[1]),hc.OUTPUTS[names[1]])
            self.assertEqual((output/hc.HEADER_NAME).read_text().count('const char* wgsl_'),2)
            self.assertFalse((output/'ggml-wgsl-shaders.hpp').exists())
            source.prepare(self.upstream,output)
            self.assertEqual(stamps,{p:(output/p).stat().st_mtime_ns for p in names})
        self.assertEqual(before,{p:copy.digest(self.upstream/p) for p in before})

    def test_contracts_and_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out'
            for field in ['CONTRACT_INPUTS','INPUTS','OUTPUTS']:
                for name in getattr(hc,field):
                    changed={**getattr(hc,field),name:'0'*64}
                    with self.subTest(field=field,path=name), patch.object(hc,field,changed):
                        with self.assertRaises(ValueError):source.prepare(self.upstream,output)
                    self.assertFalse((output/'ggml-webgpu.cpp').exists())
                    self.assertFalse((output/hc.HEADER_NAME).exists())
            for field in ['PATCH_SHA256','EMBEDDER_SHA256','HEADER_SHA256']:
                with self.subTest(field=field),patch.object(hc,field,'0'*64):
                    with self.assertRaises(ValueError):source.prepare(self.upstream,output)
                self.assertFalse((output/'ggml-webgpu.cpp').exists())

    def test_gpu_checker_extracts_only_reviewed_shader_bytes(self):
        spec = importlib.util.spec_from_file_location('hc_gpu_checker', ROOT / 'tests/webgpu-hc/check_gpu.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source.prepare(self.upstream, output)
            header = (output / hc.HEADER_NAME).read_text()
            codes = checker.extract_shaders(header)
            self.assertEqual(set(codes), {'hc_pre.wgsl', 'hc_post.wgsl'})
            with self.assertRaises(ValueError):
                checker.extract_shaders(header.replace('offset_x:', 'wrong_offset_x:', 1))
            with self.assertRaises(ValueError):
                checker.extract_shaders(header.replace('wgsl_hc_post =', 'wgsl_unknown =', 1))

    def test_canonical_shadow_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'ggml-wgsl-shaders.hpp').write_text('stale')
            with self.assertRaisesRegex(ValueError,'shadow'):source.prepare(self.upstream,output)
            self.assertEqual((output/'ggml-wgsl-shaders.hpp').read_text(),'stale')

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler required')
    def test_actual_dispatch_parameter_and_grid_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = source.prepare(self.upstream, output)
            text = result.read_text()
            snippets = []
            for signature in ['static inline void compute_2d_workgroups(',
                              'static inline uint32_t ggml_webgpu_u32_from_f32(',
                              'static webgpu_encoded_op ggml_webgpu_hc(']:
                start = text.index(signature)
                end = text.index('\n}\n', start) + 2
                snippets.append(text[start:end])
            here = ROOT / 'tests/webgpu-hc'
            path = output / 'dispatch.cpp'
            path.write_text((here / 'dispatch-prefix.cpp').read_text() + '\n'.join(snippets) +
                            (here / 'dispatch-cases.cpp').read_text())
            binary = output / 'dispatch'
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I' + str(output), '-I' + str(self.upstream / 'ggml/include'),
                            str(path), '-o', str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
            self.assertIn('actual dispatcher cases passed', result.stdout)
            print(result.stdout.strip())

    @unittest.skipUnless(shutil.which('g++'),'C++ compiler required')
    def test_actual_admission_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);source.prepare(self.upstream,output)
            exe=output/'admission'
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(output),
                '-I'+str(self.upstream/'ggml/include'),str(ROOT/'tests/webgpu-hc/admission.cpp'),'-o',str(exe)],check=True)
            result=subprocess.run([str(exe)],check=True,text=True,capture_output=True)
            self.assertIn('host admission checks passed',result.stdout)
            print(result.stdout.strip())

if __name__=='__main__':unittest.main()
