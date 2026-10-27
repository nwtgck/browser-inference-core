"""Q8 subgroup source contracts; GPU and real-model execution remain opt-in."""
import importlib.util
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
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_q8_subgroup as q8
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy

spec = importlib.util.spec_from_file_location('q8_subgroup_factory', ROOT / 'tests/webgpu-q8-subgroup/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


class Q8SubgroupOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / q8.BASE / 'ggml-webgpu-shader-lib.hpp').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_reuses_upstream_accumulator_and_preserves_upstream(self):
        before = {name: copy.digest(self.upstream / name) for name in q8.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            factory.prepare(self.upstream, out)
            self.assertEqual(copy.digest(out / 'ggml-webgpu-shader-lib.hpp'), q8.OUTPUTS['ggml-webgpu-shader-lib.hpp'])
            self.assertEqual(copy.digest(out / q8.HEADER_NAME), q8.HEADER_SHA256)
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            embedded = (out / q8.HEADER_NAME).read_text()
            expanded = ''.join(re.findall(r'R"wgsl\((.*?)\)wgsl"', embedded, re.DOTALL))
            for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
                self.assertIn((self.upstream / q8.BASE / 'wgsl-shaders' / name).read_text(), expanded)
            self.assertNotIn('#include "mul_mat_vec_acc.tmpl"', embedded)
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_stage_fails_closed_on_contract_patch_composition_and_output_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source.prepare(self.upstream, root / 'base')
            for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
                for name in getattr(q8, field):
                    with self.subTest(field=field, path=name), tempfile.TemporaryDirectory(dir=root) as tmp:
                        work = Path(tmp)
                        (work / q8.BASE).mkdir(parents=True)
                        for target in q8.INPUTS:
                            shutil.copyfile(root / 'base' / target, work / q8.BASE / target)
                        # When integrated, source.prepare has already applied this
                        # stage; use its source-bound base to test the stage alone.
                        if copy.digest(work / q8.BASE / 'ggml-webgpu-shader-lib.hpp') != q8.INPUTS['ggml-webgpu-shader-lib.hpp']:
                            header = work / q8.BASE / 'ggml-webgpu-shader-lib.hpp'
                            subprocess.run(['git', 'apply', '--reverse', '--no-index', '--include=' + str(q8.BASE / header.name),
                                            str((ROOT / source.PATCH_DIRECTORY / q8.PATCH_NAME).resolve())],
                                           cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                            self.assertEqual(copy.digest(header), q8.INPUTS[header.name])
                        with patch.object(q8, field, {**getattr(q8, field), name: '0' * 64}):
                            with self.assertRaises(ValueError):
                                q8.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            for field in ('PATCH_SHA256', 'HEADER_SHA256'):
                with self.subTest(field=field), tempfile.TemporaryDirectory(dir=root) as tmp:
                    work = Path(tmp)
                    (work / q8.BASE).mkdir(parents=True)
                    for target in q8.INPUTS:
                        shutil.copyfile(root / 'base' / target, work / q8.BASE / target)
                    if copy.digest(work / q8.BASE / 'ggml-webgpu-shader-lib.hpp') != q8.INPUTS['ggml-webgpu-shader-lib.hpp']:
                        subprocess.run(['git', 'apply', '--reverse', '--no-index', '--include=' + str(q8.BASE / 'ggml-webgpu-shader-lib.hpp'),
                                        str((ROOT / source.PATCH_DIRECTORY / q8.PATCH_NAME).resolve())],
                                       cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                    with patch.object(q8, field, '0' * 64), self.assertRaises(ValueError):
                        q8.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))

    def test_checker_rejects_invalid_node_commands_before_preparation(self):
        script = ROOT / 'tests/webgpu-q8-subgroup/check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for command in ('[]', 'null', '{}', '"node"', '[""]', '["node", 7]', '[invalid'):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(script), '--llama-source', str(out), '--work', str(out),
                                             '--webgpu-module', str(out), '--subgroup-size', '4', '--node-command-json', command],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json must', result.stderr)
                    self.assertFalse(out.exists())

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler needed')
    def test_actual_selector_cache_factory_dispatch_and_shader_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            factory.prepare(self.upstream, out / 'prepared')
            exe = factory.build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('actual subgroup selector/cache/factory/dispatch checks passed', result.stdout)
            self.assertEqual(len(list((out / 'shaders').glob('*.wgsl'))), 6)
            for sg in (4, 8, 16, 32, 64):
                text = (out / 'shaders' / f'q8-independent-sg{sg}.wgsl').read_text()
                self.assertIn('@workgroup_size(128)', text)
                self.assertNotIn('workgroupBarrier', text)
                self.assertNotIn('#include', text)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
